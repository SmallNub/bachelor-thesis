import os
import sys
import logging
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers import util

from config import DATA_DOCUMENTS, DATA_DOCUMENTS_PSEUDO, DATA_DOCUMENTS_AUG, MODELS_DIR


# NOTE
# Code can run on consumer grade GPU (<1 min)

# TODO
# Filter using CrossEncoders


# ENVIRONMENT SETUP

# Set logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("Script started...")

# Lower precision for higher performance (negligible impact on results)
torch.set_float32_matmul_precision("high")

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

TOP_N = 5

# This balances the query distribution by adding 1 extra to the eval/test splits
# Train split has 1 extra query due to retrieval training
BALANCE_SPLITS = True

logger.info(f"Using model: {MODEL_NAME}")

# Detect number of CPUs and GPUs
num_cpus = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count()))
logger.info(f"Using {num_cpus} CPU core(s)")

num_gpus = torch.cuda.device_count()
logger.info(f"Using {num_gpus} CUDA device(s)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = SentenceTransformer(
    MODEL_NAME,
    cache_folder=MODELS_DIR,
    local_files_only=False,
    device="cuda"
)

logger.info("Filtering...")

# Load dataset
pseudo_df = pd.read_csv(DATA_DOCUMENTS_PSEUDO)
documents_df = pd.read_csv(DATA_DOCUMENTS)

# Add the split column
map_columns = ["document_id", "split"]
pseudo_df = pd.merge(pseudo_df, documents_df[map_columns], on="document_id", how="left")


def get_top_queries(df: pd.DataFrame):
    """Get the top-n unique queries per document with priority for diversity and relevance"""
    doc_to_indices = df.groupby("document").indices
    filtered_rows = []
    global_query_texts = set()
    unique_docs = df["document"].unique().tolist()
    docids = df["document_id"].unique().tolist()

    # Encode the documents and queries
    document_embeddings = model.encode(unique_docs, batch_size=BATCH_SIZE, show_progress_bar=True)
    query_embeddings = model.encode(df["pseudo_query"], batch_size=BATCH_SIZE, show_progress_bar=True)

    for doc_text in tqdm(unique_docs, desc="Selecting queries"):
        indices = doc_to_indices[doc_text]
        doc_idx = unique_docs.index(doc_text)
        doc_embed = document_embeddings[doc_idx]
        docid = docids[doc_idx]
        query_texts = df.iloc[indices]["pseudo_query"].tolist()
        query_embeds = query_embeddings[indices]
        split = df.iloc[indices]["split"].iloc[0]

        # Compute document-query similarities
        doc_sims = util.cos_sim(doc_embed, query_embeds)[0]

        cur_qts = set()
        scored = []
        for qt, qe, d_sim in zip(query_texts, query_embeds, doc_sims):
            # Filter out globally/locally duplicated query texts
            if qt in global_query_texts or qt in cur_qts:
                continue

            cur_qts.add(qt)

            if not scored:
                score = d_sim  # No diversity penalty for first query
            else:
                selected_embeds = np.array([qe_ for _, qe_, _ in scored])
                q_sims = util.cos_sim(qe, selected_embeds)[0]
                avg_q_sim = q_sims.mean().item()
                score = d_sim * (1 - avg_q_sim)  # diversity score

            scored.append((qt, qe, score))  # Store all for later sorting

        # Select top_k by score
        scored.sort(key=lambda x: -x[2])

        # Add 1 extra query for the eval/test split
        top_n = TOP_N
        if BALANCE_SPLITS and split != "train":
            top_n += 1

        top_queries = scored[:top_n]

        if len(top_queries) != top_n:
            logger.warning(
                f"Not enough queries {len(top_queries)} < {top_n} for {docid} in {split}"
            )

        for qt, _, _ in top_queries:
            filtered_rows.append({
                "document_id": docid,
                "document": doc_text,
                "pseudo_query": qt,
                "split": split
            })
            global_query_texts.add(qt)

    return pd.DataFrame(filtered_rows)


# Save results
filtered_df = get_top_queries(pseudo_df)
filtered_df.to_csv(DATA_DOCUMENTS_AUG, index=False)
logger.info(f"Saved to {DATA_DOCUMENTS_AUG}")
