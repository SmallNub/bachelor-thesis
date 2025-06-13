import os
import logging
import multiprocessing
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader

from utils import (
    init_logging,
    enable_tf32,
    set_seed,
)
from config import DATA_DOCUMENTS, DATA_DOCUMENTS_PSEUDO, MODELS_DIR


# NOTE
# Code can run on consumer grade GPU (20 mins)
# It may generate duplicate queries if documents are similar

# TODO
# Rolling window or chunking instead of truncating
# Use target sentences as documents (guessing required for eval + test)


# ENVIRONMENT SETUP

# Set logging
init_logging()
logger = logging.getLogger(__name__)
logger.info("Script started...")

enable_tf32()

# Configuration
MODEL_NAME = "doc2query/all-with_prefix-t5-base-v1"
NUM_QUERIES = 20
MAX_LENGTH = 64
BATCH_SIZE = 4

SEED = 42

logger.info(f"Using model: {MODEL_NAME}")

# Detect number of CPUs and GPUs
num_cpus = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count()))
logger.info(f"Using {num_cpus} CPU core(s)")

num_gpus = torch.cuda.device_count()
logger.info(f"Using {num_gpus} CUDA device(s)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODELS_DIR,
    use_fast=True,
)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    cache_dir=MODELS_DIR,
    local_files_only=False,
    low_cpu_mem_usage=True,
).to(device)


# Custom Dataset
class DocumentDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "document_id": row["document_id"],
            "prompt": "text2query: " + row["document"],
            "document": row["document"]
        }


# Collate function for batching
def collate_fn(batch):
    texts = [item["prompt"] for item in batch]
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return {
        "document_id": [item["document_id"] for item in batch],
        "document": [item["document"] for item in batch],
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }


# Load dataset
df = pd.read_csv(DATA_DOCUMENTS)
dataset = DocumentDataset(df)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=False,
    pin_memory=True,
)

logger.info("Starting generation...")

# Output storage
output_rows = []

# Run generation
model.eval()
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Generating pseudo-queries"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        docids = batch["document_id"]
        texts = batch["document"]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_return_sequences=NUM_QUERIES,
            do_sample=True,
            top_k=100,
            top_p=0.92,
            temperature=1.3,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Associate outputs back to docids/texts
        for i, (docid, text) in enumerate(zip(docids, texts)):
            for j in range(NUM_QUERIES):
                index = i * NUM_QUERIES + j
                query = decoded[index]
                output_rows.append({
                    "document_id": docid,
                    "document": text,
                    "pseudo_query": query
                })

# Save results
pd.DataFrame(output_rows).to_csv(DATA_DOCUMENTS_PSEUDO, index=False)
logger.info(f"Saved to {DATA_DOCUMENTS_PSEUDO}")
