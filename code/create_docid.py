import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from config import (
    DATA_DOCUMENTS,
    MODELS_DIR,
    SEPARATOR,
    KEYWORDS_NGRAM_SIZE,
    KEYWORDS_DIVERSITY,
    KEYWORDS_TOP_N,
    USE_COMPANY_YEAR_IN_DOCID,
)


logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


def main(documents_df: pd.DataFrame):
    logger.info(f"Loading model: {MODEL_NAME}")

    # Load model and data
    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=MODELS_DIR,
        local_files_only=False,
        device="cuda"
    )
    kw_model = KeyBERT(model)

    logger.info("Model loaded")
    logger.info("Extracting keywords...")

    # Extract diverse bigram keywords
    keywords_list = kw_model.extract_keywords(
        documents_df["document"],
        keyphrase_ngram_range=(KEYWORDS_NGRAM_SIZE, KEYWORDS_NGRAM_SIZE),
        stop_words="english",
        use_mmr=True,
        diversity=KEYWORDS_DIVERSITY,
        top_n=KEYWORDS_TOP_N
    )
    logger.info("Keywords extracted")
    logger.info("Creating new document ids...")

    # Create new document ids
    new_docids = []
    scores = []

    for keywords, docid in zip(keywords_list, documents_df["document_id"], strict=True):
        new_docid = ""

        # Use the old document id as a base
        if USE_COMPANY_YEAR_IN_DOCID:
            company, year, _ = docid.split("/")
            new_docid = f"{company}{SEPARATOR}{year}"

        # Concatenate the keywords using hyphens
        total_score = 0
        for keyword, score in keywords:
            new_docid += f"{SEPARATOR}{SEPARATOR.join(keyword.split())}"
            total_score += score

        new_docids.append(new_docid)
        scores.append(total_score)

    # Replace old document ids
    documents_df.rename(columns={"document_id": "old_document_id"}, inplace=True)
    documents_df["document_id"] = new_docids
    documents_df["score"] = scores

    if not documents_df["document_id"].is_unique:
        logger.error("NOT ALL DOCUMENT IDS ARE UNIQUE")

    logger.info("Done")
    return documents_df


if __name__ == "__main__":
    documents_df = pd.read_csv(DATA_DOCUMENTS)
    documents_df = main(documents_df)
    documents_df.to_csv(DATA_DOCUMENTS, index=False)
