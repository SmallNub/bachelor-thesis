import re
import logging
import pandas as pd

from utils import (
    init_logging,
)
import create_docid
from config import (
    DATA_TRAIN_RAW,
    DATA_TRAIN_PROC,
    DATA_EVAL_RAW,
    DATA_EVAL_PROC,
    DATA_TEST_RAW,
    DATA_TEST_PROC,
    DATA_DOCUMENTS,
)

USED_COLUMNS = [
    "document",
    "document_id",
    "question",
    "answer",
    "exe_ans",
    "steps",
    "program",
    "program_re",
    "split"
]


init_logging()
logger = logging.getLogger(__name__)

pattern = re.compile(r'(.*)/page_(\d+)\.pdf')


def convert_filename(path: str):
    """Convert filename into a document id (ABC/2010/page_12.pdf) -> (ABC/2010/12)"""
    match = pattern.search(path)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return path


def convert_table(table: list[list[str]]):
    """Convert nested table structure to csv."""
    header, *rows = table
    df = pd.DataFrame(rows, columns=header)
    return df.to_csv(index=False)


def reformat_data(split: str, input_file_path: str, output_file_path: str = None):
    """Reformat the FinQA dataset and save it."""
    logging.info(f"Reformatting {input_file_path}")
    raw_df = pd.read_json(input_file_path)

    # Unnest the question data
    qa_df = pd.DataFrame(raw_df["qa"].to_dict()).T
    raw_df = pd.concat([raw_df, qa_df], axis="columns")

    # Format into plain text
    raw_df.loc[:, "pre_text"] = raw_df["pre_text"].map(" ".join)
    raw_df.loc[:, "post_text"] = raw_df["post_text"].map(" ".join)
    raw_df.loc[:, "table"] = raw_df["table"].map(convert_table)

    raw_df.loc[:, "document"] = (
        (raw_df["pre_text"] + " ")
        + raw_df["post_text"]
        + ("\nTable:\n" + raw_df["table"])
    )

    # Drop the unused columns
    raw_df["filename"] = raw_df["filename"].apply(convert_filename)
    raw_df.rename(columns={"filename": "document_id"}, inplace=True)

    raw_df["split"] = split

    df = raw_df[USED_COLUMNS]

    if output_file_path is not None:
        df.to_csv(output_file_path, index=False)
        logging.info(f"Saved in {output_file_path}")

    return df


def create_documents_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_file_path: str = None,
):
    """Create document and docid data."""
    logging.info("Creating corpus")
    document_columns = ["document", "document_id", "split"]
    documents_df = pd.concat(
        [
            train_df[document_columns],
            eval_df[document_columns],
            test_df[document_columns],
        ],
        axis="index",
    )
    documents_df.drop_duplicates(inplace=True)
    documents_df.reset_index(drop=True, inplace=True)

    if output_file_path is not None:
        documents_df.to_csv(output_file_path, index=False)
        logging.info(f"Saved in {output_file_path}")

    return documents_df


if __name__ == "__main__":
    train_df = reformat_data("train", DATA_TRAIN_RAW)
    eval_df = reformat_data("eval", DATA_EVAL_RAW)
    test_df = reformat_data("test", DATA_TEST_RAW)

    documents_df = create_documents_data(train_df, eval_df, test_df)
    documents_df = create_docid.main(documents_df)
    documents_df.to_csv(DATA_DOCUMENTS, index=False)

    train_df = create_docid.update_document_id(train_df, documents_df)
    eval_df = create_docid.update_document_id(eval_df, documents_df)
    test_df = create_docid.update_document_id(test_df, documents_df)

    train_df.to_csv(DATA_TRAIN_PROC, index=False)
    eval_df.to_csv(DATA_EVAL_PROC, index=False)
    test_df.to_csv(DATA_TEST_PROC, index=False)
