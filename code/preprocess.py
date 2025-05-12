import pandas as pd

from config import (
    DATA_TRAIN_RAW,
    DATA_TRAIN_PROC,
    DATA_VALID_RAW,
    DATA_VALID_PROC,
    DATA_TEST_RAW,
    DATA_TEST_PROC,
    DATA_DOCUMENTS,
    USED_COLUMNS,
)


def convert_table(table: list[list[str]]):
    """Convert nested table structure to csv."""
    header, *rows = table
    df = pd.DataFrame(rows, columns=header)
    return df.to_csv(index=False)


def reformat_data(input_file_path: str, output_file_path: str):
    """Reformat the FinQA dataset."""
    raw_df = pd.read_json(input_file_path)

    # Unnest the question data
    qa_df = pd.DataFrame(raw_df["qa"].to_dict()).T
    raw_df = pd.concat([raw_df, qa_df], axis="columns")

    # Format into plain text
    raw_df.loc[:, "pre_text"] = raw_df["pre_text"].map(" ".join)
    raw_df.loc[:, "post_text"] = raw_df["post_text"].map(" ".join)
    raw_df.loc[:, "table"] = raw_df["table"].map(convert_table)

    raw_df.loc[:, "full_text"] = (
        raw_df["pre_text"]
        + raw_df["post_text"]
        + "\nThis is a table:\n"
        + raw_df["table"]
    )

    # Drop the unused columns
    df = raw_df[USED_COLUMNS]

    df.to_csv(output_file_path, index=False)
    return df


def create_documents_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_file_path: str,
):
    """Create document and docid data."""
    document_columns = ["full_text", "id"]
    documents_df = pd.concat(
        [
            train_df[document_columns],
            valid_df[document_columns],
            test_df[document_columns],
        ],
        axis="index",
    )

    documents_df.to_csv(output_file_path, index=False)
    return documents_df


if __name__ == "__main__":
    train_df = reformat_data(DATA_TRAIN_RAW, DATA_TRAIN_PROC)
    valid_df = reformat_data(DATA_VALID_RAW, DATA_VALID_PROC)
    test_df = reformat_data(DATA_TEST_RAW, DATA_TEST_PROC)

    documents_df = create_documents_data(train_df, valid_df, test_df, DATA_DOCUMENTS)
