import re

from config import SEPARATOR, USE_COMPANY_YEAR_IN_DOCID, DOCID_SIZE


# Depends on CoT pattern
cot_pattern = re.compile(r"final answer is (\w+)\.")


def extract_docid(text: str, use_cot=False, is_label=False) -> str | None:
    """Attempts to extract the docid from the text. The resulting docid can be incorrect."""
    # If no separator is found, the text likely does not contain a docid
    if SEPARATOR not in text:
        if is_label:
            raise ValueError("Document id not found in label")
        return None

    # Without CoT, the text should correspond immediately to the docid
    if not use_cot:
        return text.strip()

    # Use the CoT pattern to extract the docid
    match = cot_pattern.search(text)
    if match:
        extracted_docid = match.group(1)
        return extracted_docid.strip()

    if is_label:
        raise ValueError("Document id not found in label")

    # Failed to match the CoT pattern
    return None


def deconstruct_docid(docid: str, is_label=False) -> list[str] | None:
    """Attempts to deconstruct the docid."""
    parts = docid.split(SEPARATOR)

    if len(parts) != DOCID_SIZE:
        return None

    return parts


def compare_parts(pred_parts: list[str], label_parts: list[str]):
    for i, (pred_part, label_part) in enumerate(zip(pred_parts, label_parts)):
        if USE_COMPANY_YEAR_IN_DOCID:
            pass
            


def contains_whole_word(word: str, text: str):
    """Checks if the whole word is inside the text, case-insensitive."""
    word = word.strip()
    text = text.strip()

    # Regex pattern matching exact whole words, case-insensitive
    pattern = rf"(?i)\b{re.escape(word)}\b"

    return bool(re.search(pattern, text))


def compute_match_accuracy(preds: list[str], labels: list[str]):
    """Computes the match accuracy, is the label inside the prediction?"""
    matches = [
        contains_whole_word(label, pred)
        for pred, label in zip(preds, labels)
    ]
    accuracy = sum(matches) / len(matches)
    return accuracy


# def compute_exact_match_accuracy(pred: str, label: str):
#     """Computes the match accuracy, is the label inside the prediction?"""
#     matches = [
#         contains_whole_word(label, pred)
#         for pred, label in zip(preds, labels)
#     ]
#     accuracy = sum(matches) / len(matches)
#     return accuracy


# def compute_metrics(preds: list[str], labels: list[str], use_cot=False):
#     """Computes various metrics and return a dictionary of metrics"""
#     for pred, label in zip(preds, labels, strict=True):
#         label_docid = extract_docid(label, use_cot=use_cot, is_label=False)
#         label_parts = deconstruct_docid(label_docid, is_label=False)
        
#         pred_docid = extract_docid(pred, use_cot=use_cot, is_label=False)
#         pred_parts = deconstruct_docid(pred_docid, is_label=False)
        
        
