import re


# Depends on CoT pattern
cot_pattern = re.compile(r"final answer is (\w+)\.")


def extract_docid(text: str):
    """Extracts the docid from a CoT pattern."""
    match = cot_pattern.search(text)
    if match:
        extracted_docid = match.group(1)
        return extracted_docid
    else:
        raise ValueError("Document id not found")


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
