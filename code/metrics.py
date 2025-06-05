import re
import numpy as np

from config import SEPARATOR, DOCID_SIZE


# Penalty multipliers for the loss (minus base)
# Base means 1 or no penalty, loss * (base + penalty)
# No penalty = penalty = 0

# Penalty is linearly scaled from 0 to 100% for the first epochs
WARMUP_EPOCHS = 10

# The maximum accumulated penalty (including base)
MAXIMUM_PENALTY = 1.5

# Penalty for missing the docid entirely (including base)
# It is capped by MAXIMUM_PENALTY
PENALTY_MISSING = MAXIMUM_PENALTY

# Penalty for exact match
# Due to other penalties, not matching exactly already receives a penalty
PENALTY_EXACT_MATCH = 0.02

# Penalty for incorrect parts
# It is linearly scaled up to this value depending the amount of incorrect parts
PENALTY_PART_MATCH = 0.3

# Penalty for incorrect amount of parts
# Capped to a difference of +/-MAXIMUM_STRUCTURE_DIFF% of the amount of parts
# Due to part match, smaller structures already receive higher penalties
# Penalty = diff_perc * penalty_score
PENALTY_STRUCTURE_SCORE = 1.0
MAXIMUM_STRUCTURE_DIFF = 0.5  # Make this value extremely high for practically no max


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


def deconstruct_docid(docid: str) -> list[str]:
    """Attempts to deconstruct the docid."""
    parts = [part.strip() for part in docid.split(SEPARATOR)]
    return parts


def compute_exact_match_accuracy(pred: str, label: str):
    """
    Computes the exact match accuracy and penalty.\\
    0 = no match, 1 = exact match
    """
    accuracy = pred == label
    penalty = accuracy * PENALTY_EXACT_MATCH
    return accuracy, penalty


def compute_part_match_accuracy(pred_parts: list[str], label_parts: list[str]):
    """
    Computes the part match accuracy and penalty by doing exact match between parts.\\
    Missing parts are considered wrong.
    """
    matches = [pred == label for pred, label in zip(pred_parts, label_parts)]
    accuracy = sum(matches) / len(label_parts)
    penalty = accuracy * PENALTY_PART_MATCH
    return accuracy, penalty


def compute_structure_score(parts: list[str]):
    """
    Computes the structure score and penalty.\\
    Negative = smaller, 0 = correct size, positive = bigger
    """
    size_diff = (len(parts) - DOCID_SIZE) / DOCID_SIZE
    scale = min(abs(size_diff), MAXIMUM_STRUCTURE_DIFF)
    penalty = scale * PENALTY_STRUCTURE_SCORE
    return size_diff, penalty


def compute_metrics(
    preds: list[str],
    labels: list[str],
    use_cot: bool = False,
    current_epoch: int = -1
):
    """Computes various metrics and return a dictionary of metrics and an array of penalties."""
    metrics = {
        "penalty_scaled": 0,
        "penalty_capped": 0,
        "penalty_uncapped": 0,
        "missing": 0,
        "exact_match_accuracy": 0,
        "part_match_accuracy": 0,
        "structure_score_norm": 0,
        "structure_score_pos": 0,
        "structure_score_neg": 0,
    }
    penalties = []

    for pred, label in zip(preds, labels, strict=True):
        pred_docid = extract_docid(pred, use_cot=use_cot, is_label=False)

        if pred_docid is None:
            metrics["missing"] += 1
            metrics["structure_score_norm"] -= 1
            metrics["structure_score_neg"] += 1
            penalties.append(PENALTY_MISSING)
            # Cannot compute other metrics without docid
            continue

        pred_parts = deconstruct_docid(pred_docid)

        label_docid = extract_docid(label, use_cot=use_cot, is_label=True)
        label_parts = deconstruct_docid(label_docid)

        # Compute metrics
        em_acc, p1 = compute_exact_match_accuracy(pred_docid, label_docid)
        pm_acc, p2 = compute_part_match_accuracy(pred_parts, label_parts)
        s_score, p3 = compute_structure_score(pred_parts)

        metrics["exact_match_accuracy"] += em_acc
        metrics["part_match_accuracy"] += pm_acc
        if s_score > 0:
            metrics["structure_score_norm"] += 1
            metrics["structure_score_pos"] += s_score
        elif s_score < 0:
            metrics["structure_score_norm"] -= 1
            metrics["structure_score_neg"] += abs(s_score)

        # Base + penalty
        penalty = 1 + p1 + p2 + p3
        penalties.append(penalty)

    # Save uncapped penalties
    penalties = np.array(penalties)
    metrics["penalty_uncapped"] = penalties.sum()

    # Limit the penalty to prevent extremes
    np.clip(penalties, 0, MAXIMUM_PENALTY, out=penalties)
    metrics["penalty_capped"] = penalties.sum()

    # Scale the penalty with the warmup
    if current_epoch >= 0 and current_epoch < WARMUP_EPOCHS:
        penalties = 1 + (penalties - 1) * min(current_epoch / WARMUP_EPOCHS, 1)
    metrics["penalty_scaled"] = penalties.sum()

    metrics = {k: v / len(preds) for k, v in metrics.items()}

    return metrics, penalties
