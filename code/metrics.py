import re
import string
import numpy as np

from config import SEPARATOR, DOCID_SIZE


# Penalty multipliers for the loss (minus base)
# Base means 1 or no penalty, loss * (base + penalty)
# No penalty = penalty = 0

# Penalty is linearly scaled from 0 to 100% for the first epochs
WARMUP_EPOCHS = 10

# The maximum accumulated penalty (including base and after the log)
MAXIMUM_PENALTY = 2

# Penalty for exact match
# Due to other penalties, not matching exactly already receives a penalty
PENALTY_EXACT_MATCH = 0.05

# Penalty for incorrect parts
# It is linearly scaled up to this value depending the amount of incorrect parts
PENALTY_PART_MATCH = 2.0

# Penalty for incorrect set of parts
# Due to part match, not matching parts already receives a penalty
# It is linearly scaled up to this value depending the amount of incorrect parts
PENALTY_SET_MATCH = 0.5

# Penalty for incorrect amount of parts
# Capped to a difference of +/-MAXIMUM_STRUCTURE_DIFF% of the amount of parts
# Due to part match, smaller structures already receive higher penalties
# Penalty = diff_perc * penalty_score
PENALTY_STRUCTURE_SCORE = 0.5
MAXIMUM_STRUCTURE_DIFF = 0.3  # Make this value extremely high for practically no max

# Combines every penalty at their maximum
# Calculation should be changed if MAXIMUM_STRUCTURE_DIFF is too high
MAXIMUM_COMBINED_PENALTY = (
    PENALTY_EXACT_MATCH +
    PENALTY_PART_MATCH +
    PENALTY_SET_MATCH +
    PENALTY_STRUCTURE_SCORE * MAXIMUM_STRUCTURE_DIFF
)

# Penalty for missing the docid entirely
# It is a ratio of the MAXIMUM_COMBINED_PENALTY
PENALTY_MISSING = 1.0


# NOTE: regex is not robust
# Depends on CoT pattern
# 1. Look for anchor keywords: answer, answers, reply, replies
# 2. Look for flexible verbs: forms of "to be" and "to have"
# 3. Capture the docid: [a-zA-Z0-9\-]+
cot_pattern = re.compile(
    r"(?:answer|answers|reply|replies)\s+(?:be|being|am|is|are|was|were|having|have|has|had)\s+(\S+)",
    re.IGNORECASE
)


def extract_docid(text: str, use_cot=False, is_label=False) -> str | None:
    """Attempts to extract the docid from the text. The resulting docid can be incorrect."""
    # If no separator is found, the text does not contain a parsable docid
    if SEPARATOR not in text:
        if is_label:
            raise ValueError("Document id not found in label")
        return None

    # Clean up special tokens
    text = text.replace("</s>", "").replace("<pad>", "").strip()

    # Without CoT, the text should correspond immediately to the docid
    if not use_cot:
        return text.strip()

    # Use the CoT pattern to extract the docid
    match = cot_pattern.search(text)
    if match:
        extracted_docid = match.group(1)
        return extracted_docid.strip(string.punctuation)

    if is_label:
        raise ValueError("Document id not found in label")

    # Failed to match the CoT pattern, use fallback: last word
    words = text.split()
    return words[-1].strip(string.punctuation) if words else None


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
    penalty = (1 - accuracy) * PENALTY_EXACT_MATCH
    return accuracy, penalty


def compute_part_match_accuracy(pred_parts: list[str], label_parts: list[str]):
    """
    Computes the part match accuracy and penalty by doing exact match between parts.\\
    Missing parts are considered wrong, overflow is discarded.
    """
    matches = [pred == label for pred, label in zip(pred_parts, label_parts)]
    accuracy = sum(matches) / len(label_parts)
    penalty = (1 - accuracy) * PENALTY_PART_MATCH
    return accuracy, penalty


def compute_set_match_accuracy(pred_parts: list[str], label_parts: list[str]):
    """
    Computes the set match accuracy and penalty by doing set match between parts.\\
    Duplicates inside predictions or labels are not used.\\
    Missing parts are considered wrong, overflow is used.
    """
    pred_set = set(pred_parts)
    label_set = set(label_parts)
    match_set = pred_set.intersection(label_set)
    accuracy = len(match_set) / len(label_set)
    penalty = (1 - accuracy) * PENALTY_SET_MATCH
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
        "set_match_accuracy": 0,
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
            penalty = 1 + MAXIMUM_COMBINED_PENALTY * PENALTY_MISSING
            penalties.append(penalty)
            # Cannot compute other metrics without docid
            continue

        pred_parts = deconstruct_docid(pred_docid)

        label_docid = extract_docid(label, use_cot=use_cot, is_label=True)
        label_parts = deconstruct_docid(label_docid)

        # Compute metrics
        em_acc, p1 = compute_exact_match_accuracy(pred_docid, label_docid)
        pm_acc, p2 = compute_part_match_accuracy(pred_parts, label_parts)
        sm_acc, p3 = compute_set_match_accuracy(pred_parts, label_parts)
        s_score, p4 = compute_structure_score(pred_parts)

        metrics["exact_match_accuracy"] += em_acc
        metrics["part_match_accuracy"] += pm_acc
        metrics["set_match_accuracy"] += sm_acc
        if s_score > 0:
            metrics["structure_score_norm"] += 1
            metrics["structure_score_pos"] += s_score
        elif s_score < 0:
            metrics["structure_score_norm"] -= 1
            metrics["structure_score_neg"] += abs(s_score)

        # Base + penalty, scaled by log
        penalty = 1 + p1 + p2 + p3 + p4
        penalties.append(penalty)

    # Save uncapped penalties
    penalties = np.array(penalties)
    metrics["penalty_uncapped"] = penalties.sum()

    # Log scale and limit the penalty to prevent extremes
    penalties = np.log(penalties) + 1
    penalties = np.clip(penalties, 0, MAXIMUM_PENALTY)
    metrics["penalty_capped"] = penalties.sum()

    # Scale the penalty with the warmup
    if current_epoch >= 0 and current_epoch < WARMUP_EPOCHS:
        penalties = 1 + (penalties - 1) * min(current_epoch / WARMUP_EPOCHS, 1)
    metrics["penalty_scaled"] = penalties.sum()

    metrics = {k: v / len(preds) for k, v in metrics.items()}

    return metrics, penalties


def calculate_mrr(target, predictions):
    """Mean Reciprocal Rank: 1/rank of the first correct match."""
    for i, p in enumerate(predictions):
        if p == target:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(target, predictions):
    """
    Normalized Discounted Cumulative Gain.
    Since there is only 1 correct target in retrieval tasks,
    IDCG is always 1.0 (perfect ranking puts the item at index 0).
    """
    for i, p in enumerate(predictions):
        if p == target:
            # log2(i + 2) because rank is i+1, and formula is log2(rank + 1)
            return 1.0 / np.log2(i + 2)
    return 0.0


def compute_ir_metrics(all_top_k_preds, all_labels):
    """
    Computes standard IR metrics: Hits@1, Hits@10, MRR, NDCG@10.
    """
    mrr_scores = []
    ndcg_scores = []
    hits_at_1 = 0
    hits_at_10 = 0

    total = len(all_labels)

    for target, preds in zip(all_labels, all_top_k_preds):
        # Metrics are defined for top 10
        preds = preds[:10]

        # Hits@k
        if target in preds[:1]:
            hits_at_1 += 1
        if target in preds:
            hits_at_10 += 1

        # Ranking Metrics
        mrr_scores.append(calculate_mrr(target, preds))
        ndcg_scores.append(calculate_ndcg(target, preds))

    return {
        "hits@1": hits_at_1 / total,
        "hits@10": hits_at_10 / total,
        "mrr": float(np.mean(mrr_scores)),
        "ndcg@10": float(np.mean(ndcg_scores)),
    }
