import os
import sys
import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
from peft import PeftModel

from utils import (
    enable_tf32,
    init_logging,
    detect_resources,
    set_seed,
    ResourceMonitorCallback
)
from metrics import compute_metrics, extract_docid, compute_ir_metrics
from process_data import load_data, DynamicDataset, DocidTrie, get_prefix_allowed_tokens_fn
from model import load_tokenizer, load_model, WeightedLossT5
from config import MODELS_DIR, SPLITS, DATA_DOCUMENTS_AUG


# NOTE
# Will use up nearly all the available VRAM on an H100
# Should complete under 10 mins

# ENVIRONMENT SETUP

enable_tf32()

home_dir = os.getenv("HOME")
job_id = os.getenv("SLURM_JOBID")

init_logging(home_dir, job_id)
logger = logging.getLogger(__name__)

resource_monitor = ResourceMonitorCallback(logger)

# Enable debug to drastically reduce values
DEBUG = False
DEBUG_INPUTS = False
DEBUG_SIZE = 4
USE_LORA = True

USE_COT = False
USE_AUG = True  # Ignored, but should still be True

BATCH_SIZE = 256
N_EXAMPLES = 2

NUM_BEAMS = 10
NUM_RETURN_SEQUENCES = 10
USE_CONSTRAINTS = True

SEED = 42
set_seed(SEED)

CHECKPOINT_NAME = "reasongr_cot"
MODEL_NAME = "google/flan-t5-base"
logger.info(f"Using model: {MODEL_NAME}")

SAVE_DIR = os.path.join(MODELS_DIR, CHECKPOINT_NAME)
logger.info(f"Input location: {SAVE_DIR}")

# Detect number of CPUs and GPUs
num_cpus, num_gpus = detect_resources(logger)

# Load trained model
tokenizer = load_tokenizer(MODEL_NAME)

if USE_LORA:
    model = load_model(MODEL_NAME, tokenizer, USE_COT)
    model = PeftModel.from_pretrained(model, SAVE_DIR)

    # Merge LoRA into model for decreased latency
    model.merge_and_unload()
else:
    model = load_model(SAVE_DIR, tokenizer, USE_COT)

# Prepare data
_, data = load_data(USE_AUG, DEBUG, DEBUG_SIZE)

documents = pd.read_csv(DATA_DOCUMENTS_AUG)

eval_data = {}
for split in SPLITS:
    eval_data[split] = DynamicDataset(
        data[split],
        tokenizer=tokenizer,
        documents_ds=None,
        format_id=0,
        n_examples=N_EXAMPLES,
        seed=SEED,
        indexing=not USE_AUG,
        use_cot=USE_COT,
        debug=DEBUG_INPUTS,
    )

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def evaluate_model(model: WeightedLossT5, dataset):
    """Evaluate the model with a dataset."""
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
        num_workers=0 if DEBUG else num_cpus,
        prefetch_factor=None if DEBUG else 2,
        pin_memory=False if DEBUG else True,
    )

    # Initialize Trie with your corpus docids
    all_docids = documents["document_id"].unique().tolist()
    trie_manager = DocidTrie(all_docids, tokenizer)

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader):
        # Move data to device
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        constraints_fn = get_prefix_allowed_tokens_fn(
            tokenizer,
            trie_manager,
            use_cot=USE_COT
        ) if USE_CONSTRAINTS else None

        # Get predictions
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                max_new_tokens=tokenizer.model_max_length,
                num_beams=NUM_BEAMS,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                prefix_allowed_tokens_fn=constraints_fn,
                early_stopping=True,
            )

        # Decode tokens
        decoded_preds = model.decode_tokens(generated_ids.detach().cpu().numpy())
        decoded_labels = model.decode_tokens(labels.detach().cpu().numpy())

        # Reshape to (batch_size, 10, seq_len)
        decoded_outputs = []
        for i in range(input_ids.shape[0]):
            decoded_outputs.append(decoded_preds[i * NUM_RETURN_SEQUENCES:(i + 1) * NUM_RETURN_SEQUENCES])

        all_preds.extend(decoded_outputs)
        all_labels.extend(decoded_labels)

    # Only use the top 1 prediction
    metrics, _ = compute_metrics([ranking[0] for ranking in all_preds], all_labels, USE_COT)

    # Extract docids for IR metrics
    all_preds_docids = []
    for ranking in all_preds:
        ranking_docids = []
        for pred in ranking:
            docid = extract_docid(pred, use_cot=USE_COT, is_label=False)
            ranking_docids.append(docid if docid is not None else "__NULL__")
        all_preds_docids.append(ranking_docids)

    all_label_docids = [
        extract_docid(label, use_cot=USE_COT, is_label=True)
        for label in all_labels
    ]

    ir_metrics = compute_ir_metrics(all_preds_docids, all_label_docids)
    metrics.update(ir_metrics)
    return metrics


if __name__ == "__main__":
    logger.info("Starting Evaluation")
    resource_monitor.log_resources()
    results = {}

    for split in SPLITS:
        logger.info(f"Evaluating split: {split}")
        results[split] = {
            **evaluate_model(model, eval_data[split])
        }
        resource_monitor.log_resources()

    logger.info("Evaluation Completed")
    print(results)

    logger.info("Script Finished")
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
