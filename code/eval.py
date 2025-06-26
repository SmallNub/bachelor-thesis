import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
from peft import PeftModel, PeftConfig

from utils import (
    enable_tf32,
    init_logging,
    detect_resources,
    set_seed,
    ResourceMonitorCallback
)
from metrics import compute_metrics
from process_data import load_data, DynamicDataset
from model import load_tokenizer, load_model
from config import MODELS_DIR, SPLITS


# NOTE
# Results are saved in test_eval.ipynb
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

USE_COT = False
USE_AUG = True  # Ignored, but should still be True

BATCH_SIZE = 256
N_EXAMPLES = 0

SEED = 42
set_seed(SEED)

SAVE_DIR = os.path.join(MODELS_DIR, "finqa_base_10_full")
logger.info(f"Input location: {SAVE_DIR}")

# Detect number of CPUs and GPUs
num_cpus, num_gpus = detect_resources(logger)

# Load trained model

peft_config = PeftConfig.from_pretrained(SAVE_DIR)
model_name = peft_config.base_model_name_or_path

tokenizer = load_tokenizer(model_name)
model = load_model(model_name, tokenizer, USE_COT)

model = PeftModel.from_pretrained(model, SAVE_DIR)

# Merge LoRA into model for decreased latency
model.merge_and_unload()

# Prepare data
_, data = load_data(USE_AUG, DEBUG, DEBUG_SIZE)

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


def evaluate_model(model, dataset):
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

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader):
        # Move data to device
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Get predictions
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                max_new_tokens=tokenizer.model_max_length,
                num_beams=10,
                early_stopping=True,
            )

        # Decode tokens
        decoded_preds = model.base_model.decode_tokens(generated_ids.detach().cpu().numpy())
        decoded_labels = model.base_model.decode_tokens(labels.detach().cpu().numpy())

        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    metrics, _ = compute_metrics(all_preds, all_labels, USE_COT)

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
