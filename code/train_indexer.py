import os
import sys
import signal
import logging
import multiprocessing
import json
import torch
import torch.distributed as dist
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb

from config import DATA_DOCUMENTS, MODELS_DIR


# ENVIRONMENT SETUP

# Set logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

logger.info("Script started...")

BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
EPOCHS = 40

MODEL_NAME = "google/flan-t5-base"
logger.info(f"Using model: {MODEL_NAME}")

OUTPUT_DIR = os.path.join(MODELS_DIR, "finqa_indexer")
logger.info(f"Output location: {OUTPUT_DIR}")

with open("code/ds_config.json", "r", encoding="utf-8") as f:
    ds_config = json.load(f)

# Detect number of CPUs and GPUs
num_cpus = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count()))
logger.info(f"Using {num_cpus} CPU core(s)")

num_gpus = torch.cuda.device_count()
logger.info(f"Using {num_gpus} CUDA device(s)")


# DATA PREPROCESSING

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODELS_DIR,
    use_fast=True
)


def preprocess_fn(example):
    prompt = f"Retrieve the document id for this document: {example['document']}"

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=4096,
    )
    targets = tokenizer(
        text_target=example["document_id"],
        truncation=True,
        max_length=32,
    )
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": targets.input_ids,
    }


# Process data
raw_ds = load_dataset("csv", data_files=DATA_DOCUMENTS, split="train")
tokenized_ds = raw_ds.map(
    preprocess_fn,
    remove_columns=raw_ds.column_names,
    num_proc=num_cpus,
)


# MODEL SETUP

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODELS_DIR,
    torch_dtype="auto",
    local_files_only=True,  # Change for first time downloads
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
)

# Fix for gradient checkpoints
model.config.use_cache = False
model.enable_input_require_grads()

model = prepare_model_for_kbit_training(model)

# LoRA config (QLoRA + OLoRA)
lora_config = LoraConfig(
    init_lora_weights="olora",
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)

# Print model statistics
# Code from model.print_trainable_parameters()
trainable_params, all_param = model.get_nb_trainable_parameters()

logger.info(
    f"trainable params: {trainable_params:,d} || "
    f"all params: {all_param:,d} || "
    f"trainable%: {100 * trainable_params / all_param:.4f}"
)

logger.info(f"Memory footprint: {model.get_memory_footprint():,}")

# Specialized LoRA optimizer
optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=bnb.optim.Adam8bit,
    lr=LEARNING_RATE,
    loraplus_lr_ratio=16,
)
scheduler = None

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def compute_metrics(eval_preds):
    """Compute substring match accuracy."""
    preds, labels = eval_preds

    # Replace pytorch pad token id with tokenizer token id
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute substring match accuracy
    matches = [
        int(label.lower().strip() in pred.lower().strip())
        for pred, label in zip(decoded_preds, decoded_labels)
    ]
    accuracy = sum(matches) / len(matches)

    return {"substring_match_accuracy": accuracy}


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    eval_accumulation_steps=ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    optim="adamw_bnb_8bit",
    bf16=True,
    deepspeed=ds_config,
    report_to="tensorboard",
    logging_strategy="epoch",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_on_start=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_substring_match_accuracy",
    greater_is_better=False,
    predict_with_generate=True,
    label_names=["labels"],
    gradient_checkpointing=True,  # Large memory impact
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_drop_last=True,
    dataloader_num_workers=num_cpus,
    dataloader_prefetch_factor=2,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=False,  # Causes hanging at the end
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    eval_dataset=tokenized_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

if __name__ == "__main__":
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
    finally:
        try:
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("Saved succesfully")
        except Exception as e:
            logger.error(f"Saving failed: {e}")

        if dist.is_initialized():
            dist.destroy_process_group()

    logger.info("Script finished")

    # Send SIGTERM to the current process
    os.kill(os.getpid(), signal.SIGTERM)
