import os
import sys
import logging
import traceback
import multiprocessing
import json
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb

from process_data import build_process_fn, DynamicDataset
from custom_trainer import WeightedLossT5, CustomTrainer, EpochTrackerCallback
from config import (
    DATA_DOCUMENTS,
    DATA_DOCUMENTS_AUG,
    DATA_TRAIN_PROC,
    DATA_EVAL_PROC,
    DATA_TEST_PROC,
    MODELS_DIR,
)


# NOTE
# Code is optimized for 1 H100 GPU with this model size
# bf16 is used which might not work on some GPUs

# TODO
# Fix CoT


# ENVIRONMENT SETUP

# Set logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# from transformers.utils import logging as hf_logging
# hf_logging.set_verbosity_info()

logger.info("Script started...")

# Enable debug to drastically reduce values
DEBUG = True
DEBUG_SIZE = 4  # Using sampling will double it
SPLITS = ["train", "eval", "test"]

USE_COT = False

# Use only pseudo-queries instead of documents
# Prompt will be truncated when indexing normally
USE_AUG = True

SEED = 42
BATCH_SIZE = min(4, DEBUG_SIZE) if DEBUG else 16
ACCUMULATION_STEPS = 1 if DEBUG else 2
LEARNING_RATE = 4e-4
EPOCHS = 100

MODEL_NAME = "google/flan-t5-base"
logger.info(f"Using model: {MODEL_NAME}")

# This should correspond to the same name inside train.sh
OUTPUT_DIR = os.path.join(MODELS_DIR, "finqa_full_base")
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

# Process documents for indexing
if USE_AUG:
    # Augmented documents use pseudo-queries which should use the retrieval task instead
    input_key = "pseudo_query"
    document_task = False
    document_file = DATA_DOCUMENTS_AUG
else:
    # Traditional documents should use the indexing task
    input_key = "document"
    document_task = True
    document_file = DATA_DOCUMENTS
    raise NotImplementedError()


raw_documents_ds = load_dataset("csv", data_files=document_file, split="train")
documents_ds = raw_documents_ds.map(
    build_process_fn(tokenizer, input_key, document_task, USE_COT),
    remove_columns=raw_documents_ds.column_names,
    num_proc=num_cpus,
    batched=True,
    batch_size=max(1, len(raw_documents_ds) // num_cpus)
)

# Process data for retrieval (train, valid, test)
file_mapping = {
    "train": DATA_TRAIN_PROC,
    "eval": DATA_EVAL_PROC,
    "test": DATA_TEST_PROC,
}

# Process queries for retrieval
raw_data_ds = load_dataset("csv", data_files=file_mapping)
tokenized_ds = raw_data_ds.map(
    build_process_fn(tokenizer, "question", False, USE_COT),
    remove_columns=raw_data_ds["train"].column_names,
    num_proc=num_cpus,
    batched=True,
    batch_size=max(1, len(raw_data_ds["eval"]) // num_cpus)
)

# Merge the indexing stage into the train split
# tokenized_ds["train"] = concatenate_datasets([tokenized_ds["train"], documents_ds])

# Reduce data size for all splits
if DEBUG:
    raw_documents_ds = raw_documents_ds.select(range(DEBUG_SIZE))

    for split in SPLITS:
        tokenized_ds[split] = tokenized_ds[split].select(range(DEBUG_SIZE))
        raw_data_ds[split] = raw_data_ds[split].select(range(DEBUG_SIZE))

# Create sampled dataset
train_data = DynamicDataset(
    raw_data_ds["train"],
    raw_documents_ds,
    tokenizer=tokenizer,
    seed=SEED,
    indexing=not USE_AUG,
    use_cot=USE_COT
)


# MODEL SETUP

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model
model = WeightedLossT5.from_pretrained(
    MODEL_NAME,
    cache_dir=MODELS_DIR,
    torch_dtype="auto",
    local_files_only=True,  # Change for first time downloads
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
)
model.use_cot = USE_COT
model.tokenizer = tokenizer

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
    lora_dropout=0.3,
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

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer=optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    threshold=1e-4,
    threshold_mode="abs",
    cooldown=0,
    min_lr=1e-6,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    eval_accumulation_steps=ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    seed=SEED,
    data_seed=SEED,
    optim="adamw_bnb_8bit",
    lr_scheduler_type="reduce_lr_on_plateau",
    bf16=True,
    deepspeed=ds_config,
    report_to="tensorboard",
    logging_strategy="epoch",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_on_start=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_part_match_accuracy",
    greater_is_better=True,
    predict_with_generate=True,
    label_names=["labels"],
    label_smoothing_factor=0,  # Keep at 0, use custom loss instead
    gradient_checkpointing=True,  # Large memory impact
    gradient_checkpointing_kwargs={
        "use_reentrant": False,  # Suppresses PyTorch warning, may behave unexpectedly
    },
    dataloader_drop_last=False,  # Changes it for all data splits
    dataloader_num_workers=0 if DEBUG else num_cpus,
    dataloader_prefetch_factor=None if DEBUG else 2,
    dataloader_pin_memory=False if DEBUG else True,
    dataloader_persistent_workers=False,  # Causes hanging at the end
)

# Trainer
# Do not pass a loss function here
# It will break the custom loss function
# Passing a loss function or setting the label_smoothing_factor inside args
# will bypass the custom loss implemented inside the model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=tokenized_ds["eval"],
    processing_class=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    compute_metrics=None,  # Handled by an internal function
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-4),
        EpochTrackerCallback()
    ],
)

# Dirty fix for init
trainer.debug = DEBUG
trainer.use_cot = USE_COT

# Dirty fix, workaround for internal functions
trainer.current_split = "eval"
trainer.compute_metrics = trainer._compute_metrics

# Stores external data not used by the model
# Only used by metrics
trainer.metrics_input_data = raw_data_ds

# Use beam search instead of greedy decoding
# Performs better at the cost of higher compute
gen_config = GenerationConfig(
    num_beams=10,
    early_stopping=True,
)


def perform_metrics(split: str):
    """Calculate metrics for a data split and log it"""
    trainer.current_split = split
    results = trainer.evaluate(tokenized_ds[split], metric_key_prefix=split, generation_config=gen_config)
    trainer.log(results)
    trainer.save_metrics(split, results)
    print(results)


if __name__ == "__main__":
    try:
        trainer_results = trainer.train()

        with open(os.path.join(OUTPUT_DIR, "log_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)

        trainer.remove_callback(EarlyStoppingCallback)
        logger.info("Training Completed")

        for split in SPLITS:
            perform_metrics(split)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Training failed: {e}")
    finally:
        try:
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("Saved succesfully")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Saving failed: {e}")

        if dist.is_initialized():
            dist.destroy_process_group()

    logger.info("Script finished")
    sys.stdout.flush()
    sys.stderr.flush()
