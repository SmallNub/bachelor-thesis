import os
import sys
import logging
import traceback
import json
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb

from utils import (
    init_logging,
    enable_tf32,
    detect_resources,
    set_seed,
    ResourceMonitorCallback,
)
from process_data import load_data, DynamicDataset
from model import load_tokenizer, load_model, CustomTrainer, EpochTrackerCallback
from config import (
    MODELS_DIR,
    SPLITS,
)


# NOTE
# Code is optimized for 1 H100 GPU with this model size
# Maximum 8 hours, average 3 hours
# Deepspeed can be enabled, but will behave differently
# bf16 is used which might not work on some GPUs
# Data shuffle seems to be different between num_workers=0 and num_workers>0
# Might not be reproducible?
# torch.compile is incompatible

# TODO
# Fix CoT


# ENVIRONMENT SETUP

# Enables way more logging
# from transformers.utils import logging as hf_logging
# hf_logging.set_verbosity_debug()

enable_tf32()

home_dir = os.getenv("HOME")
job_id = os.getenv("SLURM_JOBID")

init_logging(home_dir, job_id)
logger = logging.getLogger(__name__)

logger.info("Script started...")

# Enable debug to drastically reduce values and log many values
DEBUG = False
DEBUG_INPUTS = False
DEBUG_SIZE = 4  # Using sampling will behave differently (doubled size)
QUANTIZATION = True

USE_COT = True

# Number of examples to use for prompts
# 0 will disable examples
# High values can reach model input limit
# Can be used with or without CoT
N_EXAMPLES = 2

# Use only pseudo-queries instead of documents
# Prompt will be truncated when indexing normally
# Indexing is no longer supported, so keep this on
USE_AUG = True

SEED = 42
BATCH_SIZE = min(4, DEBUG_SIZE) if DEBUG else 32
ACCUMULATION_STEPS = 1 if DEBUG else 1
LEARNING_RATE = 2e-5
EPOCHS = 100

MODEL_NAME = "google/flan-t5-base"
logger.info(f"Using model: {MODEL_NAME}")

# This should correspond to the same name inside train.sh
OUTPUT_DIR = os.path.join(MODELS_DIR, f"finqa_base_{job_id}")
logger.info(f"Output location: {OUTPUT_DIR}")

# Use an already trained model
USE_TRAINED = False
TRAINED_DIR = os.path.join(MODELS_DIR, "finqa_base_10")

# Deepspeed config
# with open("code/ds_config.json", "r", encoding="utf-8") as f:
#     ds_config = json.load(f)

# Detect number of CPUs and GPUs
num_cpus, num_gpus = detect_resources(logger)

set_seed(SEED)


# DATA PREPROCESSING

tokenizer = load_tokenizer(MODEL_NAME)

raw_documents_ds, raw_data_ds = load_data(USE_AUG, DEBUG, DEBUG_SIZE)

# Create sampled dataset
train_data = DynamicDataset(
    raw_data_ds["train"],
    tokenizer=tokenizer,
    documents_ds=raw_documents_ds,
    format_id=-1,
    n_examples=N_EXAMPLES,
    seed=SEED,
    indexing=not USE_AUG,
    use_cot=USE_COT,
    debug=DEBUG_INPUTS,
)

eval_data = {}
for split in SPLITS:
    eval_data[split] = DynamicDataset(
        raw_data_ds[split],
        tokenizer=tokenizer,
        documents_ds=None,
        format_id=0,
        n_examples=N_EXAMPLES,
        seed=SEED,
        indexing=not USE_AUG,
        use_cot=USE_COT,
        debug=DEBUG_INPUTS,
    )


# MODEL SETUP

model = load_model(MODEL_NAME, tokenizer, USE_COT, QUANTIZATION)

# Fix for gradient checkpoints
model.config.use_cache = False

if QUANTIZATION:
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if USE_TRAINED:
        model = PeftModel.from_pretrained(model, TRAINED_DIR, is_trainable=True)
    else:
        # LoRA config (QLoRA + OLoRA)
        lora_config = LoraConfig(
            init_lora_weights="olora",
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.2,
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

if QUANTIZATION:
    # Specialized LoRA optimizer
    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=bnb.optim.AdamW8bit,
        lr=LEARNING_RATE,
        loraplus_lr_ratio=16,
    )
else:
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer=optimizer,
    mode="max",
    factor=0.5,
    patience=1,
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
    tf32=True,
    torch_empty_cache_steps=100,  # Prevent growing memory usage
    # deepspeed=ds_config,  # Disabled deepspeed
    report_to="tensorboard",
    logging_strategy="epoch",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_on_start=True,
    save_total_limit=2,
    restore_callback_states_from_checkpoint=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_part_match_accuracy",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=tokenizer.model_max_length,
    label_names=["labels"],
    label_smoothing_factor=0,  # Keep at 0, use custom loss instead
    # gradient_checkpointing=True,  # Large memory impact
    # gradient_checkpointing_kwargs={
    #     "use_reentrant": False,  # Suppresses PyTorch warning, may behave unexpectedly
    # },
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
    eval_dataset=eval_data["eval"],
    processing_class=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    compute_metrics=None,  # Handled by an internal function
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=6, early_stopping_threshold=1e-4),
        EpochTrackerCallback(),
        ResourceMonitorCallback(logger)
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


def perform_metrics(split: str):
    """Calculate metrics for a data split and log it"""
    trainer.current_split = split
    results = trainer.evaluate(eval_data[split], metric_key_prefix=split)
    trainer.log(results)
    trainer.save_metrics(split, results)
    print(results)


if __name__ == "__main__":
    try:
        logger.info("Training Started")

        model.train()
        trainer_results = trainer.train(resume_from_checkpoint=None)

        with open(os.path.join(OUTPUT_DIR, "log_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)

        trainer.remove_callback(EarlyStoppingCallback)
        logger.info("Training Completed")
        model.eval()

        with torch.no_grad():
            for split in SPLITS:
                perform_metrics(split)
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Training failed: {e}")
        logger.error(error)
    finally:
        try:
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("Saved succesfully")
        except Exception as e:
            error = traceback.format_exc()
            logger.error(f"Saving failed: {e}")
            logger.error(error)

        if dist.is_initialized():
            dist.destroy_process_group()

    logger.info("Script finished")
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
