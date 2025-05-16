import os
import json
import multiprocessing
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb
import evaluate

from config import DATA_DOCUMENTS, MODELS_DIR

# 0. Constants & environment setup
MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = os.path.join(MODELS_DIR, "finqa_indexer")

with open("code/ds_config.json", "r", encoding="utf-8") as f:
    ds_config = json.load(f)

# Detect number of CPUs and GPUs
num_cpus = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count()))
print(f"Using {num_cpus} CPU core(s)")

num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} CUDA device(s)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 1. Load tokenizer & model + PEFT LoRA
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODELS_DIR, use_fast=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODELS_DIR,
    torch_dtype="auto",
    local_files_only=False,  # Change for first time downloads
    low_cpu_mem_usage=True,
    quantization_config=bnb_config
)

base_model.config.use_cache = False
base_model.enable_input_require_grads()

base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
print(f"Memory footprint: {model.get_memory_footprint():,}")

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=bnb.optim.Adam8bit,
    lr=5e-5,
    loraplus_lr_ratio=16,
)
scheduler = None

# 2. Load & preprocess dataset
raw_ds = load_dataset("csv", data_files=DATA_DOCUMENTS, split="train")


def preprocess_fn(example):
    inputs = tokenizer(
        example["full_text"],
        truncation=True,
        max_length=4096,
    )
    targets = tokenizer(
        example["id"],
        truncation=True,
        max_length=32,
    )
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": targets.input_ids,
    }


# Map & set format for PyTorch
tokenized_ds = raw_ds.map(
    preprocess_fn,
    remove_columns=raw_ds.column_names,
    num_proc=num_cpus,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

exact_match = evaluate.load("exact_match")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # decode predictions and labels into strings
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # EM accuracy
    em = exact_match.compute(predictions=decoded_preds, references=decoded_labels)["exact_match"]
    return {"exact_match": em}


# 3. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    eval_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=20,
    bf16=True,
    deepspeed=ds_config,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_on_start=True,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    greater_is_better=True,
    predict_with_generate=True,
    label_names=["labels"],
    gradient_checkpointing=True,
)

# 4. Initialize Trainer and launch training
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    eval_dataset=tokenized_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
