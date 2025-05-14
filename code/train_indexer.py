import os
import json
import multiprocessing
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb

from config import DATA_DOCUMENTS, MODELS_DIR

# 0. Constants & environment setup
MODEL_NAME = "google/flan-t5-xl"
OUTPUT_DIR = os.path.join(MODELS_DIR, "finqa_indexer")

with open("code/ds_config.json", "r", encoding="utf-8") as f:
    ds_config = json.load(f)

# Detect number of CPUs and GPUs
num_cpus = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count()))
print(f"Using {num_cpus} CPU core(s)")

num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} CUDA device(s)")

# 1. Load tokenizer & model + PEFT LoRA
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODELS_DIR, use_fast=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODELS_DIR,
    torch_dtype=torch.float16,
    local_files_only=True,
    low_cpu_mem_usage=True
)

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

# 3. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=5,
    fp16=True,
    deepspeed=ds_config,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",
    save_total_limit=2,
    load_best_model_at_end=False,
    predict_with_generate=False,
    label_names=["labels"],
    gradient_checkpointing=True
)

# 4. Initialize Trainer and launch training
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
