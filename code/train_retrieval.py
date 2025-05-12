# stage2_retrieval.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import PeftModel

# 1. Load Stage1 model
INDEXER = "./finqa_indexer"
tokenizer = AutoTokenizer.from_pretrained(INDEXER)
base = AutoModelForSeq2SeqLM.from_pretrained(INDEXER)
model = PeftModel.from_pretrained(base, INDEXER)  # includes LoRA

# 2. Optionally freeze
for name, param in model.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False

# 3. Dataset
ret = load_dataset("czyssrs/FinQA", split="retriever")
def prep(ex):
    prompt = f"Question: {ex['question']}\nLet’s think step by step to identify which page contains the answer."
    inp = tokenizer(prompt, truncation=True, max_length=512)
    tgt = tokenizer(str(ex["page_id"]), truncation=True, max_length=8)
    return {"input_ids": inp.input_ids, "attention_mask": inp.attention_mask, "labels": tgt.input_ids}
ret_ds = ret.map(prep, remove_columns=ret.column_names)

# 4. Training args
args = Seq2SeqTrainingArguments(
    "./finqa_retriever", per_device_train_batch_size=4,
    gradient_accumulation_steps=4, fp16=True, deepspeed=DS_CONFIG,
    evaluation_strategy="epoch", save_strategy="epoch",
    num_train_epochs=3, load_best_model_at_end=True
)

# 5. Trainer & train
trainer = Seq2SeqTrainer(model, args, train_dataset=ret_ds.select(range(15000)),
                         eval_dataset=ret_ds.select(range(1000)), tokenizer=tokenizer)
if __name__=="__main__":
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
