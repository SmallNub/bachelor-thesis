import torch
from torch import Generator
from torch.utils.data import Dataset

from config import SEPARATOR


NUM_PROMPT_FORMATS = 5


def tokenize(prompt, target, tokenizer):
    # Model will silently truncate above 512 tokens
    model_inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    labels = tokenizer(
        text_target=target,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_prompt_format(indexing: bool, use_cot: bool, format_id: int = 0):
    if indexing or use_cot:
        raise NotImplementedError()

    templates = [
        "Answer the query with a document ID.",
        "Generate the document ID that answers the question.",
        "Based on the question, predict the document ID.",
        "Retrieve a document ID that fits the query.",
        "Using the question, find the document ID."
    ]
    docid = "Format: company-year-keyword-keyword-keyword-keyword"
    query = "Question: {company}-{year}, {query}"
    prompt = f"Q: {templates[format_id]}\n{docid}\n{query}\nA: "

    return prompt


def build_process_fn(tokenizer, input_key: str, indexing: bool, use_cot: bool, format_id: int = 0):
    prompt_format = get_prompt_format(indexing, use_cot, format_id)

    def process_examples(examples):
        input_texts = examples[input_key]
        docids = examples["document_id"]

        if isinstance(docids, str):
            input_texts = [input_texts]
            docids = [docids]

        prompts = []
        answers = []
        for input_text, docid in zip(input_texts, docids):
            company, year, *keywords = docid.split(SEPARATOR)

            prompt = prompt_format.format(company=company, year=year, query=input_text)
            prompts.append(prompt)

            if use_cot:
                answer = (
                    f"This is about {company} in the year {year}. "
                    f"Therefore, the final answer is {docid}."
                )
            else:
                answer = docid
            answers.append(answer)

        tokenized = tokenize(prompts, answers, tokenizer)
        return tokenized
    return process_examples


class DynamicDataset(Dataset):
    def __init__(
        self,
        data_ds,
        documents_ds,
        tokenizer,
        seed: int,
        indexing: bool,
        use_cot: bool,
    ):
        self.data_ds = data_ds
        self.documents_ds = documents_ds
        self.tokenizer = tokenizer
        self.generator = Generator("cpu").manual_seed(seed)
        self.indexing = indexing
        self.use_cot = use_cot

    def __len__(self):
        return len(self.data_ds) + len(self.documents_ds)

    def __getitem__(self, idx):
        if idx < len(self.data_ds):
            sample = self.data_ds[idx]
            input_key = "question"
        else:
            sample = self.documents_ds[idx - len(self.data_ds)]
            input_key = "pseudo_query"

        process_fn = build_process_fn(
            self.tokenizer,
            input_key,
            self.indexing,
            self.use_cot,
            torch.randint(0, NUM_PROMPT_FORMATS, (1,), generator=self.generator).item()
        )
        tokenized = process_fn(sample)

        # Unnest values
        for key in tokenized.keys():
            tokenized[key] = tokenized[key][0]

        return tokenized
