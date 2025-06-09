import torch
from torch import Generator
from torch.utils.data import Dataset

from config import SEPARATOR


NUM_PROMPT_FORMATS = 5


def tokenize(prompt, target, tokenizer):
    """Tokenize the string inputs into tokens. (target is optional)"""
    # Model will silently truncate above 512 tokens
    model_inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    if target is not None:
        labels = tokenizer(
            text_target=target,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def _get_randint(low: int, high: int, generator: Generator = None):
    """
    Returns a random int between low (inclusive) and high (exclusive).\\
    Will return `low` if no generator is given.
    """
    if generator is None:
        return low

    return torch.randint(low, high, (1,), generator=generator).item()


def _get_prompt_format_default(format_id: int = 0):
    templates = [
        "Answer the query with a document ID.",
        "Generate the document ID that answers the question.",
        "Based on the question, predict the document ID.",
        "Retrieve a document ID that fits the query.",
        "Using the question, find the document ID."
    ]
    assert len(templates) == NUM_PROMPT_FORMATS, "Templates do not match NUM_PROMPT_FORMATS"

    docid = "Format: company-year-keyword-keyword-keyword-keyword"
    query = "Question: {company}-{year}, {query}"
    prompt = f"Q: {templates[format_id]}\n{docid}\n{query}\nA: "
    return prompt


def _get_prompt_format_cot(n_examples: int, generator: Generator = None):
    full_template = ""
    for i in range(n_examples):
        pass
    return full_template


def get_prompt_format(
    indexing: bool,
    use_cot: bool,
    **kwargs,
):
    """Get a prompt format based on the type."""
    if indexing:
        raise NotImplementedError()

    if use_cot:
        prompt = _get_prompt_format_cot()
    else:
        if "format_id" not in kwargs:
            kwargs["format_id"] = 0

        prompt = _get_prompt_format_default(kwargs["format_id"])

    return prompt


def build_process_fn(
    tokenizer,
    input_key: str,
    indexing: bool,
    use_cot: bool,
    **kwargs,
):
    """Creates a process function compatible with Huggingface datasets."""
    # Get a random prompt
    prompt_format = get_prompt_format(indexing, use_cot, **kwargs)

    def process_examples(examples):
        """Process inputs into proper model inputs."""
        input_texts = examples[input_key]
        docids = examples["document_id"]

        if isinstance(docids, str):
            # For singular inputs
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

        if "debug" in kwargs and kwargs["debug"]:
            print(f"Prompts: {prompts}\nAnswers: {answers}")

        tokenized = tokenize(prompts, answers, tokenizer)
        return tokenized
    return process_examples


class DynamicDataset(Dataset):
    """Dataset with randomized prompt sampling."""
    def __init__(
        self,
        data_ds,
        documents_ds,
        tokenizer,
        seed: int,
        indexing: bool,
        use_cot: bool,
        debug: bool = False,
        epoch: int = 0,
    ):
        self.data_ds = data_ds
        self.documents_ds = documents_ds
        self.tokenizer = tokenizer
        self.base_seed = seed
        self.indexing = indexing
        self.use_cot = use_cot
        self.debug = debug
        self.epoch = epoch
        self.total_length = len(self.data_ds) + len(self.documents_ds)

    def __len__(self):
        return self.total_length

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, idx):
        if idx < len(self.data_ds):
            sample = self.data_ds[idx]
            input_key = "question"
        else:
            sample = self.documents_ds[idx - len(self.data_ds)]
            input_key = "pseudo_query"

        # Set seed based on iteration
        # Ensures behaviour is the same as the single-threaded case
        seed = (self.base_seed + self.epoch * self.total_length + idx) % (2**32)
        generator = torch.Generator("cpu").manual_seed(seed)

        random_int = _get_randint(0, NUM_PROMPT_FORMATS, generator)

        process_fn = build_process_fn(
            self.tokenizer,
            input_key,
            self.indexing,
            self.use_cot,
            format_id=random_int,
            debug=self.debug
        )
        tokenized = process_fn(sample)

        # Unnest values
        for key in tokenized.keys():
            tokenized[key] = tokenized[key][0]

        return tokenized
