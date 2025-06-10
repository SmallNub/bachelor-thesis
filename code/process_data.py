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


def _get_prompt_format(format_id: int = 0, use_cot: bool = False):
    templates = [
        "Answer the query with a document ID",
        "Generate the document ID that answers the question",
        "Based on the question, predict the document ID",
        "Retrieve a document ID that fits the query",
        "Using the question, find the document ID"
    ]
    assert len(templates) == NUM_PROMPT_FORMATS, "Templates do not match NUM_PROMPT_FORMATS"

    if use_cot:
        template = f"{templates[format_id]} by reasoning step-by-step."
    else:
        template = f"{templates[format_id]}."

    docid = "Format: company-year-keyword-keyword-keyword-keyword"
    query = "Question: {company}-{year}, {query}"
    prompt = f"Q: {template}\n{docid}\n{query}\nA: "
    return prompt


def _get_answer_format(format_id: int = 0, use_cot: bool = False):
    if not use_cot:
        return "{docid}"

    # Templates are linked to the prompt templates
    templates = [
        "{company} in {year} "
    ]
    return templates[format_id]


def process_pair(input_text: str, docid: str, format_id: int = 0, use_cot: bool = False):
    company, year, *keywords = docid.split(SEPARATOR)

    prompt_format = _get_prompt_format(format_id, use_cot)
    answer_format = _get_answer_format(format_id, use_cot)

    prompt = prompt_format.format(company=company, year=year, query=input_text)

    answer_map = {
        "docid": docid,
        "company": company,
        "year": year,
    }
    answer_map.update({f"keyword{i}": keyword for i, keyword in enumerate(keywords)})

    answer = answer_format.format_map(answer_map)

    return prompt, answer


def create_example_prompt(pairs: list[tuple[str, str]], format_ids: list[int], use_cot=True):
    full_prompt = ""
    for (input_text, docid), format_id in zip(pairs, format_ids):
        prompt, answer = process_pair(input_text, docid, format_id, use_cot)
        full_prompt


def get_prompt_format(
    indexing: bool,
    use_cot: bool,
    **kwargs,
):
    """Get a prompt format based on the type."""
    if indexing:
        raise NotImplementedError()

    if use_cot:
        pass
    else:
        if "format_id" not in kwargs:
            kwargs["format_id"] = 0

        prompt = _get_prompt_format(kwargs["format_id"])

    return prompt


def build_process_fn(
    tokenizer,
    input_key: str,
    indexing: bool,
    use_cot: bool,
    debug: bool = False,
    **kwargs,
):
    """Creates a process function compatible with Huggingface datasets."""
    if indexing:
        raise NotImplementedError("Indexing is not supported.")

    def process_samples(samples):
        """Process inputs into proper model inputs."""
        input_texts = samples[input_key]
        docids = samples["document_id"]

        if isinstance(docids, str):
            # For singular inputs
            input_texts = [input_texts]
            docids = [docids]

        prompts = []
        answers = []
        for input_text, docid in zip(input_texts, docids):
            prompt, answer = process_pair(input_text, docid, use_cot=use_cot, **kwargs)
            prompts.append(prompt)
            answers.append(answer)

        if debug:
            print(f"Prompts: {prompts}\nAnswers: {answers}")

        tokenized = tokenize(prompts, answers, tokenizer)
        return tokenized
    return process_samples


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
            debug=self.debug,
            format_id=random_int
        )
        tokenized = process_fn(sample)

        # Unnest values
        for key in tokenized.keys():
            tokenized[key] = tokenized[key][0]

        return tokenized
