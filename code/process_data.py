import warnings
import torch
from torch import Generator
from torch.utils.data import Dataset
from datasets import load_dataset

from config import (
    DATA_TRAIN_PROC,
    DATA_EVAL_PROC,
    DATA_TEST_PROC,
    DATA_DOCUMENTS,
    DATA_DOCUMENTS_AUG,
    SPLITS,
    SEPARATOR
)


NUM_PROMPT_FORMATS = 5
CONSTRAINTS_ANCHOR_STARTS = " answer answers reply replies"
CONSTRAINTS_ANCHOR_VERBS = " be being am is are was were having have has had"


def load_data(use_aug=True, debug=False, debug_size=4):
    """Load documents and data."""
    # Process documents for indexing
    if use_aug:
        # Augmented documents use pseudo-queries which should use the retrieval task instead
        document_file = DATA_DOCUMENTS_AUG
    else:
        # Traditional documents should use the indexing task
        document_file = DATA_DOCUMENTS
        raise NotImplementedError("Indexing is not supported.")

    # Load documents
    raw_documents_ds = load_dataset("csv", data_files=document_file, split="train")

    # Process data for retrieval (train, valid, test)
    file_mapping = {
        "train": DATA_TRAIN_PROC,
        "eval": DATA_EVAL_PROC,
        "test": DATA_TEST_PROC,
    }

    # Process queries for retrieval
    raw_data_ds = load_dataset("csv", data_files=file_mapping)

    # Reduce data size for all splits
    if debug:
        raw_documents_ds = raw_documents_ds.select(range(debug_size))

        for split in SPLITS:
            raw_data_ds[split] = raw_data_ds[split].select(range(debug_size))

    return raw_documents_ds, raw_data_ds


def _warn_overflow(model_input, tokenizer):
    """Warns if tokens overflow the model input limit."""
    if len(model_input["input_ids"]) >= tokenizer.model_max_length:
        warnings.warn("input_ids is exceeding the input limit", RuntimeWarning)

    if "labels" in model_input and len(model_input["labels"]) >= tokenizer.model_max_length:
        warnings.warn("labels is exceeding the input limit", RuntimeWarning)


def tokenize(prompt, target, tokenizer, warn_overflow=False):
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

    if warn_overflow:
        _warn_overflow(model_inputs, tokenizer)

    return model_inputs


def _get_randint(low: int, high: int, generator: Generator = None):
    """
    Returns a random int between `low` (inclusive) and `high` (exclusive).\\
    Will return `low` if no generator is given.
    """
    if generator is None:
        return low

    return torch.randint(low, high, (1,), generator=generator).item()


def _get_prompt_format(format_id: int = 0, use_cot: bool = False):
    """Get a prompt format template."""
    templates = [
        "Answer the query with a document ID.",
        "Generate the document ID that answers the question.",
        "Based on the question, predict the document ID.",
        "Retrieve a document ID that fits the query.",
        "Using the question, find the document ID."
    ]
    assert len(templates) == NUM_PROMPT_FORMATS, "Templates do not match NUM_PROMPT_FORMATS"

    template = f"{templates[format_id]}"

    if use_cot:
        cot_templates = [
            "Use step-by-step reasoning.",
            "You need to explain your answer.",
            "Think this through carefully.",
            "Let's think step-by-step.",
            "Explain your reasoning before answering.",
        ]
        template += " " + cot_templates[format_id]

    docid = "Format: company-year-keyword-keyword-keyword-keyword"
    query = "Question: {company}-{year}, {query}"
    prompt = f"Q: {template}\n{docid}\n{query}\nA: "
    return prompt


def _get_answer_format(use_cot: bool = False):
    """Get an answer format template."""
    if not use_cot:
        return "{docid}"

    answer = (
        "The query is about {company} in {year}. "
        "They are related to {keyword_1} {keyword_2}. "
        "Related documents talk about {keyword_3} {keyword_4}. "
        "Therefore, the answer is {docid}"
    )
    return answer


def process_pair(input_text: str, docid: str, format_id: int = 0, use_cot: bool = False):
    """Process an input text and docid pair."""
    company, year, *keywords = docid.split(SEPARATOR)

    prompt_format = _get_prompt_format(format_id, use_cot)
    answer_format = _get_answer_format(use_cot)

    prompt = prompt_format.format(company=company, year=year, query=input_text)

    answer_map = {
        "docid": docid,
        "company": company,
        "year": year,
    }
    answer_map.update({f"keyword_{i}": keyword for i, keyword in enumerate(keywords, 1)})

    answer = answer_format.format_map(answer_map)

    return prompt, answer


def process_example_pairs(pairs: list[tuple[str, str]], format_ids: list[int], use_cot=True):
    """Create examples using pairs of questions and answers."""
    full_prompt = ""
    final_answer = ""

    for i, ((input_text, docid), format_id) in enumerate(zip(pairs, format_ids, strict=True)):
        prompt, answer = process_pair(input_text, docid, format_id, use_cot)

        if i < len(pairs) - 1:
            example = prompt + answer + "\n"
            full_prompt += example
        else:
            full_prompt += prompt
            final_answer = answer

    return full_prompt, final_answer


def prepare_samples(samples, input_key):
    """Get the data from the huggingface dataset."""
    input_texts = samples[input_key]
    docids = samples["document_id"]
    return input_texts, docids


class DynamicDataset(Dataset):
    """Dataset with randomized prompt sampling."""
    def __init__(
        self,
        data_ds,
        tokenizer,
        documents_ds=None,
        format_id: int = 0,  # -1 enables random prompts and examples
        n_examples: int = 0,
        seed: int = 42,
        indexing: bool = False,
        use_cot: bool = False,
        debug: bool = False,
        epoch: int = 0,
    ):
        self.data_ds = data_ds
        self.tokenizer = tokenizer
        self.documents_ds = documents_ds
        self.format_id = format_id
        self.n_examples = n_examples
        self.base_seed = seed
        self.indexing = indexing
        self.use_cot = use_cot
        self.debug = debug
        self.epoch = epoch

        if documents_ds is None:
            self.total_length = len(self.data_ds)
        else:
            self.total_length = len(self.data_ds) + len(self.documents_ds)

        if self.indexing:
            raise NotImplementedError("Indexing is not supported.")

    def __len__(self):
        return self.total_length

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_generator(self, idx: int):
        """Get a `torch.Generator` using the seed and the iteration."""
        # Set seed based on iteration
        # Ensures behaviour is the same as the single-threaded case
        seed = (self.base_seed + self.epoch * self.total_length + idx) % (2**32)
        generator = torch.Generator("cpu").manual_seed(seed)
        return generator

    def get_sample(self, idx: int):
        """Get a sample from the internal datasets."""
        if idx < len(self.data_ds):
            sample = self.data_ds[idx]
            input_key = "question"
        else:
            sample = self.documents_ds[idx - len(self.data_ds)]
            input_key = "pseudo_query"

        return sample, input_key

    def get_format_id(self, idx: int):
        """Get a (randomized) format id from the iteration."""
        if self.format_id == -1:
            # Random format id
            generator = self.get_generator(idx)
            prompt_id = _get_randint(0, NUM_PROMPT_FORMATS, generator)
        else:
            prompt_id = self.format_id

        return prompt_id

    def get_examples(self, idx: int):
        """Get examples for prepending the prompt."""
        generator = self.get_generator(idx)
        pairs = []
        format_ids = []

        for _ in range(self.n_examples):
            # Get random samples
            random_idx = _get_randint(0, self.total_length, generator)
            sample, input_key = self.get_sample(random_idx)
            format_id = self.get_format_id(random_idx)
            input_text, docid = prepare_samples(sample, input_key)

            pairs.append((input_text, docid))
            format_ids.append(format_id)

        return pairs, format_ids

    def __getitem__(self, idx: int):
        # Get main sample
        sample, input_key = self.get_sample(idx)
        main_pair = prepare_samples(sample, input_key)
        format_id = self.get_format_id(idx)

        # Prepend examples
        if self.n_examples > 0:
            pairs, format_ids = self.get_examples(idx)
            pairs.append(main_pair)
            format_ids.append(format_id)
        else:
            pairs = [main_pair]
            format_ids = [format_id]

        # Create prompt and answer
        prompt, answer = process_example_pairs(pairs, format_ids, self.use_cot)

        if self.debug:
            print(f"Prompt: (\n{prompt}\n) Answer: {answer}")

        tokenized = tokenize(prompt, answer, self.tokenizer, True)

        return tokenized


class DocidTrie:
    def __init__(self, docids, tokenizer):
        self.trie = {}
        self.tokenizer = tokenizer
        for docid in docids:
            # Add both variations to the trie
            variants = [f" {docid}", docid]
            for variant in variants:
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                node = self.trie
                for token in tokens:
                    if token not in node:
                        node[token] = {}
                    node = node[token]
                # Allow EOS at the end of every path
                node[tokenizer.eos_token_id] = {}

    def get_allowed_tokens(self, current_token_ids):
        node = self.trie
        for token in current_token_ids:
            # If we already hit an EOS, stay in EOS/Pad loop
            if token == self.tokenizer.eos_token_id or token == self.tokenizer.pad_token_id:
                return [self.tokenizer.eos_token_id]

            node = node.get(token)
            if node is None:
                # Path is broken, fallback to allowing all tokens
                return list(range(self.tokenizer.vocab_size))

        return list(node.keys())


def get_prefix_allowed_tokens_fn(tokenizer, trie_manager, use_cot=True):
    # Pre-encoded anchor sets
    anchor_starts = set(tokenizer.encode(CONSTRAINTS_ANCHOR_STARTS, add_special_tokens=False))
    verbs = set(tokenizer.encode(CONSTRAINTS_ANCHOR_VERBS, add_special_tokens=False))

    # Persistent state: maps batch_id -> trigger_position
    # Initialized with -1 (not yet found)
    batch_state = {}

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent_list = sent.tolist()

        # Only check for trigger in CoT mode
        if use_cot:
            # Check if already found the trigger
            trigger_pos = batch_state.get(batch_id, -1)

            # If not found, only check the most recent 2 tokens
            if trigger_pos == -1 and len(sent_list) >= 2:
                # Check the last 4 tokens for a combination of Anchor + Verb
                recent = sent_list[-4:]
                has_anchor = any(t in anchor_starts for t in recent)
                has_verb = any(t in verbs for t in recent)

                if has_anchor and has_verb:
                    print("trigger")
                    batch_state[batch_id] = len(sent_list)

            if trigger_pos == -1:
                # Still in Reasoning mode
                return list(range(tokenizer.vocab_size))

        # Constrained Mode (Trie Lookup)
        # Only pass the tokens generated after the trigger_pos
        docid_prefix_tokens = sent_list[trigger_pos:]
        allowed = trie_manager.get_allowed_tokens(docid_prefix_tokens)

        # Return Trie keys, or EOS if the docid is complete
        return allowed if allowed else [tokenizer.eos_token_id]

    return prefix_allowed_tokens_fn
