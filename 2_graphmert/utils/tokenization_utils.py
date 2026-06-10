#!/usr/bin/env python3
"""
tokenization_utils.py

Tokenizes text files using a BioMedBERT-family tokenizer and saves:
  1. Tokenized train/validation datasets
  2. A "stable tokenizer" with consistent PAD/MASK special tokens

The stable tokenizer is used by all subsequent pipeline steps to ensure
consistent token IDs between tokenization, entity discovery, and training.

Called by run_tokenization.py.
"""

import os
import argparse
import logging
import hashlib
from itertools import chain
from copy import deepcopy
from typing import List

import datasets
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, set_seed
import spacy
from spacy.language import Language


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
datasets.utils.logging.set_verbosity_info()


def parse_args():
    ap = argparse.ArgumentParser(description="Tokenize text corpus for GraphMERT")
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing .txt files (one file per chapter/section)")
    ap.add_argument("--output_dir", required=True,
                    help="Base output directory; tokenized datasets and stable tokenizer saved here")
    ap.add_argument("--tokenizer", default="dmis-lab/biobert-base-cased-v1.2",
                    help="HuggingFace model name or local path to base tokenizer")
    ap.add_argument("--max_seq_length", type=int, default=128,
                    help="Max sequence length for chunking (default: 128)")
    ap.add_argument("--validation_split_pct", type=int, default=5,
                    help="Percentage of data to hold out for validation (default: 5)")
    ap.add_argument("--num_workers", type=int, default=32,
                    help="Number of parallel workers for dataset map (default: 32)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _unique_run_tag(paths: List[str]) -> str:
    h = hashlib.md5()
    payload = "\n".join(sorted(os.path.abspath(p) for p in paths))
    h.update(payload.encode("utf-8"))
    return h.hexdigest()[:10]


def build_spacy_pipeline():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
    nlp.max_length = max(nlp.max_length, 2_000_000)

    @Language.component("lower_case_lemmas")
    def lower_case_lemmas(doc):
        for token in doc:
            token.lemma_ = token.lemma_.lower()
        return doc

    nlp.add_pipe("lower_case_lemmas", after="tagger")
    return nlp


def load_raw_datasets(input_files: List[str], validation_split_pct: int, cache_dir: str) -> DatasetDict:
    raw = DatasetDict({
        "train": load_dataset(
            path="text",
            data_files={"train": input_files},
            split=f"train[{validation_split_pct}%:]",
            cache_dir=cache_dir,
        ),
        "validation": load_dataset(
            path="text",
            data_files={"train": input_files},
            split=f"train[:{validation_split_pct}%]",
            cache_dir=cache_dir,
        ),
    })
    return raw


def tokenize_dataset(raw_datasets: DatasetDict, tokenizer, nlp, num_workers: int) -> DatasetDict:
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    remove_columns = [c for c in column_names if c != text_column_name]

    def tokenize_function(examples):
        texts = [t if isinstance(t, str) else "" for t in examples[text_column_name]]
        max_len = max((len(t) for t in texts), default=0)
        if max_len + 1000 > nlp.max_length:
            nlp.max_length = max_len + 1000
        try:
            docs = list(nlp.pipe(texts))
            word_lists = [[str(tok) for tok in doc] for doc in docs]
            tokenized = tokenizer(
                word_lists,
                is_split_into_words=True,
                add_special_tokens=False,
                return_special_tokens_mask=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )
        except Exception as e:
            logger.warning("spaCy pipe fallback: %r", e)
            all_input_ids = []
            for t in texts:
                doc = nlp(t)
                enc = tokenizer([str(tok) for tok in doc], is_split_into_words=True, add_special_tokens=False)
                all_input_ids.append(enc["input_ids"])
            tokenized = {"input_ids": all_input_ids}
        tokenized[text_column_name] = examples[text_column_name]
        return tokenized

    tokenized_datasets = DatasetDict()
    for split, ds in raw_datasets.items():
        tokenized_datasets[split] = ds.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns,
            num_proc=num_workers,
            load_from_cache_file=False,
            desc=f"Tokenizing {split}",
        )
    return tokenized_datasets


def concatenate_texts_into_chunks(dataset, tokenizer, max_seq_length: int, split: str, num_workers: int):
    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        max_raw = max_seq_length - tokenizer.num_special_tokens_to_add()
        if total_length >= max_raw:
            total_length = (total_length // max_raw) * max_raw
        out = {}
        out["input_ids"] = [
            tokenizer.build_inputs_with_special_tokens(concatenated["input_ids"][i: i + max_raw])
            for i in range(0, total_length, max_raw)
        ]
        return out

    return dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=False,
        desc=f"Chunking {split} into seq_len={max_seq_length}",
    )


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    tokenizer_name = args.tokenizer

    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "hf_cache")
    stable_tokenizer_dir = os.path.join(output_dir, "stable_tokenizer")
    tokenized_dir = os.path.join(output_dir, "tokenized_inputs")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(tokenized_dir, exist_ok=True)

    # Discover input .txt files
    input_files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".txt")
    ])
    if not input_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    missing = [p for p in input_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    set_seed(args.seed)

    logger.info("Loading base tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )

    # Ensure PAD and MASK tokens exist for consistent downstream use
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        logger.info("Added [PAD] special token")
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        logger.info("Added [MASK] special token")

    # Save stable tokenizer — all subsequent steps MUST use this
    tokenizer.save_pretrained(stable_tokenizer_dir)
    logger.info("Stable tokenizer saved to: %s", stable_tokenizer_dir)

    run_tag = _unique_run_tag(input_files)
    out_train = os.path.join(tokenized_dir, f"train_{run_tag}_tokenized")
    out_val   = os.path.join(tokenized_dir, f"validation_{run_tag}_tokenized")

    nlp = build_spacy_pipeline()
    raw_datasets = load_raw_datasets(input_files, args.validation_split_pct, cache_dir)

    logger.info("Tokenizing %d train + %d validation examples...",
                len(raw_datasets["train"]), len(raw_datasets["validation"]))
    tokenized = tokenize_dataset(raw_datasets, tokenizer, nlp, args.num_workers)

    for split in list(tokenized.keys()):
        if "text" in tokenized[split].column_names:
            tokenized[split] = tokenized[split].remove_columns(["text"])

    grouped = DatasetDict()
    for split in tokenized.keys():
        grouped[split] = concatenate_texts_into_chunks(
            tokenized[split], tokenizer, args.max_seq_length, split, args.num_workers
        )

    grouped["train"].save_to_disk(out_train)
    grouped["validation"].save_to_disk(out_val)

    logger.info("Tokenized train saved to:      %s", out_train)
    logger.info("Tokenized validation saved to: %s", out_val)
    logger.info("Stable tokenizer saved to:     %s", stable_tokenizer_dir)
    logger.info("NEXT STEP: set TOKENIZED_TRAIN_DIR=%s for entity_discovery.py", out_train)


if __name__ == "__main__":
    main()
