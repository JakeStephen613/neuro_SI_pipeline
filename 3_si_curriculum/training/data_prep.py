#!/usr/bin/env python3
"""
data_prep.py — SI Pipeline Step 4a

Converts validated Q&A JSON into tokenized HuggingFace datasets for SFT training.

Usage:
  python training/data_prep.py \\
    --input_file  ${OUTPUT_BASE}/SI/QA_items/verified/merged_concise.json \\
    --output_path ${OUTPUT_BASE}/SI/QA_items/training_data/ \\
    --model_name  /path/to/base_model
"""

import os
import json
import argparse
import logging
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare SFT training data from Q&A JSON")
    ap.add_argument("--input_file", required=True,
                    help="Validated Q&A JSON file (output of verify_questions.py)")
    ap.add_argument("--output_path", required=True,
                    help="Output directory for tokenized HF dataset")
    ap.add_argument("--model_name", required=True,
                    help="Base model path or HF repo ID (for tokenizer)")
    ap.add_argument("--max_length", type=int, default=32768,
                    help="Max sequence length for tokenization (default: 32768)")
    ap.add_argument("--cache_dir", default=None,
                    help="HF cache directory (optional)")
    return ap.parse_args()


CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful neuroscience expert.<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n{thinking}\n</think>\n{answer}<|im_end|>"
)


def format_item(item: dict) -> str:
    question = item.get("question", "")
    thinking = item.get("thinking_trace", item.get("explanation", ""))
    answer = item.get("answer", "")
    return CHAT_TEMPLATE.format(question=question, thinking=thinking, answer=answer)


def main():
    args = parse_args()

    logger.info("Loading data from: %s", args.input_file)
    with open(args.input_file, "r") as f:
        items = json.load(f)
    logger.info("Loaded %d items", len(items))

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    formatted = [format_item(item) for item in items]
    logger.info("Formatting complete. Tokenizing...")

    tokenized = tokenizer(
        formatted,
        max_length=args.max_length,
        truncation=True,
        padding=False,
        return_attention_mask=True,
    )

    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    })

    os.makedirs(args.output_path, exist_ok=True)
    dataset.save_to_disk(args.output_path)
    logger.info("Tokenized dataset saved to: %s (%d examples)", args.output_path, len(dataset))


if __name__ == "__main__":
    main()
