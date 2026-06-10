#!/usr/bin/env python3
"""
run_tokenization.py  — GraphMERT Pipeline Step 1

Tokenizes the text corpus and creates the stable tokenizer used by all
subsequent pipeline steps.

Usage:
  python run_tokenization.py \
    --input_dir   ${OUTPUT_BASE}/graphrag/input \
    --output_dir  ${OUTPUT_BASE}/graphmert \
    --tokenizer   dmis-lab/biobert-base-cased-v1.2

Outputs (inside --output_dir):
  stable_tokenizer/          <- Use this tokenizer for ALL subsequent steps
  tokenized_inputs/train_*   <- Tokenized training dataset
  tokenized_inputs/val_*     <- Tokenized validation dataset
"""

import logging
import sys

from utils import tokenization_utils

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting tokenization pipeline")
    tokenization_utils.main()
    logger.info("Tokenization complete — stable tokenizer and datasets saved")
