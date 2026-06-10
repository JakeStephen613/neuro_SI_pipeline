#!/usr/bin/env python3
"""
models.py

Model downloader / cache-setup helper.

Usage:
  python models.py --model_name deepseek-r1  --output_dir ${SCRATCH}/models
  python models.py --model_name qwen3-8b     --output_dir ${SCRATCH}/models

Or call download_model() from your own scripts.
"""

import os
import argparse


KNOWN_MODELS = {
    "deepseek-r1":   "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "qwen3-8b":      "Qwen/Qwen3-8B",
    "qwen3-14b":     "Qwen/Qwen3-14B",
    "qwen3-32b":     "Qwen/Qwen3-32B",
    "mistral-nemo":  "mistralai/Mistral-Nemo-Instruct-2407",
}


def setup_cache(cache_base: str) -> None:
    os.environ["HF_HOME"]            = os.path.join(cache_base, "hf_home")
    os.environ["HF_DATASETS_CACHE"]  = os.path.join(cache_base, "hf_datasets_cache")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "transformers")
    for p in [os.environ["HF_HOME"], os.environ["HF_DATASETS_CACHE"], os.environ["TRANSFORMERS_CACHE"]]:
        os.makedirs(p, exist_ok=True)


def download_model(model_id: str, output_dir: str, cache_base: str = None) -> str:
    """
    Download a model from HuggingFace Hub to output_dir.
    model_id can be a HF repo ID (e.g. 'Qwen/Qwen3-14B') or a shorthand from KNOWN_MODELS.
    Returns the local path where the model was saved.
    """
    from huggingface_hub import snapshot_download

    if model_id in KNOWN_MODELS:
        repo_id = KNOWN_MODELS[model_id]
    else:
        repo_id = model_id

    model_name = repo_id.split("/")[-1]
    local_path = os.path.join(output_dir, model_name)

    if os.path.isdir(local_path) and os.listdir(local_path):
        print(f"Model already exists at: {local_path}")
        return local_path

    if cache_base:
        setup_cache(cache_base)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {repo_id} → {local_path}")
    snapshot_download(repo_id=repo_id, local_dir=local_path)
    print(f"Download complete: {local_path}")
    return local_path


def parse_args():
    ap = argparse.ArgumentParser(description="Download HuggingFace models")
    ap.add_argument("--model_name", required=True,
                    help=f"Model name or HF repo ID. Shorthands: {list(KNOWN_MODELS.keys())}")
    ap.add_argument("--output_dir", required=True,
                    help="Directory where model will be saved")
    ap.add_argument("--cache_dir", default=None,
                    help="Optional HF cache directory")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_model(args.model_name, args.output_dir, cache_base=args.cache_dir)
