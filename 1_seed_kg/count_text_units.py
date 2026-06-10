#!/usr/bin/env python3
"""
count_text_units.py

Helper to count how many text units were created by step 2 and determine
how many SLURM array jobs you need for step 3.

Usage:
  python count_text_units.py --root_dir ${OUTPUT_BASE}/graphrag
"""

import argparse
import asyncio
from pathlib import Path

from graphrag.config.load_config import load_config
from graphrag.storage.factory import StorageFactory
from graphrag.cache.factory import CacheFactory
from graphrag.index.run.utils import create_run_context
from graphrag.utils.storage import load_table_from_storage


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True,
                    help="Root directory with settings.yaml (same as graphrag_index.py --root_dir)")
    ap.add_argument("--rows_per_job", type=int, default=8000,
                    help="Rows processed per SLURM array job (default: 8000)")
    return ap.parse_args()


async def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    rows_per_job = args.rows_per_job

    cli_overrides = {}
    config = load_config(root_dir, None, cli_overrides)

    storage = StorageFactory().create_storage(
        storage_type=config.output.type,
        kwargs=config.output.model_dump(),
    )
    cache = CacheFactory().create_cache(
        cache_type=config.cache.type,
        root_dir=config.root_dir,
        kwargs=config.cache.model_dump(),
    )
    context = create_run_context(storage=storage, cache=cache, stats=None)

    text_units = await load_table_from_storage("text_units", context.storage)
    n = len(text_units)
    num_jobs = (n + rows_per_job - 1) // rows_per_job

    print("=======================================")
    print(f"Total text units:  {n}")
    print(f"Rows per job:      {rows_per_job}")
    print(f"Jobs needed:       {num_jobs}")
    print("=======================================")
    print(f"Set #SBATCH --array=0-{num_jobs - 1} in your SLURM script for step 3.")


if __name__ == "__main__":
    asyncio.run(main())
