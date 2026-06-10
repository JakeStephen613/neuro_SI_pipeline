# Part 2 — GraphMERT Expansion Training

Trains a graph-structured Masked Language Model (GraphMERT) on the seed KG
from Part 1, then uses it to predict and score novel tail entities, expanding
the KG for use in Part 3.

---

## Overview

GraphMERT treats each KG triple `(head, relation, tail)` as a tree: the head
and its attending text tokens form the root, and the tail entities are *leaf
nodes*.  The model is trained with a BERT-style MLM objective over these
leaf nodes, learning to fill in masked tails given a head + relation context.

The expanded KG produced here has two to three times the coverage of the seed
KG and forms the basis for multi-hop curriculum generation in Part 3.

---

## Directory Layout

```
2_graphmert/
├── graphmert_model.py              # GraphMERT model architecture
├── run_mlm.py                      # MLM training entry point
├── run_tokenization.py             # Step 1 — tokenise corpus & save stable tokenizer
├── run_dataset_preprocessing.py    # Step 4 — ground triples → training samples
├── predict_tails_llm.py            # Step 5 — predict novel tails with trained model
├── models.py                       # Download / cache HF models
├── launch_configs/
│   └── args_mlm.yaml               # All training hyperparameters & paths
├── utils/
│   ├── tokenization_utils.py       # Tokenisation helpers; creates stable tokenizer
│   ├── dataset_preprocessing_utils.py  # Co-occurrence grounding (bridge logic)
│   ├── mlm_utils.py                # MLM collator and data loading
│   ├── trainer_utils.py            # Custom Trainer subclass
│   ├── training_arguments.py       # Dataclass for training args
│   ├── data_utils.py               # Generic data helpers
│   ├── predict_tails.py            # Inference helpers
│   ├── entity_discovery/           # Step 2 — mine head entities from corpus
│   ├── relation_matching/          # Step 3 — assign relations via LLM
│   ├── combine_tails/              # Step 6 — merge & deduplicate predicted tails
│   └── llm_scores/                 # Step 7 — two-LLM fact scoring
└── slurm/
    ├── train_graphmert.slurm
    ├── predict_tails.slurm         # Array job — one shard per task
    └── entity_discovery.slurm
```

---

## Step-by-Step Usage

Set common variables first:

```bash
export REPO_DIR=/path/to/neuro_SI_pipeline
export OUTPUT_BASE=/path/to/your/scratch/output
export SEED_KG_PATH="${OUTPUT_BASE}/kg_final.parquet"   # output of Part 1
export MODEL_ID=dmis-lab/biobert-base-cased-v1.2        # or similar BioBERT
```

### Step 1 — Tokenise corpus & create stable tokenizer

```bash
python 2_graphmert/run_tokenization.py \
    --input_dir  "${OUTPUT_BASE}/text_units" \
    --output_dir "${OUTPUT_BASE}/graphmert/tokenized" \
    --tokenizer  "${MODEL_ID}"
```

This saves a *stable tokenizer* to `${OUTPUT_BASE}/graphmert/tokenized/stable_tokenizer/`.
Every subsequent step loads from that path so token IDs are consistent
throughout the pipeline.

### Step 2 — Entity discovery

Mine head-entity candidates from the tokenised corpus:

```bash
python 2_graphmert/utils/entity_discovery/entity_discovery.py \
    --tokenized_dir "${OUTPUT_BASE}/graphmert/tokenized" \
    --output_dir    "${OUTPUT_BASE}/graphmert/entity_discovery" \
    --model_id      "${MODEL_ID}" \
    --tokenizer     "${OUTPUT_BASE}/graphmert/tokenized/stable_tokenizer"
```

Find head token positions:

```bash
python 2_graphmert/utils/entity_discovery/find_heads_positions.py \
    --heads_chunks_dir "${OUTPUT_BASE}/graphmert/entity_discovery" \
    --output_dir       "${OUTPUT_BASE}/graphmert/head_positions" \
    --tokenizer        "${OUTPUT_BASE}/graphmert/tokenized/stable_tokenizer"
```

Or submit via SLURM:

```bash
sbatch --export=ALL 2_graphmert/slurm/entity_discovery.slurm
```

### Step 3 — Relation matching

Assign relation labels to head entities via LLM:

```bash
python 2_graphmert/utils/relation_matching/add_llm_relations.py \
    --dataset_path "${OUTPUT_BASE}/graphmert/head_positions" \
    --output_root  "${OUTPUT_BASE}/graphmert/relation_matching" \
    --model_id     "Qwen/Qwen2.5-72B-Instruct" \
    --tokenizer    "${OUTPUT_BASE}/graphmert/tokenized/stable_tokenizer"

python 2_graphmert/utils/relation_matching/clean_llm_relations.py \
    --input_dir  "${OUTPUT_BASE}/graphmert/relation_matching" \
    --output_dir "${OUTPUT_BASE}/graphmert/relation_matching_clean" \
    --tokenizer  "${OUTPUT_BASE}/graphmert/tokenized/stable_tokenizer"
```

### Step 4 — Build training dataset (co-occurrence grounding)

Grounds seed KG triples to text snippets to create MLM training samples.
This step replaces the old `graphmert_bridge.py`:

```bash
python 2_graphmert/run_dataset_preprocessing.py \
    launch_configs/args_mlm.yaml \
    --seed_kg_path "${SEED_KG_PATH}" \
    --tokenizer    "${OUTPUT_BASE}/graphmert/tokenized/stable_tokenizer" \
    --output_dir   "${OUTPUT_BASE}/graphmert/dataset"
```

Outputs:
- `${OUTPUT_BASE}/graphmert/dataset/ready_for_training_train/`
- `${OUTPUT_BASE}/graphmert/dataset/ready_for_training_eval/`
- `${OUTPUT_BASE}/graphmert/dataset/relation_map.json`

### Step 5 — Train GraphMERT

Edit `launch_configs/args_mlm.yaml` to set all paths (marked `<YOUR_SCRATCH>`),
then:

```bash
sbatch 2_graphmert/slurm/train_graphmert.slurm
# or directly:
python 2_graphmert/run_mlm.py launch_configs/args_mlm.yaml
```

Key hyperparameters in `args_mlm.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_nodes` | 512 | Tokens per root node |
| `num_leaves` | 3 | Leaf (tail) nodes per training sample |
| `max_nodes` | 2048 | Total node budget |
| `num_train_epochs` | 10 | Training epochs |
| `per_device_train_batch_size` | 8 | Batch size per GPU |

### Step 6 — Predict novel tail entities

Run inference with the trained GraphMERT checkpoint:

```bash
# Single shard:
python 2_graphmert/predict_tails_llm.py \
    --model_id   "${OUTPUT_BASE}/graphmert/checkpoints/best" \
    --tokenizer  "${OUTPUT_BASE}/graphmert/tokenized/stable_tokenizer" \
    --dataset    "${OUTPUT_BASE}/graphmert/dataset/ready_for_training_eval" \
    --output_dir "${OUTPUT_BASE}/graphmert/predictions" \
    --num_shards 4 --shard_id 0

# All shards via SLURM array:
sbatch --export=ALL 2_graphmert/slurm/predict_tails.slurm
```

### Step 7 — Merge predictions

```bash
python 2_graphmert/utils/combine_tails/combine_tails.py \
    --pred_dir   "${OUTPUT_BASE}/graphmert/predictions" \
    --output_dir "${OUTPUT_BASE}/graphmert/final_kg" \
    --model_id   "${OUTPUT_BASE}/graphmert/checkpoints/best"
```

### Step 8 — LLM fact scoring (two-model validation)

Keep only triples both LLMs agree are factually supported:

```bash
python 2_graphmert/utils/llm_scores/fact_score.py \
    --input_csv  "${OUTPUT_BASE}/graphmert/final_kg/expanded_triples.csv" \
    --output_csv "${OUTPUT_BASE}/graphmert/final_kg/expanded_triples_scored.csv" \
    --model_ids  "Qwen/Qwen2.5-72B-Instruct" "meta-llama/Llama-3.1-70B-Instruct"
```

---

## Architecture Constants

These must be consistent across tokenization, dataset preprocessing, and training:

| Constant | Value | Where set |
|----------|-------|-----------|
| `ROOT_NODES` | 512 | `args_mlm.yaml` → `root_nodes` |
| `NUM_LEAVES` | 3 | `args_mlm.yaml` → `num_leaves` |
| `MAX_NODES` | 2048 | `args_mlm.yaml` → `max_nodes` |

---

## Note on the Bridge Script

Previous versions of this pipeline included a `graphmert_bridge.py` that
performed tokenizer creation, relation map building, and co-occurrence grounding
as a separate post-processing step.  That script has been eliminated:

- Stable tokenizer creation → `run_tokenization.py` (Step 1)
- Relation map + grounding → `dataset_preprocessing_utils.py` (Step 4)

There is no longer a separate bridge step.
