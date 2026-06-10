# Part 1 — Seed Knowledge Graph Generation

Extracts a structured neuroscience knowledge graph from a corpus of PDFs/text
using a fine-tuned LLM (GraphRAG).  The output is a set of `(head, relation,
tail)` triples that serve as the seed KG for Part 2.

---

## Directory Layout

```
1_seed_kg/
├── graphrag_index.py       # Main extraction pipeline (steps 1–5)
├── count_text_units.py     # Count chunks to size SLURM array jobs
├── merge_kgs.py            # Merge incremental KG shards
├── settings.yaml           # GraphRAG run settings
├── prompts_kg.py           # Relation-type prompts for LLM extraction
├── llm_kg/                 # Standalone LLM-only KG builder (no GraphRAG)
│   ├── settings.yaml
│   └── README.md
└── slurm/
    └── job.slurm           # Array job template
```

---

## Quickstart

### 1. Set environment variables

```bash
export REPO_DIR=/path/to/neuro_SI_pipeline
export OUTPUT_BASE=/path/to/your/scratch/output
export MODEL_ID=Qwen/Qwen2.5-72B-Instruct   # or any vLLM-compatible model
```

### 2. Prepare your corpus

Place your text files (`.txt` or `.pdf`) inside a directory, e.g.:
```
${OUTPUT_BASE}/corpus/
```

### 3. Count text units (sets SLURM array size)

```bash
python 1_seed_kg/count_text_units.py \
    --root_dir  "${OUTPUT_BASE}" \
    --rows_per_job 512
# Prints: "Use --array=0-<N>" for the next step
```

### 4. Run extraction (steps 1–5 sequentially, or via SLURM array)

```bash
# Step 1 — build text unit chunks
python 1_seed_kg/graphrag_index.py --root_dir "${OUTPUT_BASE}" --step 1

# Step 2 — build document records
python 1_seed_kg/graphrag_index.py --root_dir "${OUTPUT_BASE}" --step 2

# Step 3 — LLM extraction (parallelise over shards via SLURM array)
#           SLURM_ARRAY_TASK_ID controls the shard index
python 1_seed_kg/graphrag_index.py \
    --root_dir "${OUTPUT_BASE}" \
    --step 3 \
    --model_id "${MODEL_ID}"

# Step 4 — parse LLM responses into triples
python 1_seed_kg/graphrag_index.py --root_dir "${OUTPUT_BASE}" --step 4

# Step 5 — clean & finalise KG
python 1_seed_kg/graphrag_index.py --root_dir "${OUTPUT_BASE}" --step 5
```

Or submit everything via SLURM:

```bash
sbatch --export=ALL \
    --array=0-<N> \
    1_seed_kg/slurm/job.slurm
```

### 5. (Optional) Merge KG shards

If you ran extraction in multiple batches:

```bash
python 1_seed_kg/merge_kgs.py \
    --new "${OUTPUT_BASE}/output/kg_new.parquet" \
    --old "${OUTPUT_BASE}/output/kg_old.parquet" \
    --out "${OUTPUT_BASE}/output/kg_merged.parquet"
```

---

## Output

| File | Description |
|------|-------------|
| `${OUTPUT_BASE}/output/kg_final.parquet` | Final seed KG — head/relation/tail triples |
| `${OUTPUT_BASE}/output/text_units/`      | Tokenised text chunks (used by Part 2)     |

The `kg_final.parquet` path should be passed as `--seed_kg_path` in Part 2.

---

## Relation Types

Defined in `prompts_kg.py` — covers neuroscience-specific relations such as
`regulates`, `inhibits`, `activates`, `expressed_in`, `associated_with`, etc.
Edit this file to extend or restrict the relation vocabulary.

---

## Settings

`settings.yaml` controls chunk size, overlap, LLM sampling parameters, and
graph community detection.  Key fields to review:

```yaml
chunks:
  size: 512          # tokens per chunk
  overlap: 64
llm:
  max_tokens: 4096
  temperature: 0.0
```
