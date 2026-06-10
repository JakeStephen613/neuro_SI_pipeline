# Part 3 — SI Curriculum Generation, SFT & RL Training

Generates a multi-hop neuroscience Q&A curriculum from the expanded KG
produced by Part 2, then fine-tunes a reasoning model using Supervised
Fine-Tuning (SFT) followed by Reinforcement Learning (RL / GRPO).

---

## Directory Layout

```
3_si_curriculum/
├── calculate_hops.py               # Pre-step: compute hop distances, build manifest
├── curriculum_generator/
│   ├── generate_questions.py       # Core Q&A generation logic (QAGenerator class)
│   ├── generate_curriculum.py      # CLI wrapper — generates curriculum.json
│   └── verify_questions.py         # Two-LLM verification filter
├── training/
│   ├── data_prep.py                # Convert curriculum → SFT HF dataset
│   ├── trainer.py                  # LoRA SFT training (torchrun / FSDP)
│   ├── merge_lora.py               # Merge LoRA adapter into base weights
│   └── fsdp_config_qwen.json       # FSDP config for multi-GPU training
├── RL/
│   ├── data_prep.py                # Convert curriculum → GRPO HF dataset
│   ├── rl_training.py              # GRPO RL training (TRL)
│   └── deepspeed_config.json       # DeepSpeed ZeRO config
├── test_models/
│   ├── eval_models.py              # Multi-checkpoint evaluation runner
│   ├── data_analysis.py            # Accuracy analysis & hop-breakdown plots
│   └── correctness_similarity.py   # Error-overlap & checkpoint progression analysis
└── slurm/
    ├── generate_curriculum.slurm
    ├── verify_questions.slurm
    ├── sft_trainer.slurm
    ├── rl_training.slurm
    └── eval_models.slurm
```

---

## Step-by-Step Usage

Set common variables first:

```bash
export REPO_DIR=/path/to/neuro_SI_pipeline
export OUTPUT_BASE=/path/to/your/scratch/output
export FINAL_KG="${OUTPUT_BASE}/graphmert/final_kg"   # output of Part 2
```

---

### Pre-step — Compute hop distances

Build a manifest that records the minimum hop distance between every entity
pair in the expanded KG.  This drives hop-stratified sampling during
curriculum generation.

```bash
python 3_si_curriculum/calculate_hops.py \
    --kg_path      "${FINAL_KG}/expanded_triples_scored.csv" \
    --seed_kg_path "${OUTPUT_BASE}/kg_final.parquet" \
    --output_path  "${OUTPUT_BASE}/curriculum/kg_manifest.json"
```

---

### Step 1 — Generate curriculum Q&A items

```bash
python 3_si_curriculum/curriculum_generator/generate_curriculum.py \
    --manifest_path "${OUTPUT_BASE}/curriculum/kg_manifest.json" \
    --output_dir    "${OUTPUT_BASE}/curriculum" \
    --min_hops      3 \
    --max_hops      5 \
    --target_count  50000 \
    --api_key       "${OPENAI_API_KEY}"   # or set via env
```

Output: `${OUTPUT_BASE}/curriculum/curriculum.json`

Or via SLURM:

```bash
sbatch --export=ALL 3_si_curriculum/slurm/generate_curriculum.slurm
```

---

### Step 2 — Two-LLM verification

Keep only Q&A items where two independent LLMs agree on the answer:

```bash
python 3_si_curriculum/curriculum_generator/verify_questions.py \
    --input_json  "${OUTPUT_BASE}/curriculum/curriculum.json" \
    --output_json "${OUTPUT_BASE}/curriculum_verified/curriculum_verified.json" \
    --model_ids   "Qwen/Qwen3-14B" "meta-llama/Llama-3.1-8B-Instruct"
```

Requires exactly 2 `--model_ids`.  Items where the two models disagree are
discarded.

---

### Step 3 — SFT data preparation

```bash
INPUT_PATH="${OUTPUT_BASE}/curriculum_verified/curriculum_verified.json" \
OUTPUT_PATH="${OUTPUT_BASE}/sft_dataset" \
python 3_si_curriculum/training/data_prep.py
# MODE defaults to "sft"; edit top of file to change LAST_N / EVAL_SPLIT_RATIO
```

---

### Step 4 — SFT training (LoRA)

```bash
export MODEL_NAME=/path/to/base/model    # HF model id or local path
export DATASET_PATH="${OUTPUT_BASE}/sft_dataset"
export WANDB_DIR="${OUTPUT_BASE}/wandb_logs"

# Multi-GPU (4 GPUs):
torchrun --nproc_per_node=4 3_si_curriculum/training/trainer.py \
    --model_name         "${MODEL_NAME}" \
    --train_dataset_path "${DATASET_PATH}" \
    --output_dir         "${OUTPUT_BASE}/sft_checkpoints" \
    --wandb_dir          "${WANDB_DIR}"

# Or via SLURM:
sbatch --export=ALL 3_si_curriculum/slurm/sft_trainer.slurm
```

---

### Step 5 — Merge LoRA adapter

Merge the LoRA weights into the base model for RL or deployment:

```bash
python 3_si_curriculum/training/merge_lora.py \
    --base_model  "${MODEL_NAME}" \
    --adapter_path "${OUTPUT_BASE}/sft_checkpoints/checkpoint-XXXX"
# Saves merged model to: ${OUTPUT_BASE}/sft_checkpoints/checkpoint-XXXX/merged_final_model/
```

---

### Step 6 — RL data preparation

```bash
INPUT_PATH="${OUTPUT_BASE}/curriculum_verified/curriculum_verified.json" \
OUTPUT_PATH="${OUTPUT_BASE}/rl_dataset" \
python 3_si_curriculum/RL/data_prep.py
# Set MODE="rl" at top of file (default)
```

---

### Step 7 — RL training (GRPO)

```bash
export MODEL_NAME="${OUTPUT_BASE}/sft_checkpoints/checkpoint-XXXX/merged_final_model"
export DATASET_PATH="${OUTPUT_BASE}/rl_dataset"

python 3_si_curriculum/RL/rl_training.py \
    --model_name_or_path "${MODEL_NAME}" \
    --output_dir         "${OUTPUT_BASE}/rl_checkpoints" \
    --dataset_path       "${DATASET_PATH}"

# Resume from checkpoint:
RESUME_CHECKPOINT="${OUTPUT_BASE}/rl_checkpoints/checkpoint-YYY" \
python 3_si_curriculum/RL/rl_training.py ...

# Or via SLURM:
sbatch --export=ALL 3_si_curriculum/slurm/rl_training.slurm
```

---

### Step 8 — Evaluation

```bash
export EVAL_INPUT_DIR="${OUTPUT_BASE}/curriculum_verified"
export EVAL_OUTPUT_DIR="${OUTPUT_BASE}/eval_results"
export MODEL_PATH_1="${OUTPUT_BASE}/rl_checkpoints/checkpoint-1000"
export MODEL_PATH_2="${OUTPUT_BASE}/rl_checkpoints/checkpoint-2000"

python 3_si_curriculum/test_models/eval_models.py \
    --input_dir  "${EVAL_INPUT_DIR}" \
    --output_dir "${EVAL_OUTPUT_DIR}"

# Or via SLURM:
sbatch --export=ALL 3_si_curriculum/slurm/eval_models.slurm
```

---

### Step 9 — Analysis

```bash
# Accuracy by hop count + CSV + plot:
EVAL_INPUT_DIR="${EVAL_OUTPUT_DIR}" \
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}/analysis" \
python 3_si_curriculum/test_models/data_analysis.py

# Error overlap & checkpoint progression:
EVAL_INPUT_FILE="${EVAL_OUTPUT_DIR}/eval_results.json" \
python 3_si_curriculum/test_models/correctness_similarity.py
```

---

## Curriculum Format

Each item in `curriculum.json` follows this schema:

```json
{
  "question_and_explanation": "<Question>...</Question><Options>...</Options><Explanation>...</Explanation><Answer>X</Answer>",
  "k_hops": 3,
  "paths": ["entity_A -> rel -> entity_B -> rel -> entity_C"]
}
```

The `k_hops` field is used for hop-stratified sampling and evaluation breakdown.

---

## Training Config Notes

| File | Purpose |
|------|---------|
| `training/fsdp_config_qwen.json` | FSDP sharding for multi-GPU SFT |
| `RL/deepspeed_config.json` | DeepSpeed ZeRO-2 for RL training |

Both configs are ready to use.  Adjust `num_processes` and `per_device_*_batch_size`
to match your GPU allocation.
