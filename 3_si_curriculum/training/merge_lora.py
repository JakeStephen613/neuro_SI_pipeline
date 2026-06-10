import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def merge_adapter(base_model_path, adapter_path):
    print(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    print(f"Reloading base model: {base_model_path}")
    # ML ENGINEER FIX: Force CPU loading to prevent VRAM OOM spikes during merge.
    # We rely on your 400GB of SLURM system RAM instead of the GPUs.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        trust_remote_code=True
    )

    # We must resize embeddings on the base model so the shapes match the adapter
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"Loading LoRA adapter from: {adapter_path}")
    model_to_merge = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights (this will take a few minutes on CPU)...")
    merged_model = model_to_merge.merge_and_unload()

    merged_output_dir = os.path.join(adapter_path, "merged_final_model")
    os.makedirs(merged_output_dir, exist_ok=True)

    print(f"Saving fully merged model to: {merged_output_dir}")
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)

    # Clean up massive RAM footprint
    del merged_model
    del base_model
    gc.collect()

    print(f"✅ MERGE COMPLETE. Model saved to {merged_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to your saved checkpoint directory")
    args = parser.parse_args()

    merge_adapter(args.base_model, args.adapter_path)