import os
import json
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/HPC environments
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = os.environ.get("EVAL_INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR", "")

# --- Multiple input files (set to None to skip) ---
INPUT_2 = None
INPUT_1 = "eval_single_hf_1k_test_3_to_5.json"
INPUT_3 = None  # e.g. "eval_third_results.json"

OUTPUT_FILENAME = "final_model_analysis_new.csv"

# --- Model selection & aliasing ---
# Set to None to auto-detect all models (no filtering/renaming)
# MODEL_SELECTION = None

MODEL_SELECTION = {
    "checkpoint-1000": "checkpoint-1000",
    "checkpoint-1300": "checkpoint-1300",
}

# ==========================================
# MAIN SCRIPT
# ==========================================

def load_json_items(filepath):
    """Load a JSON file and return a list of items."""
    if not os.path.exists(filepath):
        print(f"  WARNING: File not found, skipping: {filepath}")
        return []
    try:
        with open(filepath, 'r') as f:
            content = json.load(f)
            if isinstance(content, list):
                return content
            elif isinstance(content, dict):
                return [content]
    except Exception as e:
        print(f"  ERROR: Failed to read {os.path.basename(filepath)}: {e}")
    return []


def analyze_performance():
    print(f"--- Starting Analysis ---")

    # 1. Load & merge data from up to 3 input sources
    all_items = []
    for i, fname in enumerate([INPUT_1, INPUT_2, INPUT_3], start=1):
        if fname is None:
            continue
        path = os.path.join(INPUT_DIR, fname)
        print(f"Loading INPUT_{i}: {path}")
        items = load_json_items(path)
        print(f"  -> {len(items)} items loaded")
        all_items.extend(items)

    if not all_items:
        print("CRITICAL ERROR: No data loaded from any input file.")
        return

    print(f"\nTotal items after merge: {len(all_items)}")

    # 2. Detect models from correctness_ keys
    detected_models = set()
    for item in all_items:
        for key in item.keys():
            if key.startswith("correctness_"):
                detected_models.add(key.replace("correctness_", ""))

    detected_models = sorted(list(detected_models))
    if not detected_models:
        print("No model data found. (Did you look for 'correctness_' keys?)")
        return

    print(f"Models Detected: {', '.join(detected_models)}")

    # 3. Resolve model list & aliases
    if MODEL_SELECTION is not None:
        # Filter to only selected models, apply aliases
        models = [m for m in MODEL_SELECTION.keys() if m in detected_models]
        missing = [m for m in MODEL_SELECTION.keys() if m not in detected_models]
        if missing:
            print(f"WARNING: Requested models not found in data: {', '.join(missing)}")
        aliases = {m: MODEL_SELECTION[m] for m in models}
    else:
        models = detected_models
        aliases = {m: m for m in models}

    if not models:
        print("No matching models to analyze.")
        return

    print(f"Analyzing {len(models)} model(s): {', '.join(aliases[m] for m in models)}")

    # 4. Initialize Metrics
    stats_overall = {m: {"correct": 0, "total": 0, "errors": 0} for m in models}
    stats_hops = {m: defaultdict(lambda: {"correct": 0, "total": 0}) for m in models}

    # 5. Calculate Stats
    for item in all_items:
        k_hops = item.get("k_hops", "Unknown")

        for m in models:
            result_key = f"correctness_{m}"
            if result_key not in item:
                continue

            val = str(item[result_key]).lower()
            if val == "n/a":
                continue

            stats_overall[m]["total"] += 1
            stats_hops[m][k_hops]["total"] += 1

            if val == "yes":
                stats_overall[m]["correct"] += 1
                stats_hops[m][k_hops]["correct"] += 1
            elif "error" in val:
                stats_overall[m]["errors"] += 1

    # 6. Build Report & Print
    report_rows = []

    # --- Part A: Overall ---
    print("\n" + "=" * 80)
    print(f"{' OVERALL PERFORMANCE ':^80}")
    print("=" * 80)
    print(f"{'Model':<30} | {'Acc %':<10} | {'Correct':<8} | {'Total':<8} | {'Parse Err':<10}")
    print("-" * 80)

    for m in models:
        t = stats_overall[m]["total"]
        if t == 0:
            continue
        c = stats_overall[m]["correct"]
        e = stats_overall[m]["errors"]
        acc = (c / t) * 100

        display = aliases[m]
        print(f"{display:<30} | {acc:6.2f}%    | {c:<8} | {t:<8} | {e:<10}")

        report_rows.append({
            "Model": display,
            "Metric_Type": "Overall",
            "Category": "All",
            "Accuracy": acc,
            "Correct": c,
            "Total": t,
            "Parse_Errors": e
        })

    # --- Part B: By Hops ---
    print("\n" + "=" * 80)
    print(f"{' REASONING DEPTH BREAKDOWN (HOPS) ':^80}")
    print("=" * 80)

    all_hops = set()
    for m in models:
        all_hops.update(stats_hops[m].keys())

    try:
        sorted_hops = sorted(list(all_hops), key=lambda x: int(x))
    except:
        sorted_hops = sorted(list(all_hops))

    for h in sorted_hops:
        print(f"\n--- {h}-Hop Questions ---")
        print(f"{'Model':<30} | {'Acc %':<10} | {'Correct':<8} | {'Total':<8}")
        print("-" * 65)

        for m in models:
            s = stats_hops[m][h]
            t = s["total"]
            if t == 0:
                continue
            c = s["correct"]
            acc = (c / t) * 100
            display = aliases[m]

            print(f"{display:<30} | {acc:6.2f}%    | {c:<8} | {t:<8}")

            report_rows.append({
                "Model": display,
                "Metric_Type": "Hop_Breakdown",
                "Category": f"{h}-Hop",
                "Accuracy": acc,
                "Correct": c,
                "Total": t,
                "Parse_Errors": 0
            })

    # 7. Save CSV
    df = pd.DataFrame(report_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print(f"COMPLETE. Metrics saved to: {out_path}")
    print("=" * 80)

    # 8. Plot
    plot_hop_accuracy(models, aliases, stats_hops, OUTPUT_DIR)


def plot_hop_accuracy(models, aliases, stats_hops, output_dir):
    """Generate a publication-quality line chart of accuracy vs. hop count
    with a dynamic y-axis that zooms into the actual data range."""

    # Colorblind-friendly palette (Wong 2011)
    PALETTE = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # pink
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
    ]

    HOP_LABELS = [3, 4, 5]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    all_accs = []  # collect all plotted accuracies for dynamic y-axis
    plotted_any = False
    for i, m in enumerate(models):
        accs = []
        valid_hops = []
        for h in HOP_LABELS:
            s = stats_hops[m].get(h) or stats_hops[m].get(str(h))
            if s and s["total"] > 0:
                acc = (s["correct"] / s["total"]) * 100
                accs.append(acc)
                valid_hops.append(h)
        if not valid_hops:
            continue
        all_accs.extend(accs)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            valid_hops, accs,
            marker="o", linewidth=2.0, markersize=7,
            label=aliases[m], color=color, zorder=3
        )
        plotted_any = True

    if not plotted_any:
        print("WARNING: No hop data found to plot.")
        plt.close(fig)
        return

    # Dynamic y-axis: pad 10% below min and above max, rounded to nearest 5
    data_min = min(all_accs)
    data_max = max(all_accs)
    data_range = data_max - data_min if data_max > data_min else 10
    padding = max(data_range * 0.25, 3)  # at least 3pp padding

    y_lo = max(0, 5 * int((data_min - padding) / 5))
    y_hi = min(100, 5 * int((data_max + padding) / 5 + 1))

    ax.set_xticks(HOP_LABELS)
    ax.set_xlabel("Number of Reasoning Hops")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("Model Accuracy by Reasoning Depth")

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="gray")
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="best", framealpha=0.9, edgecolor="0.8")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "hop_accuracy_plot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    analyze_performance()