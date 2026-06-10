#!/usr/bin/env python3
"""
Checkpoint Error Overlap & Diagnostic Analysis
================================================
Auto-detects ALL models from correctness_* keys in the eval JSON,
validates completeness per-item, and computes detailed error overlap,
divergence, and difficulty metrics across every model found.

Usage:
    python checkpoint_error_analysis.py [path_to_json]
"""

import os
import sys
import json
from collections import defaultdict
from itertools import combinations

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_INPUT = os.environ.get("EVAL_INPUT_FILE", "")

# ==========================================
# HELPERS
# ==========================================

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def detect_models(items):
    models = set()
    for item in items:
        for k in item:
            if k.startswith("correctness_"):
                models.add(k.replace("correctness_", ""))
    return sorted(models)


def get_correctness(item, model):
    val = str(item.get(f"correctness_{model}", "")).strip().lower()
    if val == "yes":
        return True
    elif val == "no":
        return False
    elif val in ("", "n/a") or "error" in val:
        return None
    return False


def get_item_id(item, idx):
    for key in ["id", "item_id", "question_id", "idx", "index"]:
        if key in item:
            return str(item[key])
    return f"item_{idx}"


def fmt_pct(num, denom):
    if denom == 0:
        return "   N/A"
    return f"{(num / denom) * 100:6.2f}%"


# ==========================================
# MAIN
# ==========================================

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    print(f"Loading: {input_path}")
    items = load_data(input_path)
    N = len(items)
    print(f"Total items loaded: {N}")

    models = detect_models(items)
    print(f"Models detected ({len(models)}): {', '.join(models)}\n")

    # =====================================================================
    # Build per-item result matrix
    # =====================================================================
    result_matrix = []
    hop_labels = []
    item_ids = []

    for idx, item in enumerate(items):
        iid = get_item_id(item, idx)
        item_ids.append(iid)
        hops = item.get("k_hops")
        hop_labels.append(hops)
        row = {}
        for m in models:
            row[m] = get_correctness(item, m)
        result_matrix.append(row)

    hop_distribution = defaultdict(int)
    for h in hop_labels:
        if h is not None:
            hop_distribution[h] += 1

    # =====================================================================
    # SECTION 1: DATA INTEGRITY
    # =====================================================================
    print("=" * 90)
    print(f"{'SECTION 1: DATA INTEGRITY & COVERAGE':^90}")
    print("=" * 90)

    print(f"\n{'Model':<40} | {'Answered':<9} | {'Correct':<9} | {'Wrong':<9} | {'Missing/Err':<11} | {'Acc %':<8}")
    print("-" * 90)

    model_correct = {}
    model_wrong = {}
    model_valid = {}

    for m in models:
        correct = sum(1 for row in result_matrix if row[m] is True)
        wrong = sum(1 for row in result_matrix if row[m] is False)
        missing = sum(1 for row in result_matrix if row[m] is None)
        answered = correct + wrong
        model_correct[m] = correct
        model_wrong[m] = wrong
        model_valid[m] = answered
        print(f"{m:<40} | {answered:<9} | {correct:<9} | {wrong:<9} | {missing:<11} | {fmt_pct(correct, answered)}")

    # --- Filter: keep only models that answered ALL N questions ---
    incomplete = [m for m in models if model_valid[m] < N]
    if incomplete:
        print(f"\n*** REMOVED {len(incomplete)} model(s) with incomplete test sets: ***")
        for m in incomplete:
            print(f"  - {m}: answered {model_valid[m]}/{N} ({fmt_pct(model_valid[m], N).strip()})")
        models = [m for m in models if model_valid[m] == N]
        print(f"\nContinuing with {len(models)} fully-evaluated model(s): {', '.join(models)}")

    print(f"\nHop distribution across {N} items:")
    sorted_hops = sorted(hop_distribution.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
    for h in sorted_hops:
        print(f"  {h}-hop: {hop_distribution[h]}")
    missing_hop_count = sum(1 for h in hop_labels if h is None)
    if missing_hop_count:
        print(f"  Missing hop label: {missing_hop_count}")

    # =====================================================================
    # SECTION 2: PER-HOP ACCURACY (all models)
    # =====================================================================
    print("\n" + "=" * 90)
    print(f"{'SECTION 2: PER-HOP ACCURACY (ALL MODELS)':^90}")
    print("=" * 90)

    for h in sorted_hops:
        indices = [i for i, hl in enumerate(hop_labels) if hl == h]
        print(f"\n--- {h}-Hop ({len(indices)} questions) ---")
        print(f"{'Model':<40} | {'Correct':<9} | {'Wrong':<9} | {'Acc %':<8}")
        print("-" * 72)
        for m in models:
            c = sum(1 for i in indices if result_matrix[i][m] is True)
            w = sum(1 for i in indices if result_matrix[i][m] is False)
            total = c + w
            print(f"{m:<40} | {c:<9} | {w:<9} | {fmt_pct(c, total)}")

    # =====================================================================
    # SECTION 3: PAIRWISE ERROR OVERLAP
    # =====================================================================
    print("\n" + "=" * 90)
    print(f"{'SECTION 3: PAIRWISE ERROR OVERLAP':^90}")
    print("=" * 90)
    print("Jaccard = |both wrong| / |either wrong|.  High = same errors.  Low = different errors.")
    print("Overlap% = |both wrong| / |fewer wrong model's errors|.  Subset measure.\n")

    wrong_sets_global = {m: set() for m in models}
    for i, row in enumerate(result_matrix):
        for m in models:
            if row[m] is False:
                wrong_sets_global[m].add(i)

    print(f"{'Model A':<25} {'Model B':<25} | {'Both':>5} | {'Only A':>6} | {'Only B':>6} | {'Jaccard':>7} | {'Ovlp%':>6}")
    print("-" * 95)

    for m_a, m_b in combinations(models, 2):
        common_valid = set(i for i, row in enumerate(result_matrix) if row[m_a] is not None and row[m_b] is not None)
        wrong_a = wrong_sets_global[m_a] & common_valid
        wrong_b = wrong_sets_global[m_b] & common_valid
        both = wrong_a & wrong_b
        only_a = wrong_a - wrong_b
        only_b = wrong_b - wrong_a
        union = wrong_a | wrong_b

        jaccard = len(both) / len(union) if union else 0
        smaller = min(len(wrong_a), len(wrong_b))
        overlap_pct = len(both) / smaller if smaller else 0

        print(f"{m_a:<25} {m_b:<25} | {len(both):>5} | {len(only_a):>6} | {len(only_b):>6} | {jaccard:>6.3f} | {overlap_pct:>5.1%}")

    # =====================================================================
    # SECTION 4: ERROR OVERLAP BY HOP
    # =====================================================================
    print("\n" + "=" * 90)
    print(f"{'SECTION 4: PAIRWISE ERROR OVERLAP BY HOP':^90}")
    print("=" * 90)

    for h in sorted_hops:
        indices_set = set(i for i, hl in enumerate(hop_labels) if hl == h)
        print(f"\n--- {h}-Hop ---")
        print(f"{'Model A':<25} {'Model B':<25} | {'Both':>5} | {'Only A':>6} | {'Only B':>6} | {'Jaccard':>7}")
        print("-" * 85)
        for m_a, m_b in combinations(models, 2):
            valid = set(i for i in indices_set if result_matrix[i][m_a] is not None and result_matrix[i][m_b] is not None)
            wa = set(i for i in valid if result_matrix[i][m_a] is False)
            wb = set(i for i in valid if result_matrix[i][m_b] is False)
            both = wa & wb
            union = wa | wb
            jac = len(both) / len(union) if union else 0
            print(f"{m_a:<25} {m_b:<25} | {len(both):>5} | {len(wa - wb):>6} | {len(wb - wa):>6} | {jac:>6.3f}")

    # =====================================================================
    # SECTION 5: QUESTION DIFFICULTY
    # =====================================================================
    print("\n" + "=" * 90)
    print(f"{'SECTION 5: QUESTION DIFFICULTY DISTRIBUTION':^90}")
    print("=" * 90)
    print("For each item: count how many models got it wrong (among those that answered).\n")

    difficulty_dist = defaultdict(int)
    difficulty_by_hop = defaultdict(lambda: defaultdict(int))
    hardest_items = []

    for i, row in enumerate(result_matrix):
        valid_models = [m for m in models if row[m] is not None]
        num_wrong = sum(1 for m in valid_models if row[m] is False)
        num_valid = len(valid_models)
        difficulty_dist[num_wrong] += 1
        h = hop_labels[i]
        if h is not None:
            difficulty_by_hop[h][num_wrong] += 1
        if num_wrong == num_valid and num_valid == len(models):
            hardest_items.append(i)

    print(f"{'Models Wrong':<15} | {'# Questions':<12} | {'% of Total':<10}")
    print("-" * 42)
    for k in sorted(difficulty_dist.keys()):
        label = f"{k}/{len(models)}"
        print(f"{label:<15} | {difficulty_dist[k]:<12} | {fmt_pct(difficulty_dist[k], N)}")

    print(f"\nBy hop:")
    for h in sorted_hops:
        print(f"\n  {h}-Hop:")
        for k in sorted(difficulty_by_hop[h].keys()):
            cnt = difficulty_by_hop[h][k]
            hop_total = hop_distribution[h]
            print(f"    {k}/{len(models)} models wrong: {cnt} questions ({fmt_pct(cnt, hop_total).strip()})")

    print(f"\nUniversally wrong (ALL {len(models)} models wrong): {len(hardest_items)} questions")
    if hardest_items and len(hardest_items) <= 50:
        print(f"  Item indices: {hardest_items[:50]}")

    # =====================================================================
    # SECTION 6: UNIQUE ERRORS PER MODEL
    # =====================================================================
    print("\n" + "=" * 90)
    print(f"{'SECTION 6: UNIQUE ERRORS PER MODEL':^90}")
    print("=" * 90)
    print("Questions that ONLY this model got wrong (all others that answered got right).\n")

    print(f"{'Model':<40} | {'Unique Errors':<14} | {'Total Errors':<12} | {'% Unique':<8}")
    print("-" * 80)

    unique_errors = {}
    for m in models:
        unique = []
        total_wrong = 0
        for i, row in enumerate(result_matrix):
            if row[m] is False:
                total_wrong += 1
                others_all_right = all(row[m2] is not False for m2 in models if m2 != m and row[m2] is not None)
                if others_all_right:
                    unique.append(i)
        unique_errors[m] = unique
        unique_pct = fmt_pct(len(unique), total_wrong) if total_wrong else "   N/A"
        print(f"{m:<40} | {len(unique):<14} | {total_wrong:<12} | {unique_pct}")

    print(f"\nUnique errors by hop:")
    for h in sorted_hops:
        indices_set = set(i for i, hl in enumerate(hop_labels) if hl == h)
        print(f"\n  {h}-Hop:")
        for m in models:
            unique = sum(1 for i in indices_set if i in set(unique_errors[m]))
            if unique > 0:
                print(f"    {m:<40}: {unique} unique errors")

    # =====================================================================
    # SECTION 7: CHECKPOINT PROGRESSION
    # =====================================================================
    print("\n" + "=" * 90)
    print(f"{'SECTION 7: CHECKPOINT PROGRESSION':^90}")
    print("=" * 90)
    print("Models sorted by checkpoint step to visualize training progression.\n")

    def extract_ckpt_num(name):
        if "checkpoint-" in name:
            try:
                return int(name.split("checkpoint-")[1])
            except ValueError:
                pass
        return None

    ckpt_models = [(m, extract_ckpt_num(m)) for m in models if extract_ckpt_num(m) is not None]
    ckpt_models.sort(key=lambda x: x[1])
    non_ckpt_models = [m for m in models if extract_ckpt_num(m) is None]

    if ckpt_models:
        print(f"{'Checkpoint':<20} | {'Overall':>8}", end="")
        for h in sorted_hops:
            print(f" | {h}-Hop", end="")
        print(f" | {'Uniq Err':>8}")
        print("-" * (40 + 9 * len(sorted_hops)))

        for m, step in ckpt_models:
            print(f"{m:<20} | {fmt_pct(model_correct[m], model_valid[m])}", end="")
            for h in sorted_hops:
                idxs = [i for i, hl in enumerate(hop_labels) if hl == h]
                c = sum(1 for i in idxs if result_matrix[i][m] is True)
                v = sum(1 for i in idxs if result_matrix[i][m] is not None)
                print(f" | {fmt_pct(c, v)}", end="")
            uniq = len(unique_errors[m])
            print(f" | {uniq:>8}")

    if non_ckpt_models:
        print(f"\nOther models:")
        print(f"{'Model':<40} | {'Overall':>8}", end="")
        for h in sorted_hops:
            print(f" | {h}-Hop", end="")
        print(f" | {'Uniq Err':>8}")
        print("-" * (60 + 9 * len(sorted_hops)))
        for m in non_ckpt_models:
            print(f"{m:<40} | {fmt_pct(model_correct[m], model_valid[m])}", end="")
            for h in sorted_hops:
                idxs = [i for i, hl in enumerate(hop_labels) if hl == h]
                c = sum(1 for i in idxs if result_matrix[i][m] is True)
                v = sum(1 for i in idxs if result_matrix[i][m] is not None)
                print(f" | {fmt_pct(c, v)}", end="")
            uniq = len(unique_errors[m])
            print(f" | {uniq:>8}")

    # =====================================================================
    # SECTION 8: PAIRWISE AGREEMENT MATRIX
    # =====================================================================
    print("\n" + "=" * 90)
    print(f"{'SECTION 8: PAIRWISE AGREEMENT RATE':^90}")
    print("=" * 90)
    print("% of shared questions where both models gave the same answer.\n")

    col_w = 11
    header = " " * 28
    for m in models:
        header += f"{m[:10]:>{col_w}}"
    print(header)
    print("-" * len(header))

    for m_a in models:
        row_str = f"{m_a:<28}"
        for m_b in models:
            if m_a == m_b:
                row_str += f"{'---':>{col_w}}"
                continue
            agree = 0
            valid = 0
            for i, row in enumerate(result_matrix):
                if row[m_a] is not None and row[m_b] is not None:
                    valid += 1
                    if row[m_a] == row[m_b]:
                        agree += 1
            pct = f"{(agree/valid)*100:.1f}%" if valid else "N/A"
            row_str += f"{pct:>{col_w}}"
        print(row_str)

    # =====================================================================
    # SECTION 9: FLIPS BETWEEN CONSECUTIVE CHECKPOINTS
    # =====================================================================
    if len(ckpt_models) >= 2:
        print("\n" + "=" * 90)
        print(f"{'SECTION 9: FLIPS BETWEEN CONSECUTIVE CHECKPOINTS':^90}")
        print("=" * 90)
        print("Regressions = was right, now wrong.  Gains = was wrong, now right.\n")

        for j in range(1, len(ckpt_models)):
            prev_m, prev_step = ckpt_models[j - 1]
            curr_m, curr_step = ckpt_models[j]
            gains = 0
            regressions = 0
            gains_by_hop = defaultdict(int)
            regr_by_hop = defaultdict(int)

            for i, row in enumerate(result_matrix):
                prev_val = row[prev_m]
                curr_val = row[curr_m]
                if prev_val is None or curr_val is None:
                    continue
                h = hop_labels[i]
                if prev_val is False and curr_val is True:
                    gains += 1
                    if h is not None:
                        gains_by_hop[h] += 1
                elif prev_val is True and curr_val is False:
                    regressions += 1
                    if h is not None:
                        regr_by_hop[h] += 1

            net = gains - regressions
            print(f"  {prev_m} -> {curr_m}:")
            print(f"    Gains (wrong->right): {gains}")
            print(f"    Regressions (right->wrong): {regressions}")
            print(f"    Net improvement: {net:+d}")
            for h in sorted_hops:
                g = gains_by_hop.get(h, 0)
                r = regr_by_hop.get(h, 0)
                print(f"      {h}-hop: +{g} gains, -{r} regressions (net {g-r:+d})")
            print()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("=" * 90)
    print(f"{'SUMMARY':^90}")
    print("=" * 90)

    total_universally_wrong = len(hardest_items)
    total_universally_right = sum(
        1 for i, row in enumerate(result_matrix)
        if all(row[m] is True for m in models if row[m] is not None)
        and sum(1 for m in models if row[m] is not None) == len(models)
    )
    contested = N - total_universally_right - total_universally_wrong

    print(f"\n  Total questions:             {N}")
    print(f"  Total models:                {len(models)}")
    print(f"  Universally correct:         {total_universally_right} ({fmt_pct(total_universally_right, N).strip()})")
    print(f"  Universally wrong:           {total_universally_wrong} ({fmt_pct(total_universally_wrong, N).strip()})")
    print(f"  Contested (models disagree): {contested} ({fmt_pct(contested, N).strip()})")
    print()

    best_m = max(models, key=lambda m: model_correct[m] / model_valid[m] if model_valid[m] else 0)
    worst_m = min(models, key=lambda m: model_correct[m] / model_valid[m] if model_valid[m] else 1)
    print(f"  Best overall:  {best_m} ({fmt_pct(model_correct[best_m], model_valid[best_m]).strip()})")
    print(f"  Worst overall: {worst_m} ({fmt_pct(model_correct[worst_m], model_valid[worst_m]).strip()})")
    print()


if __name__ == "__main__":
    main()