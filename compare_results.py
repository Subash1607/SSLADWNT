"""
compare_results.py
==================
Compare Baseline vs Adaptive NeuTraL-AD results on NATOPS.

Reads from:
    RESULTS/RESULTS_BASELINE_natops/assessment_results.json   (or RESULTS_natops/)
    RESULTS/RESULTS_ADAPTIVE_natops/assessment_results.json

Prints a formatted comparison table and saves:
    RESULTS/comparison_baseline_vs_adaptive.json
    RESULTS/comparison_baseline_vs_adaptive.txt

Usage:
    python compare_results.py
    python compare_results.py --baseline RESULTS/RESULTS_natops --adaptive RESULTS/RESULTS_ADAPTIVE_natops
"""

import os
import json
import argparse
import numpy as np

BASELINE_CLASS_RESULTS = {
    "Class 1": {"AUROC": 0.9331, "AP": 0.9853, "F1": 0.9400},
    "Class 2": {"AUROC": 0.8522, "AP": 0.9698, "F1": 0.8933},
    "Class 3": {"AUROC": 0.9076, "AP": 0.9815, "F1": 0.9000},
    "Class 4": {"AUROC": 1.0000, "AP": 1.0000, "F1": 1.0000},
    "Class 5": {"AUROC": 0.9996, "AP": 0.9999, "F1": 0.9933},
    "Class 6": {"AUROC": 0.9987, "AP": 0.9997, "F1": 0.9867},
}
BASELINE_MEAN = {"AUROC": 0.9485, "AP": 0.9894, "F1": 0.9522}


def load_assessment(folder, dataset_name):
    """Load assessment_results.json from a results folder."""
    path = os.path.join(folder + dataset_name, 'assessment_results.json')
    if not os.path.exists(path):
        # Try alternate layout
        path = os.path.join(folder, 'assessment_results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_per_class(folder, dataset_name, num_cls=6):
    """Load individual class results.json files."""
    results = {}
    base = folder + dataset_name
    for i in range(num_cls):
        p = os.path.join(base, f'{i}_CLS', 'results.json')
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            results[i] = {
                'AUROC': float(np.mean(d.get('TS_AUC', [0]))),
                'AP':    float(np.mean(d.get('TS_AP',  [0]))),
                'F1':    float(np.mean(d.get('TS_F1',  [0]))),
            }
    return results


def fmt(v, ref=None):
    """Format a metric value with optional delta colouring."""
    s = f"{v:.4f}"
    if ref is not None:
        d = v - ref
        sign = '+' if d >= 0 else ''
        s += f"  ({sign}{d:.4f})"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='RESULTS/RESULTS_BASELINE_',
                        help='Baseline results folder prefix (without dataset name)')
    parser.add_argument('--adaptive', default='RESULTS/RESULTS_ADAPTIVE_',
                        help='Adaptive results folder prefix')
    parser.add_argument('--dataset',  default='natops')
    parser.add_argument('--num-cls',  type=int, default=6)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  NeuTraL-AD: Baseline vs Adaptive — NATOPS Results Comparison")
    print(f"{'='*70}\n")

    # ── Load adaptive results ────────────────────────────────────────────────
    adaptive_per_class = load_per_class(args.adaptive, args.dataset, args.num_cls)
    adaptive_assessment = load_assessment(args.adaptive, args.dataset)

    if not adaptive_per_class:
        print(f"[!] Could not find adaptive results at: {args.adaptive}{args.dataset}/")
        print("    Run the adaptive experiment first, then re-run this script.\n")
        print("    Command:")
        print("      python Launch_Exps.py --config-file config_files/config_natops_adaptive_gpu.yml --dataset-name natops")
        return

    # ── Try to load baseline from file; fall back to hardcoded values ────────
    baseline_per_class = load_per_class(args.baseline, args.dataset, args.num_cls)
    if not baseline_per_class:
        print("[i] Using hardcoded baseline results (Colab-reproduced, 6 classes, 100 epochs)\n")
        baseline_per_class = {
            i: v for i, v in enumerate(BASELINE_CLASS_RESULTS.values())
        }

    # ── Per-class table ──────────────────────────────────────────────────────
    headers = f"{'Class':<10} {'Metric':<8} {'Baseline':>10} {'Adaptive':>10} {'Δ (Adaptive−Baseline)':>24}"
    print(headers)
    print('-' * 65)

    comparison = {}
    adap_aurocs, adap_aps, adap_f1s = [], [], []

    for cls_idx in range(args.num_cls):
        cls_name = f"Class {cls_idx+1}"
        b = baseline_per_class.get(cls_idx, {})
        a = adaptive_per_class.get(cls_idx, {})

        if not a:
            print(f"{cls_name:<10} — adaptive results missing for this class")
            continue

        comparison[cls_name] = {}
        for metric in ['AUROC', 'AP', 'F1']:
            bv = b.get(metric, 0.0)
            av = a.get(metric, 0.0)
            d  = av - bv
            comparison[cls_name][metric] = {'baseline': bv, 'adaptive': av, 'delta': d}
            sign = '+' if d >= 0 else ''
            print(f"{cls_name:<10} {metric:<8} {bv:>10.4f} {av:>10.4f} {sign}{d:>+22.4f}")

        adap_aurocs.append(a.get('AUROC', 0))
        adap_aps.append(a.get('AP', 0))
        adap_f1s.append(a.get('F1', 0))
        print()

    # ── Mean row ─────────────────────────────────────────────────────────────
    print('-' * 65)
    mean_a_auroc = float(np.mean(adap_aurocs))
    mean_a_ap    = float(np.mean(adap_aps))
    mean_a_f1    = float(np.mean(adap_f1s))

    for metric, b_mean, a_mean in [
        ('AUROC', BASELINE_MEAN['AUROC'], mean_a_auroc),
        ('AP',    BASELINE_MEAN['AP'],    mean_a_ap),
        ('F1',    BASELINE_MEAN['F1'],    mean_a_f1),
    ]:
        d = a_mean - b_mean
        sign = '+' if d >= 0 else ''
        print(f"{'MEAN':<10} {metric:<8} {b_mean:>10.4f} {a_mean:>10.4f} {sign}{d:>+22.4f}")

    comparison['MEAN'] = {
        'AUROC': {'baseline': BASELINE_MEAN['AUROC'], 'adaptive': mean_a_auroc, 'delta': mean_a_auroc - BASELINE_MEAN['AUROC']},
        'AP':    {'baseline': BASELINE_MEAN['AP'],    'adaptive': mean_a_ap,    'delta': mean_a_ap    - BASELINE_MEAN['AP']},
        'F1':    {'baseline': BASELINE_MEAN['F1'],    'adaptive': mean_a_f1,    'delta': mean_a_f1    - BASELINE_MEAN['F1']},
    }

    print(f"\n{'='*70}\n")

    # ── Adaptive weight summary (if saved) ───────────────────────────────────
    weight_path = os.path.join(args.adaptive + args.dataset, 'weight_history.json')
    if os.path.exists(weight_path):
        with open(weight_path) as f:
            wh = json.load(f)
        print("Transformation Weight Summary (final epoch mean across all classes):")
        print(f"{'Trans':<12} {'Final Weight':>14}")
        print('-' * 28)
        # wh is expected to be {cls: [[w1..wK per epoch]]}
        if isinstance(wh, dict):
            all_final = []
            for cls_key, hist in wh.items():
                if hist:
                    all_final.append(hist[-1])  # last epoch
            if all_final:
                mean_final = np.mean(all_final, axis=0)
                for i, w in enumerate(mean_final):
                    print(f"T{i+1:<11} {w:>14.4f}")
        print()

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs('RESULTS', exist_ok=True)
    json_out = 'RESULTS/comparison_baseline_vs_adaptive.json'
    with open(json_out, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to: {json_out}")

    txt_out = 'RESULTS/comparison_baseline_vs_adaptive.txt'
    with open(txt_out, 'w') as f:
        f.write("NeuTraL-AD: Baseline vs Adaptive — NATOPS Results\n")
        f.write("="*70 + "\n\n")
        for cls_name, metrics in comparison.items():
            f.write(f"{cls_name}:\n")
            for metric, vals in metrics.items():
                d = vals['delta']
                sign = '+' if d >= 0 else ''
                f.write(f"  {metric:<6} Baseline={vals['baseline']:.4f}  Adaptive={vals['adaptive']:.4f}  Δ={sign}{d:.4f}\n")
            f.write("\n")
    print(f"Comparison text saved to: {txt_out}\n")


if __name__ == '__main__':
    main()
