"""
Step 7: Load results from both variants, produce comparison tables and plots.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.config import load_config

    config = load_config(args.config)
    results_dir = Path(config["output_dir"]) / "results"
    figures_dir = Path(config["output_dir"]) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_dir / "control_pos_results.json", encoding="utf-8") as f:
        control = json.load(f)
    with open(results_dir / "sterile_pos_results.json", encoding="utf-8") as f:
        sterile = json.load(f)

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON: The Sterile Language Effect")
    print("=" * 70)
    print(f"{'Metric':<25} {'Control':>15} {'Sterile':>15} {'Delta':>15}")
    print("-" * 70)

    metrics_to_compare = [
        ("EN Accuracy", "en_test", "eval_accuracy"),
        ("EN F1 (macro)", "en_test", "eval_f1_macro"),
        ("EN F1 (weighted)", "en_test", "eval_f1_weighted"),
        ("FR Accuracy (0-shot)", "fr_test_zeroshot", "eval_accuracy"),
        ("FR F1 (macro, 0-shot)", "fr_test_zeroshot", "eval_f1_macro"),
        ("FR F1 (weighted, 0-shot)", "fr_test_zeroshot", "eval_f1_weighted"),
    ]

    for label, split_key, metric_key in metrics_to_compare:
        c_val = control[split_key].get(metric_key, 0)
        s_val = sterile[split_key].get(metric_key, 0)
        delta = s_val - c_val
        print(f"{label:<25} {c_val:>14.4f} {s_val:>14.4f} {delta:>+14.4f}")

    # Cross-lingual transfer gap
    print("\n--- Cross-Lingual Transfer Gap ---")
    for variant_name, data in [("Control", control), ("Sterile", sterile)]:
        en_acc = data["en_test"].get("eval_accuracy", 0)
        fr_acc = data["fr_test_zeroshot"].get("eval_accuracy", 0)
        gap = en_acc - fr_acc
        print(f"  {variant_name}: EN={en_acc:.4f}, FR={fr_acc:.4f}, "
              f"Gap={gap:.4f}")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    en_metrics = ["eval_accuracy", "eval_f1_macro"]
    en_labels = ["Accuracy", "F1 (macro)"]
    x = np.arange(len(en_labels))
    w = 0.35

    # English performance
    axes[0].bar(x - w / 2,
                [control["en_test"].get(m, 0) for m in en_metrics],
                w, label="Control", color="#4C72B0")
    axes[0].bar(x + w / 2,
                [sterile["en_test"].get(m, 0) for m in en_metrics],
                w, label="Sterile", color="#DD8452")
    axes[0].set_title("English POS (Fine-tuned)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(en_labels)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_ylabel("Score")

    # French zero-shot performance
    axes[1].bar(x - w / 2,
                [control["fr_test_zeroshot"].get(m, 0) for m in en_metrics],
                w, label="Control", color="#4C72B0")
    axes[1].bar(x + w / 2,
                [sterile["fr_test_zeroshot"].get(m, 0) for m in en_metrics],
                w, label="Sterile", color="#DD8452")
    axes[1].set_title("French POS (Zero-shot)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(en_labels)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].set_ylabel("Score")

    plt.suptitle("The Sterile Language Effect: POS Tagging Performance",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "pos_comparison.png", dpi=150)
    plt.savefig(figures_dir / "pos_comparison.pdf")
    print(f"\nFigure saved to {figures_dir / 'pos_comparison.png'}")


if __name__ == "__main__":
    main()
