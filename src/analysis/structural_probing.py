"""
Step 8: Structural probing — compare cross-lingual alignment of hidden
representations using CKA (Centered Kernel Alignment) similarity across layers.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (
    BertForTokenClassification,
    PreTrainedTokenizerFast,
)


def load_ud_sentences(conllu_path: str) -> list:
    """Parse sentence texts from a CoNLL-U file using '# text = ' comments."""
    sentences = []
    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# text = "):
                sentences.append(line[len("# text = "):].strip())
    return sentences


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA similarity between two representation matrices.
    X: (n_samples, d1), Y: (n_samples, d2)
    Returns: scalar in [0, 1]
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def extract_layer_representations(model, tokenizer, sentences: list,
                                  device: str = "cuda") -> list:
    """
    Extract mean-pooled hidden representations from each layer.
    Returns: list of arrays, one per layer, shape (n_sentences, hidden_size).
    """
    model.eval()
    model.to(device)

    num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings
    all_layer_reps = [[] for _ in range(num_layers)]

    with torch.no_grad():
        for sent in sentences:
            inputs = tokenizer(
                sent, return_tensors="pt", truncation=True,
                max_length=128, padding=True,
            ).to(device)

            outputs = model.bert(
                **inputs, output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states

            for layer_idx, hs in enumerate(hidden_states):
                # Mean-pool over tokens (excluding [PAD])
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)
                all_layer_reps[layer_idx].append(pooled.cpu().numpy())

    return [np.concatenate(reps, axis=0) for reps in all_layer_reps]


def run_cka_analysis(variant: str, config: dict) -> list:
    """Compute CKA between English and French representations at each layer."""
    f_cfg = config["finetune"]

    # Try to find the best model from POS fine-tuning, fall back to pre-trained
    pos_dir = Path(config["output_dir"]) / "checkpoints" / f"{variant}_pos"
    pretrain_dir = Path(config["output_dir"]) / "checkpoints" / variant / "final"

    # Find the model directory
    if pos_dir.exists():
        # Use the best checkpoint from fine-tuning
        checkpoints = sorted(pos_dir.glob("checkpoint-*"))
        if checkpoints:
            model_dir = checkpoints[-1]
        else:
            model_dir = pretrain_dir
    else:
        model_dir = pretrain_dir

    print(f"  Loading model from: {model_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    model = BertForTokenClassification.from_pretrained(str(model_dir))

    # Load UD test sentences from local CoNLL-U files
    project_root = Path(__file__).resolve().parent.parent.parent
    ud_dir = project_root / config.get("ud_dir", "data/ud")
    en_conllu = ud_dir / f"{f_cfg['en_treebank']}-ud-test.conllu"
    fr_conllu = ud_dir / f"{f_cfg['fr_treebank']}-ud-test.conllu"

    en_sents = load_ud_sentences(str(en_conllu))
    fr_sents = load_ud_sentences(str(fr_conllu))

    # Take first N sentences from each
    n = min(500, len(en_sents), len(fr_sents))
    en_sents = en_sents[:n]
    fr_sents = fr_sents[:n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Extracting representations on {device} ({n} sentences each)...")

    en_reps = extract_layer_representations(model, tokenizer, en_sents, device)
    fr_reps = extract_layer_representations(model, tokenizer, fr_sents, device)

    cka_scores = []
    for layer_idx in range(len(en_reps)):
        score = linear_CKA(en_reps[layer_idx], fr_reps[layer_idx])
        cka_scores.append(score)
        print(f"    Layer {layer_idx}: CKA = {score:.4f}")

    return cka_scores


def main():
    parser = argparse.ArgumentParser(description="CKA structural probing")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.config import load_config

    config = load_config(args.config)
    figures_dir = Path(config["output_dir"]) / "figures"
    results_dir = Path(config["output_dir"]) / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_cka = {}
    for variant in ["control", "sterile"]:
        print(f"\n=== CKA Analysis for {variant.upper()} ===")
        all_cka[variant] = run_cka_analysis(variant, config)

    # Save CKA results
    with open(results_dir / "cka_results.json", "w", encoding="utf-8") as f:
        json.dump(all_cka, f, indent=2)

    # Plot CKA comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = list(range(len(all_cka["control"])))
    ax.plot(layers, all_cka["control"], "o-", label="Control",
            color="#4C72B0", linewidth=2, markersize=8)
    ax.plot(layers, all_cka["sterile"], "s--", label="Sterile",
            color="#DD8452", linewidth=2, markersize=8)
    ax.set_xlabel("Layer (0 = embeddings)", fontsize=12)
    ax.set_ylabel("CKA Similarity (EN vs FR)", fontsize=12)
    ax.set_title("Cross-Lingual Alignment by Layer", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "cka_analysis.png", dpi=150)
    plt.savefig(figures_dir / "cka_analysis.pdf")
    print(f"\nCKA plot saved to {figures_dir / 'cka_analysis.png'}")


if __name__ == "__main__":
    main()
