#!/bin/bash
# ============================================================================
# Pre-download all data and cache all HuggingFace datasets.
# Run this interactively on the LOGIN NODE (has internet access) before
# submitting any Slurm jobs.
#
# Usage:  bash scripts/00_download_cache.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project root : $PROJECT_ROOT"
echo ""

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

# --- Wikipedia ---------------------------------------------------------------
echo "=== Step 1: Download Wikipedia (EN + FR) ==="
python src/data/download_wiki.py --config configs/base_config.yaml

# --- Universal Dependencies treebanks ----------------------------------------
# datasets>=3.0 no longer supports Python-script-based Hub loaders, so we
# download the raw CoNLL-U files directly from the UD GitHub repos and convert
# them to HuggingFace Datasets saved on disk.
echo ""
echo "=== Downloading and caching Universal Dependencies treebanks ==="
python - <<'EOF'
import io
import urllib.request
from pathlib import Path

from conllu import parse_incr
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
upos2id = {tag: i for i, tag in enumerate(UPOS_TAGS)}

features = Features({
    "tokens": Sequence(Value("string")),
    "upos":   Sequence(ClassLabel(names=UPOS_TAGS)),
})

UD_RELEASE = "r2.14"
BASE = "https://raw.githubusercontent.com/UniversalDependencies"

TREEBANKS = {
    "en_ewt": {
        "repo": "UD_English-EWT",
        "files": {
            "train":      "en_ewt-ud-train.conllu",
            "validation": "en_ewt-ud-dev.conllu",
            "test":       "en_ewt-ud-test.conllu",
        },
    },
    "fr_gsd": {
        "repo": "UD_French-GSD",
        "files": {
            "train":      "fr_gsd-ud-train.conllu",
            "validation": "fr_gsd-ud-dev.conllu",
            "test":       "fr_gsd-ud-test.conllu",
        },
    },
}

def parse_conllu(content: str) -> dict:
    tokens_list, upos_list = [], []
    for sentence in parse_incr(io.StringIO(content)):
        tokens, upos_ids = [], []
        for token in sentence:
            if not isinstance(token["id"], int):  # skip multi-word / empty nodes
                continue
            tokens.append(token["form"])
            tag = token["upostag"] or "X"
            upos_ids.append(upos2id.get(tag, upos2id["X"]))
        if tokens:
            tokens_list.append(tokens)
            upos_list.append(upos_ids)
    return {"tokens": tokens_list, "upos": upos_list}

output_dir = Path("data/ud_datasets")

for treebank, info in TREEBANKS.items():
    print(f"\nProcessing {treebank}...")
    splits = {}
    for split_name, filename in info["files"].items():
        url = f"{BASE}/{info['repo']}/{UD_RELEASE}/{filename}"
        print(f"  Downloading {split_name} from {url}")
        with urllib.request.urlopen(url) as resp:
            content = resp.read().decode("utf-8")
        data = parse_conllu(content)
        splits[split_name] = Dataset.from_dict(data, features=features)
        print(f"    {len(data['tokens'])} sentences")

    out_path = output_dir / treebank
    DatasetDict(splits).save_to_disk(str(out_path))
    print(f"  Saved to {out_path}")

print("\nAll UD treebanks saved to data/ud_datasets/")
EOF

echo ""
echo "=== All downloads complete ==="
echo "You can now submit Slurm jobs:"
echo "  bash slurm/run_all.sh"
