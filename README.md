# The Sterile Language Effect

Testing whether M-BERT's cross-lingual transfer depends on shared "anchor tokens" (digits, URLs, punctuation) or on deeper structural similarities between languages.

## Hypothesis

If multilingual BERT's shared representation space is primarily anchored by cross-lingual tokens (numbers, URLs, shared punctuation), then a model pre-trained on text **stripped of these tokens** should show significantly lower zero-shot cross-lingual transfer compared to a model trained on standard text.

## Method

Two small BERT models (4 layers, 256 hidden, ~11M params) are pre-trained with MLM on bilingual English+French Wikipedia:

- **Control**: Standard, unprocessed text
- **Sterile**: Text with all digits, URLs, and shared punctuation removed

Both models are fine-tuned on English POS tagging (Universal Dependencies) and evaluated zero-shot on French POS tagging. CKA structural probing measures cross-lingual alignment at each layer.

## Project Structure

```
├── configs/base_config.yaml       # All hyperparameters and paths
├── pyproject.toml                 # Dependencies (managed by uv)
├── run_pipeline.py                # Run full pipeline locally (steps 1-8)
├── scripts/
│   └── 00_download_cache.sh      # Login-node: download Wikipedia + UD treebanks
├── slurm/                         # SLURM scripts for cluster execution
│   ├── setup_env.sh               # One-time environment setup (creates .venv)
│   ├── 01_data_prep.sh            # Steps 2-4 (CPU job)
│   ├── 02_pretrain.sh             # Step 5 (GPU job)
│   ├── 03_finetune_eval.sh        # Steps 6-8 (GPU job)
│   └── run_all.sh                 # Submit all jobs with dependency chaining
├── src/
│   ├── data/
│   │   ├── download_wiki.py       # Step 1: Download Wikipedia subsets
│   │   ├── sterilize.py           # Step 2: Remove anchor tokens
│   │   ├── train_tokenizer.py     # Step 3: Train WordPiece tokenizers
│   │   └── prepare_mlm_data.py    # Step 4: Tokenize + chunk for MLM
│   ├── models/
│   │   ├── pretrain_mlm.py        # Step 5: MLM pre-training
│   │   └── finetune_pos.py        # Step 6: POS fine-tune + zero-shot eval
│   ├── analysis/
│   │   ├── compare_results.py     # Step 7: Comparison tables + bar charts
│   │   └── structural_probing.py  # Step 8: CKA representation analysis
│   └── utils/
│       ├── config.py              # YAML config loader
│       └── metrics.py             # Accuracy/F1 computation
└── tests/                         # Unit tests
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install it if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the virtual environment and install all dependencies (including PyTorch with CUDA 12.4):

```bash
uv sync
```

Or run the one-time setup script which handles this automatically:

```bash
bash slurm/setup_env.sh
```

To activate the environment manually:

```bash
source .venv/bin/activate
```

## Running

### Local

```bash
# Full pipeline
python run_pipeline.py

# Resume from a specific step after failure
python run_pipeline.py --start-step 5

# Run individual steps
python src/data/download_wiki.py --config configs/base_config.yaml
python src/models/pretrain_mlm.py --config configs/base_config.yaml --variant control
```

### SLURM (supercomputer)

Compute nodes have no internet access, so all data must be downloaded on the login node first. The full workflow is three steps:

**1. One-time environment setup** (login node, run once):

```bash
bash slurm/setup_env.sh
```

**2. Download all data** (login node, requires internet):

```bash
bash scripts/00_download_cache.sh
```

This downloads English and French Wikipedia subsets and fetches the Universal Dependencies treebanks (EN-EWT and FR-GSD), saving everything locally under `data/`.

**3. Submit the compute jobs** (from the project root):

```bash
bash slurm/run_all.sh
```

This submits three jobs with dependency chaining — each waits for the previous one to complete successfully before starting.

Before submitting, edit `MODULE_CUDA` and `--qos` / `--partition` in the SLURM scripts to match your cluster.

### Monitoring

```bash
# Training logs
tensorboard --logdir outputs/logs

# SLURM job output
tail -f slurm/logs/02_pretrain_<jobid>.out
```

## Outputs

| Path | Contents |
|------|----------|
| `outputs/results/*.json` | POS tagging accuracy/F1 for both models |
| `outputs/results/cka_results.json` | Layer-wise CKA cross-lingual alignment scores |
| `outputs/figures/pos_comparison.png` | Bar chart comparing EN/FR POS performance |
| `outputs/figures/cka_analysis.png` | CKA similarity by layer (control vs sterile) |

## Interpreting Results

- **Large performance gap** (sterile << control on French zero-shot): Supports the anchor token hypothesis — shared tokens are critical for cross-lingual alignment.
- **Small or no gap**: Suggests structural similarities (e.g., shared SVO word order between English and French) drive cross-lingual transfer more than surface-level token overlap.
- **CKA plots**: Higher CKA = more aligned EN/FR representations. If the control model shows consistently higher CKA across layers, anchor tokens contribute to representational alignment.

## Tests

```bash
pytest tests/ -v
```
