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
├── run_pipeline.py                # Run full pipeline locally (steps 1-8)
├── slurm/                         # SLURM scripts for cluster execution
│   ├── setup_env.sh               # One-time environment setup
│   ├── 01_data_prep.sh            # Steps 1-4 (CPU)
│   ├── 02_pretrain.sh             # Step 5 (GPU)
│   ├── 03_finetune_eval.sh        # Steps 6-8 (GPU)
│   └── run_all.sh                 # Submit all jobs with dependencies
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

```bash
# Option A: conda (may require manual PyTorch install if download fails)
conda env create -f environment.yml
conda activate sterile-lang

# Option B: manual install
conda create -n sterile-lang python=3.11 -y
conda activate sterile-lang
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda install numpy scipy scikit-learn matplotlib seaborn pyyaml tqdm tensorboard pytest -y
pip install "transformers>=4.40" "datasets>=3.0" "tokenizers>=0.19" "accelerate>=0.30" evaluate seqeval pandas
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

Edit the `MODULE_CONDA`, `MODULE_CUDA`, and `--partition` values in the SLURM scripts to match your cluster, then:

```bash
bash slurm/setup_env.sh    # one-time, run interactively
bash slurm/run_all.sh       # submits 3 jobs with dependency chaining
```

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
