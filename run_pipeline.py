"""
Run the full Sterile Language Effect pipeline.
Each step can also be run independently via its own script.

Usage:
    python run_pipeline.py                           # Run all steps
    python run_pipeline.py --start-step 5            # Resume from step 5
    python run_pipeline.py --start-step 5 --end-step 6  # Run steps 5-6 only
"""
import argparse
import subprocess
import sys

STEPS = [
    ("1. Download Wikipedia",       "src/data/download_wiki.py"),
    ("2. Sterilize text",           "src/data/sterilize.py"),
    ("3. Train tokenizers",         "src/data/train_tokenizer.py"),
    ("4. Prepare MLM datasets",     "src/data/prepare_mlm_data.py"),
    ("5. MLM pre-training",         "src/models/pretrain_mlm.py"),
    ("6. POS fine-tune + eval",     "src/models/finetune_pos.py"),
    ("7. Compare results",          "src/analysis/compare_results.py"),
    ("8. Structural probing",       "src/analysis/structural_probing.py"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the Sterile Language Effect pipeline"
    )
    parser.add_argument("--config", default="configs/base_config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--start-step", type=int, default=1,
                        help="Step number to start from (1-8)")
    parser.add_argument("--end-step", type=int, default=8,
                        help="Step number to end at (1-8)")
    args = parser.parse_args()

    for i, (name, script) in enumerate(STEPS, 1):
        if i < args.start_step or i > args.end_step:
            continue

        print(f"\n{'=' * 60}")
        print(f"  STEP {name}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(
            [sys.executable, script, "--config", args.config],
            check=False,
        )
        if result.returncode != 0:
            print(f"\nERROR: Step {name} failed (exit code {result.returncode})")
            print(f"Fix the issue and restart with: --start-step {i}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print("\nResults in: outputs/results/")
    print("Figures in: outputs/figures/")
    print("TensorBoard: tensorboard --logdir outputs/logs")


if __name__ == "__main__":
    main()
