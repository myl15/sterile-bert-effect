"""
Step 4: Tokenize the bilingual corpus and chunk into fixed-length sequences
for efficient MLM training. Saves as HuggingFace Dataset on disk.
"""
import argparse
import sys
from itertools import chain
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerFast


def load_text_lines(file_paths: list) -> Dataset:
    """Load multiple text files into a single HuggingFace Dataset."""
    all_lines = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_lines.append({"text": line})
    return Dataset.from_list(all_lines)


def tokenize_and_chunk(dataset: Dataset, tokenizer: PreTrainedTokenizerFast,
                       max_seq_length: int = 512) -> Dataset:
    """
    Tokenize text and concatenate + chunk into fixed-length sequences.
    Standard approach for MLM pre-training data preparation.
    """
    # Step 1: Tokenize (without truncation — we chunk manually)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=False,
                         return_attention_mask=False)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="  Tokenizing",
        num_proc=1,  # Windows compatibility
    )

    # Step 2: Concatenate all tokens and chunk into max_seq_length blocks
    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # Drop the remainder that doesn't fill a full block
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i:i + max_seq_length]
                for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }
        return result

    chunked = tokenized.map(
        group_texts,
        batched=True,
        desc="  Chunking",
        num_proc=1,  # Windows compatibility
    )

    return chunked


def main():
    parser = argparse.ArgumentParser(description="Prepare MLM datasets")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.config import load_config

    cfg = load_config(args.config)
    proc_dir = Path(cfg["data_dir"]) / "processed"
    tok_dir = Path(cfg["data_dir"]) / "tokenizers"
    mlm_dir = Path(cfg["data_dir"]) / "mlm_datasets"
    max_seq = cfg["pretrain"]["max_seq_length"]

    for variant in ["control", "sterile"]:
        print(f"\n=== Preparing MLM data for {variant.upper()} ===")

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(tok_dir / variant)
        )

        file_paths = [
            proc_dir / f"wiki_en_{variant}.txt",
            proc_dir / f"wiki_fr_{variant}.txt",
        ]
        dataset = load_text_lines(file_paths)
        print(f"  Loaded {len(dataset)} text lines")

        chunked = tokenize_and_chunk(dataset, tokenizer, max_seq)
        print(f"  Created {len(chunked)} chunks of length {max_seq}")

        save_path = mlm_dir / variant
        save_path.mkdir(parents=True, exist_ok=True)
        chunked.save_to_disk(str(save_path))
        print(f"  Saved to {save_path}")

    print("\nDone! MLM datasets in:", mlm_dir)


if __name__ == "__main__":
    main()
