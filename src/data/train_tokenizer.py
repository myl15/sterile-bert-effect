"""
Step 3: Train two separate WordPiece tokenizers:
  1. Control tokenizer (on standard en+fr text)
  2. Sterile tokenizer (on sterilized en+fr text)
"""
import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


def text_iterator(file_paths: list, batch_size: int = 1000):
    """Yield batches of text lines from multiple files."""
    batch = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
    if batch:
        yield batch


def train_wordpiece_tokenizer(
    file_paths: list,
    save_dir: Path,
    vocab_size: int = 30000,
    min_frequency: int = 3,
    special_tokens: list = None,
) -> PreTrainedTokenizerFast:
    """Train a WordPiece tokenizer and save as HuggingFace-compatible."""
    if special_tokens is None:
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    save_dir.mkdir(parents=True, exist_ok=True)

    # Build tokenizer from components
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train from iterator (memory-efficient)
    tokenizer.train_from_iterator(text_iterator(file_paths), trainer=trainer)

    # Add BERT-style post-processing (CLS/SEP)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # Save raw tokenizer JSON
    tokenizer.save(str(save_dir / "tokenizer.json"))

    # Wrap as HuggingFace PreTrainedTokenizerFast for compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    hf_tokenizer.save_pretrained(str(save_dir))

    print(f"  Tokenizer saved to {save_dir} "
          f"(vocab_size={tokenizer.get_vocab_size()})")
    return hf_tokenizer


def verify_sterile_tokenizer(tokenizer_dir: Path) -> list:
    """Verify that the sterile tokenizer contains no digit subwords."""
    hf_tok = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    vocab = hf_tok.get_vocab()
    digit_tokens = [tok for tok in vocab if any(c.isdigit() for c in tok)]
    if digit_tokens:
        print(f"  WARNING: Sterile tokenizer contains {len(digit_tokens)} "
              f"digit-bearing tokens: {digit_tokens[:20]}...")
    else:
        print("  PASS: Sterile tokenizer contains no digit-bearing tokens.")
    return digit_tokens


def main():
    parser = argparse.ArgumentParser(description="Train WordPiece tokenizers")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.config import load_config

    cfg = load_config(args.config)
    proc_dir = Path(cfg["data_dir"]) / "processed"
    tok_dir = Path(cfg["data_dir"]) / "tokenizers"
    t_cfg = cfg["tokenizer"]

    # Control tokenizer: trained on standard text
    print("=== Training CONTROL tokenizer ===")
    train_wordpiece_tokenizer(
        file_paths=[proc_dir / "wiki_en_control.txt",
                    proc_dir / "wiki_fr_control.txt"],
        save_dir=tok_dir / "control",
        vocab_size=t_cfg["vocab_size"],
        min_frequency=t_cfg["min_frequency"],
        special_tokens=t_cfg["special_tokens"],
    )

    # Sterile tokenizer: trained on sterilized text
    print("\n=== Training STERILE tokenizer ===")
    train_wordpiece_tokenizer(
        file_paths=[proc_dir / "wiki_en_sterile.txt",
                    proc_dir / "wiki_fr_sterile.txt"],
        save_dir=tok_dir / "sterile",
        vocab_size=t_cfg["vocab_size"],
        min_frequency=t_cfg["min_frequency"],
        special_tokens=t_cfg["special_tokens"],
    )

    # Verification
    print("\n=== Verifying sterile tokenizer ===")
    verify_sterile_tokenizer(tok_dir / "sterile")

    print("\nDone! Tokenizers saved to:", tok_dir)


if __name__ == "__main__":
    main()
