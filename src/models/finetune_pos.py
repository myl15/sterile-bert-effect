"""
Step 6: Fine-tune pre-trained models on English POS tagging (Universal Dependencies).
Then zero-shot evaluate on French POS tagging.
"""
import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    BertConfig,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


def get_upos_labels(dataset):
    """Extract UPOS label names from the dataset features."""
    return dataset.features["upos"].feature.names


def load_ud_dataset(treebank: str, split: str):
    """Load a Universal Dependencies treebank split from the local disk cache."""
    project_root = Path(__file__).resolve().parent.parent.parent
    ud_path = project_root / "data" / "ud_datasets" / treebank
    return load_from_disk(str(ud_path))[split]


def tokenize_and_align_labels(examples, tokenizer, max_seq_length=256):
    """
    Tokenize UD examples and align POS labels with WordPiece subwords.
    First subword of each word gets the label, rest get -100 (ignored by loss).
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
        padding=False,  # DataCollator will handle padding
    )

    all_labels = []
    for i, upos_ids in enumerate(examples["upos"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a new word
                label_ids.append(upos_ids[word_idx])
            else:
                # Continuation subword
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def finetune_and_evaluate(variant: str, config: dict):
    """Fine-tune on English POS, zero-shot eval on French."""
    f_cfg = config["finetune"]
    model_dir = Path(config["output_dir"]) / "checkpoints" / variant / "final"
    results_dir = Path(config["output_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))

    # Load English UD data to get label names
    en_train = load_ud_dataset(f_cfg["en_treebank"], "train")
    en_val = load_ud_dataset(f_cfg["en_treebank"], "validation")
    en_test = load_ud_dataset(f_cfg["en_treebank"], "test")

    # Get label info from dataset
    upos_labels = get_upos_labels(en_train)
    num_labels = len(upos_labels)
    label2id = {l: i for i, l in enumerate(upos_labels)}
    id2label = {i: l for i, l in enumerate(upos_labels)}
    print(f"  UPOS labels ({num_labels}): {upos_labels}")

    # Load French UD data (for zero-shot evaluation)
    fr_test = load_ud_dataset(f_cfg["fr_treebank"], "test")

    # Create token classification model from pre-trained checkpoint
    pretrained_config = BertConfig.from_pretrained(str(model_dir))
    pretrained_config.num_labels = num_labels
    pretrained_config.id2label = id2label
    pretrained_config.label2id = label2id

    model = BertForTokenClassification.from_pretrained(
        str(model_dir),
        config=pretrained_config,
        ignore_mismatched_sizes=True,  # Classification head is new
    )

    # Tokenize and align labels
    def tok_align(examples):
        return tokenize_and_align_labels(
            examples, tokenizer, f_cfg["max_seq_length"]
        )

    en_train_tok = en_train.map(tok_align, batched=True,
                                remove_columns=en_train.column_names)
    en_val_tok = en_val.map(tok_align, batched=True,
                            remove_columns=en_val.column_names)
    en_test_tok = en_test.map(tok_align, batched=True,
                              remove_columns=en_test.column_names)
    fr_test_tok = fr_test.map(tok_align, batched=True,
                              remove_columns=fr_test.column_names)

    # Data collator for dynamic padding
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True,
    )

    # Import metrics
    from src.utils.metrics import compute_pos_metrics

    ft_output_dir = Path(config["output_dir"]) / "checkpoints" / f"{variant}_pos"
    training_args = TrainingArguments(
        output_dir=str(ft_output_dir),
        num_train_epochs=f_cfg["epochs"],
        per_device_train_batch_size=f_cfg["batch_size"],
        per_device_eval_batch_size=f_cfg["batch_size"],
        learning_rate=f_cfg["learning_rate"],
        warmup_ratio=f_cfg["warmup_ratio"],
        weight_decay=f_cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        seed=f_cfg["seed"],
        dataloader_num_workers=0,  # Windows compatibility
        bf16=True,
        report_to="tensorboard",
        logging_dir=str(Path(config["output_dir"]) / "logs" / f"{variant}_pos"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=en_train_tok,
        eval_dataset=en_val_tok,
        data_collator=data_collator,
        compute_metrics=compute_pos_metrics,
    )

    # Fine-tune on English
    print(f"\n=== Fine-tuning {variant.upper()} on English POS ===")
    trainer.train()

    # Evaluate on English test
    print(f"\n--- English test evaluation ---")
    en_results = trainer.evaluate(en_test_tok)
    print(f"  {en_results}")

    # Zero-shot evaluate on French test
    print(f"\n--- French zero-shot evaluation ---")
    fr_results = trainer.evaluate(fr_test_tok)
    print(f"  {fr_results}")

    # Save results
    all_results = {
        "variant": variant,
        "en_test": en_results,
        "fr_test_zeroshot": fr_results,
    }
    results_path = results_dir / f"{variant}_pos_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="POS fine-tuning + zero-shot eval")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--variant", choices=["control", "sterile", "both"],
                        default="both")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.config import load_config

    config = load_config(args.config)

    variants = ["control", "sterile"] if args.variant == "both" else [args.variant]
    for variant in variants:
        finetune_and_evaluate(variant, config)

    print("\nFine-tuning and evaluation complete!")


if __name__ == "__main__":
    main()
