"""
Step 5: Pre-train a small BERT model with MLM objective.
Runs for both control and sterile variants.
"""
import argparse
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


def create_model(config: dict, vocab_size: int) -> BertForMaskedLM:
    """Create a TinyBERT-like model from scratch."""
    m_cfg = config["model"]
    bert_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=m_cfg["hidden_size"],
        num_hidden_layers=m_cfg["num_hidden_layers"],
        num_attention_heads=m_cfg["num_attention_heads"],
        intermediate_size=m_cfg["intermediate_size"],
        max_position_embeddings=m_cfg["max_position_embeddings"],
        type_vocab_size=2,
    )
    model = BertForMaskedLM(bert_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Created model with {num_params:,} parameters")
    return model


def pretrain(variant: str, config: dict):
    """Run MLM pre-training for a given variant (control or sterile)."""
    p_cfg = config["pretrain"]
    tok_dir = Path(config["data_dir"]) / "tokenizers" / variant
    data_dir = Path(config["data_dir"]) / "mlm_datasets" / variant
    out_dir = Path(config["output_dir"]) / "checkpoints" / variant

    # Load tokenizer and data
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
    dataset = load_from_disk(str(data_dir))

    # Split into train/eval (95/5)
    split = dataset.train_test_split(test_size=0.05, seed=p_cfg["seed"])
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Create model
    model = create_model(config, tokenizer.vocab_size)

    # Data collator handles random masking at training time
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=p_cfg["mlm_probability"],
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=p_cfg["epochs"],
        per_device_train_batch_size=p_cfg["batch_size"],
        per_device_eval_batch_size=p_cfg["batch_size"],
        gradient_accumulation_steps=p_cfg["gradient_accumulation_steps"],
        learning_rate=p_cfg["learning_rate"],
        warmup_ratio=p_cfg["warmup_ratio"],
        weight_decay=p_cfg["weight_decay"],
        bf16=p_cfg["bf16"],
        logging_dir=str(Path(config["output_dir"]) / "logs" / variant),
        logging_steps=p_cfg["logging_steps"],
        save_steps=p_cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=p_cfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=p_cfg["seed"],
        dataloader_num_workers=0,  # Windows compatibility
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(f"\n=== Starting MLM pre-training for {variant.upper()} ===")
    trainer.train()

    # Save final model + tokenizer together
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"  Model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="MLM pre-training")
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
        pretrain(variant, config)

    print("\nPre-training complete!")


if __name__ == "__main__":
    main()
