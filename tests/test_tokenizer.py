import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOK_DIR = Path(__file__).resolve().parent.parent / "data" / "tokenizers"


class TestTokenizer:
    def test_sterile_tokenizer_no_digits(self):
        """Sterile tokenizer vocabulary should contain no digit characters."""
        from transformers import PreTrainedTokenizerFast

        tok_path = TOK_DIR / "sterile"
        if not tok_path.exists():
            pytest.skip("Sterile tokenizer not yet trained")

        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
        vocab = tokenizer.get_vocab()
        digit_tokens = [t for t in vocab if any(c.isdigit() for c in t)]
        assert len(digit_tokens) == 0, (
            f"Found {len(digit_tokens)} digit tokens: {digit_tokens[:10]}"
        )

    def test_special_tokens_present(self):
        """Both tokenizers should have all required special tokens."""
        from transformers import PreTrainedTokenizerFast

        for variant in ["control", "sterile"]:
            tok_path = TOK_DIR / variant
            if not tok_path.exists():
                pytest.skip(f"{variant} tokenizer not yet trained")

            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
            for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
                assert token in tokenizer.get_vocab(), (
                    f"{token} missing from {variant} tokenizer"
                )

    def test_tokenizer_roundtrip(self):
        """Encoding then decoding should roughly preserve the text."""
        from transformers import PreTrainedTokenizerFast

        for variant in ["control", "sterile"]:
            tok_path = TOK_DIR / variant
            if not tok_path.exists():
                pytest.skip(f"{variant} tokenizer not yet trained")

            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
            text = "The cat sat on the mat"
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            assert "cat" in decoded.lower()
