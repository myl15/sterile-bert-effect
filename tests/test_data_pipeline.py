import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestDataPipeline:
    def test_mlm_dataset_shape(self):
        """MLM datasets should have input_ids of correct length (512)."""
        from datasets import load_from_disk

        for variant in ["control", "sterile"]:
            ds_dir = DATA_DIR / "mlm_datasets" / variant
            if not ds_dir.exists():
                pytest.skip(f"{variant} MLM data not yet prepared")

            ds = load_from_disk(str(ds_dir))
            assert "input_ids" in ds.column_names
            assert len(ds[0]["input_ids"]) == 512

    def test_mlm_dataset_not_empty(self):
        """MLM datasets should have a reasonable number of sequences."""
        from datasets import load_from_disk

        for variant in ["control", "sterile"]:
            ds_dir = DATA_DIR / "mlm_datasets" / variant
            if not ds_dir.exists():
                pytest.skip(f"{variant} MLM data not yet prepared")

            ds = load_from_disk(str(ds_dir))
            assert len(ds) > 100, f"Dataset too small: {len(ds)} sequences"

    def test_raw_data_exists(self):
        """Raw Wikipedia files should exist after download."""
        for lang in ["en", "fr"]:
            path = DATA_DIR / "raw" / f"wiki_{lang}.txt"
            if not path.exists():
                pytest.skip("Raw data not yet downloaded")
            assert path.stat().st_size > 0, f"{path} is empty"

    def test_processed_data_exists(self):
        """Processed (sterilized + control) files should exist."""
        for lang in ["en", "fr"]:
            for variant in ["control", "sterile"]:
                path = DATA_DIR / "processed" / f"wiki_{lang}_{variant}.txt"
                if not path.exists():
                    pytest.skip("Processed data not yet created")
                assert path.stat().st_size > 0, f"{path} is empty"

    def test_sterile_data_has_no_digits(self):
        """Sterile data files should contain no digit characters."""
        for lang in ["en", "fr"]:
            path = DATA_DIR / "processed" / f"wiki_{lang}_sterile.txt"
            if not path.exists():
                pytest.skip("Sterile data not yet created")

            with open(path, "r", encoding="utf-8") as f:
                # Check first 100 lines
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    assert not any(c.isdigit() for c in line), (
                        f"Digit found in sterile data line {i}: "
                        f"{line[:80]}..."
                    )
