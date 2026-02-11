"""
Step 2: Apply sterilization — remove anchor tokens (digits, URLs, shared punctuation).
Produces parallel 'sterile' versions of each Wikipedia text file.
"""
import argparse
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from tqdm import tqdm

# Compile patterns once for performance.
# Order matters: apply URL removal first (relies on dots/slashes being present).
PATTERNS = {
    "urls": re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    "digits": re.compile(r"\d+"),
    "shared_punct": re.compile(r"[.,!?;:()\"\'\-]"),
}
WHITESPACE = re.compile(r"\s+")


def _remove_unicode_numbers(text: str) -> str:
    """Remove any character whose Unicode category starts with 'N' (Number).
    This catches superscripts, subscripts, circled digits, etc. that \\d misses."""
    return "".join(c for c in text if not unicodedata.category(c).startswith("N"))


def sterilize_line(line: str, *, remove_urls=True, remove_digits=True,
                   remove_shared_punct=True) -> str:
    """Remove anchor tokens from a single line of text."""
    text = line
    # Order: URLs first, then digits, then punctuation
    if remove_urls:
        text = PATTERNS["urls"].sub("", text)
    if remove_digits:
        text = PATTERNS["digits"].sub("", text)
        text = _remove_unicode_numbers(text)
    if remove_shared_punct:
        text = PATTERNS["shared_punct"].sub("", text)
    # Collapse whitespace
    text = WHITESPACE.sub(" ", text).strip()
    return text


def sterilize_file(input_path: Path, output_path: Path, **kwargs):
    """Process an entire file line by line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"  Sterilizing {input_path.name}"):
            total += 1
            sterile = sterilize_line(line, **kwargs)
            if len(sterile) > 50:  # Skip lines that became too short
                fout.write(sterile + "\n")
                kept += 1

    print(f"  Kept {kept}/{total} lines in {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Sterilize Wikipedia text")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.config import load_config

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data_dir"]) / "raw"
    proc_dir = Path(cfg["data_dir"]) / "processed"
    s_cfg = cfg["sterilize"]

    for lang in ["en", "fr"]:
        print(f"\nProcessing {lang}...")
        # Sterilize
        sterilize_file(
            raw_dir / f"wiki_{lang}.txt",
            proc_dir / f"wiki_{lang}_sterile.txt",
            remove_urls=s_cfg["remove_urls"],
            remove_digits=s_cfg["remove_digits"],
            remove_shared_punct=s_cfg["remove_shared_punct"],
        )
        # Copy control (unmodified) data for uniformity
        control_path = proc_dir / f"wiki_{lang}_control.txt"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(raw_dir / f"wiki_{lang}.txt", control_path)
        print(f"  Copied control data to {control_path.name}")

    print("\nDone! Processed data in:", proc_dir)


if __name__ == "__main__":
    main()
