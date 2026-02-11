"""
Step 1: Download a subset of English and French Wikipedia.
Saves raw text files with one article per line.
"""
import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_wiki(lang_config: str, output_path: Path, max_articles: int,
                  min_length: int):
    """Stream Wikipedia and save a subset to disk as a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {lang_config} (max {max_articles} articles)...")

    # Try streaming first to avoid downloading the full dataset
    try:
        dataset = load_dataset(
            "wikipedia", lang_config,
            split="train", streaming=True, trust_remote_code=True,
        )
    except Exception as e:
        print(f"Streaming with 'wikipedia' failed ({e}), trying wikimedia/wikipedia...")
        # Fallback to newer dataset
        fallback_config = lang_config.replace("20220301", "20231101")
        dataset = load_dataset(
            "wikimedia/wikipedia", fallback_config,
            split="train", streaming=True, trust_remote_code=True,
        )

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for article in tqdm(dataset, desc=f"  {lang_config}", total=max_articles):
            text = article["text"].strip()
            if len(text) < min_length:
                continue
            # Replace internal newlines to keep one-article-per-line
            clean_text = " ".join(text.split())
            f.write(clean_text + "\n")
            count += 1
            if count >= max_articles:
                break

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {count} articles to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia subsets")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    # Add project root to path for imports
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.config import load_config

    cfg = load_config(args.config)
    data_dir = Path(cfg["data_dir"]) / "raw"

    download_wiki(
        cfg["wiki"]["en_config"],
        data_dir / "wiki_en.txt",
        cfg["wiki"]["max_articles_en"],
        cfg["wiki"]["min_article_length"],
    )
    download_wiki(
        cfg["wiki"]["fr_config"],
        data_dir / "wiki_fr.txt",
        cfg["wiki"]["max_articles_fr"],
        cfg["wiki"]["min_article_length"],
    )

    print("\nDone! Raw data saved to:", data_dir)


if __name__ == "__main__":
    main()
