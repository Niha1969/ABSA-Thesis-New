"""Prepare Label Studio tasks from cleaned reviews by splitting into sentences and sampling (no NLTK needed)."""
import re
import random
import pandas as pd
from pathlib import Path
from loguru import logger
from ..config import get_settings
from ..utils import ensure_dir

# Simple regex-based sentence splitter:
# splits on ., !, ? followed by whitespace and an uppercase or digit start.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # Basic cleanup to avoid weird splits
    txt = " ".join(text.split())
    parts = _SENT_SPLIT.split(txt)
    # Filter overly short/long noise
    return [s.strip() for s in parts if 20 <= len(s.strip()) <= 300]

def main(n: int = None):
    cfg = get_settings()
    clean_pq = ensure_dir(cfg.paths["clean_dir"]) / "reviews_clean.parquet"
    if not clean_pq.exists():
        logger.warning("Cleaned dataset missing. Run ETL first (make etl).")
        return

    df = pd.read_parquet(clean_pq)
    sents = []
    for _, row in df.iterrows():
        for s in split_sentences(row["text"]):
            sents.append({"text": s})

    if not sents:
        logger.error("No sentences extracted. Check your cleaned data.")
        return

    random.seed(13)
    random.shuffle(sents)
    n = n or cfg.annotation["sentence_sample_n"]
    sents = sents[:n]

    out_path = Path(cfg.annotation["labelstudio_tasks"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in sents:
            f.write(pd.Series(item).to_json(date_format="iso") + "\n")

    logger.info(f"Wrote {len(sents)} Label Studio tasks to {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="Number of sentences to sample")
    args = ap.parse_args()
    main(n=args.n)
