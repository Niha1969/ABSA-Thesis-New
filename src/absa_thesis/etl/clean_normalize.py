"""Clean and normalize raw reviews into de-duplicated, trimmed text with basic heuristics."""
import pandas as pd
from loguru import logger
from ..config import get_settings
from ..utils import ensure_dir

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").strip()
    return s

def main():
    cfg = get_settings()
    raw_pq = ensure_dir(cfg.paths["raw_dir"]) / "reviews_raw.parquet"
    clean_dir = ensure_dir(cfg.paths["clean_dir"])

    if not raw_pq.exists():
        logger.warning("Raw parquet not found. Run `make etl` step 1 first.")
        return

    df = pd.read_parquet(raw_pq)

    # clean text
    df["text"] = df["text"].map(clean_text)

    # enforce min length
    df = df[df["text"].str.len() >= cfg.etl["min_text_len"]].copy()

    # drop duplicates on specified keys
    df = df.drop_duplicates(subset=cfg.etl["dedup_on"]).reset_index(drop=True)

    out = clean_dir / "reviews_clean.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Cleaned dataset saved to {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
