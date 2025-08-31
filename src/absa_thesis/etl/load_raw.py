"""Load raw files (samples + anything in data/data_raw) into canonical schema with strict dtypes."""
import pandas as pd
from pathlib import Path
from loguru import logger
from ..config import get_settings
from ..utils import ensure_dir

CANON = ["platform", "source_id", "rating", "date", "text"]

def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all required cols
    for c in CANON:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    d = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df["date"] = d.dt.date.astype("string")
    for c in ["platform", "source_id", "text"]:
        df[c] = df[c].astype("string").fillna("")
    return df[CANON]

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unsupported file: {path}")

def main():
    cfg = get_settings()
    raw_dir = ensure_dir(cfg.paths["raw_dir"])

    parts = []

    # 1) samples
    for sample in [cfg.data["sample_files"]["amazon_jsonl"], cfg.data["sample_files"]["gplay_csv"]]:
        p = Path(sample)
        if p.exists():
            try:
                df = _read_any(p)
                # attempt column rename if needed
                df = df.rename(columns={
                    "platform": "platform", "source_id": "source_id",
                    "rating": "rating", "date": "date", "text": "text"
                })
                df = _coerce_schema(df)
                parts.append(df)
                logger.info(f"Loaded sample: {p.name} rows={len(df)}")
            except Exception as e:
                logger.warning(f"Sample {p} skipped: {e}")

    # 2) everything in data/data_raw/
    for p in Path(cfg.paths["raw_dir"]).glob("*"):
        if p.suffix.lower() not in {".csv", ".jsonl", ".json"}:
            continue
        if p.name in {"reviews_raw.parquet"}:
            continue
        try:
            df = _read_any(p)
            # try to map common alt headers once
            df = df.rename(columns={
                "asin": "source_id", "item_id": "source_id",
                "stars": "rating",
                "timestamp": "date",
                "review_date": "date",
                "review_text": "text", "content": "text", "review": "text", "body": "text"
            })
            # default platform if missing
            if "platform" not in df.columns:
                df["platform"] = "amazon"
            df = _coerce_schema(df)
            parts.append(df)
            logger.info(f"Loaded raw: {p.name} rows={len(df)}")
        except Exception as e:
            logger.warning(f"Raw {p} skipped: {e}")

    if not parts:
        logger.warning("No source files found.")
        return

    df_all = pd.concat(parts, ignore_index=True)
    out = raw_dir / "reviews_raw.parquet"
    df_all.to_parquet(out, index=False)
    logger.info(f"Wrote {out} with {len(df_all)} rows (schema enforced).")

if __name__ == "__main__":
    main()
