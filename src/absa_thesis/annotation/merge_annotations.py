"""Merge exported Label Studio JSON and split into train/val/test for ABSA."""
import json, random
import pandas as pd
from pathlib import Path
from loguru import logger
from ..config import get_settings
from ..utils import ensure_dir

def main(export_jsonl: str):
    cfg = get_settings()
    out_dir = ensure_dir("artifacts/absa")

    rows = []
    with open(export_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Flexible extract depending on LS export
            text = rec.get("text") or rec.get("data", {}).get("text")
            aspect = rec.get("aspect")
            polarity = rec.get("polarity")
            # Fallback for 'result' style exports (choices list)
            if (aspect is None or polarity is None) and "result" in rec:
                vals = [r.get("value", {}).get("choices") for r in rec["result"] if r.get("value")]
                vals = [v for v in vals if v]
                if len(vals) >= 2:
                    aspect = aspect or (vals[0][0] if vals[0] else None)
                    polarity = polarity or (vals[1][0] if vals[1] else None)

            if text and aspect and polarity:
                rows.append({"sentence": text, "aspect": aspect, "polarity": polarity})

    if not rows:
        logger.error("No labeled rows found. Check your export format.")
        return

    df = pd.DataFrame(rows).dropna().drop_duplicates()
    logger.info(f"Merged {len(df)} labeled sentences.")

    random.seed(13)
    df = df.sample(frac=1.0, random_state=13).reset_index(drop=True)
    n = len(df)
    n_train, n_val = int(0.8*n), int(0.1*n)
    train, val, test = df.iloc[:n_train], df.iloc[n_train:n_train+n_val], df.iloc[n_train+n_val:]

    train.to_csv(cfg.modeling["absa_train_file"], index=False)
    val.to_csv(cfg.modeling["absa_val_file"], index=False)
    test.to_csv(cfg.modeling["absa_test_file"], index=False)
    logger.info("Saved train/val/test to artifacts/absa/")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("export_jsonl", type=str, help="Path to Label Studio export JSONL")
    args = ap.parse_args()
    main(args.export_jsonl)
