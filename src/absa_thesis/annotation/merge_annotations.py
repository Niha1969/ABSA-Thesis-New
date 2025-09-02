"""Merge Label Studio export into train/val/test for ABSA (CSV/TSV/JSON/JSONL)."""
from __future__ import annotations
import argparse, json, random
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import pandas as pd
from loguru import logger

from ..config import get_settings
from ..utils import ensure_dir

def _from_jsonl(p: Path) -> List[Dict[str, str]]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text, aspect, polarity = _extract_triplet(rec)
            if text and aspect and polarity:
                rows.append({"sentence": text, "aspect": aspect, "polarity": polarity})
    return rows

def _from_json(p: Path) -> List[Dict[str, str]]:
    rows = []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict): data = [data]
    for rec in data:
        text, aspect, polarity = _extract_triplet(rec)
        if text and aspect and polarity:
            rows.append({"sentence": text, "aspect": aspect, "polarity": polarity})
    return rows

def _from_csv(df: pd.DataFrame) -> List[Dict[str, str]]:
    rows = []
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("sentence") or cols.get("text") or cols.get("data.text")
    aspect_col = cols.get("aspect")
    polarity_col = cols.get("polarity")
    # If LS dumped a 'result' JSON column, try that per row
    has_result = "result" in df.columns
    for _, r in df.iterrows():
        text, aspect, polarity = None, None, None
        if has_result and pd.notna(r["result"]):
            try:
                parsed = json.loads(r["result"]) if isinstance(r["result"], str) else r["result"]
                rec = {"result": parsed}
                if "data.text" in df.columns and pd.notna(r["data.text"]):
                    rec["data"] = {"text": r["data.text"]}
                elif "text" in df.columns and pd.notna(r["text"]):
                    rec["text"] = r["text"]
                text, aspect, polarity = _extract_triplet(rec)
            except Exception:
                pass
        if text is None and text_col and pd.notna(r[text_col]): text = str(r[text_col])
        if aspect is None and aspect_col and pd.notna(r[aspect_col]): aspect = str(r[aspect_col])
        if polarity is None and polarity_col and pd.notna(r[polarity_col]): polarity = str(r[polarity_col])
        if text and aspect and polarity:
            rows.append({"sentence": text, "aspect": aspect, "polarity": polarity})
    return rows

def _extract_triplet(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    text = rec.get("text") or rec.get("data", {}).get("text")
    aspect = rec.get("aspect")
    polarity = rec.get("polarity")
    if (aspect is None or polarity is None) and isinstance(rec.get("result"), list):
        choices = []
        for r in rec["result"]:
            val = r.get("value")
            if isinstance(val, dict) and "choices" in val and val["choices"]:
                choices.append(val["choices"][0])
        if len(choices) >= 2:
            if aspect is None: aspect = choices[0]
            if polarity is None: polarity = choices[1]
    return text, aspect, polarity

def main(export_path: str):
    cfg = get_settings()
    out_dir = ensure_dir(Path("artifacts/absa"))
    p = Path(export_path)
    if not p.exists():
        logger.error(f"Export not found: {p}")
        return
    rows: List[Dict[str, str]] = []
    ext = p.suffix.lower()
    try:
        if ext == ".csv":
            rows = _from_csv(pd.read_csv(p))
        elif ext == ".tsv":
            rows = _from_csv(pd.read_csv(p, sep="\t"))
        elif ext == ".jsonl":
            rows = _from_jsonl(p)
        elif ext == ".json":
            rows = _from_json(p)
        else:
            logger.warning(f"Unknown extension '{ext}', trying JSONL parse...")
            rows = _from_jsonl(p)
    except Exception as e:
        logger.error(f"Failed to parse export: {e}")
        return

    if not rows:
        logger.error("No labeled rows found. Check your export columns (need text/sentence, aspect, polarity).")
        return

    df = pd.DataFrame(rows).dropna().drop_duplicates()
    logger.info(f"Merged {len(df)} labeled sentences.")
    df = df.sample(frac=1.0, random_state=13).reset_index(drop=True)
    n = len(df); n_train = int(0.8*n); n_val = int(0.1*n)
    train = df.iloc[:n_train]; val = df.iloc[n_train:n_train+n_val]; test = df.iloc[n_train+n_val:]
    train.to_csv(cfg.modeling["absa_train_file"], index=False)
    val.to_csv(cfg.modeling["absa_val_file"], index=False)
    test.to_csv(cfg.modeling["absa_test_file"], index=False)
    logger.info(f"Saved splits to {out_dir.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export_path", type=str, help="Path to LS export (csv/tsv/json/jsonl)")
    args = ap.parse_args()
    main(args.export_path)
