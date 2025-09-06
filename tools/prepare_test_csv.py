import argparse, pandas as pd
from pathlib import Path
import yaml

LABEL_MAPS = [
    # common numeric encodings
    { -1:"negative", 0:"neutral", 1:"positive" },
    { 0:"negative", 1:"neutral", 2:"positive" },
    { 0:"negative", 1:"positive" },  # binary -> map neutral to positive (fallback)
]

TEXT_COLS = ["sentence","text","review","content","comment"]
GOLD_COLS = ["gold","label","labels","polarity","target","y"]

def normalize_labels(series):
    s = series.astype(str).str.lower().str.strip()
    # already string labels?
    if set(s.unique()) <= {"negative","neutral","positive"}:
        return s
    # try numeric maps
    try:
        num = pd.to_numeric(series, errors="raise")
        for m in LABEL_MAPS:
            if set(num.unique()).issubset(set(m.keys())):
                return num.map(m).astype(str)
    except Exception:
        pass
    # last resort: heuristics
    s2 = s.replace({"pos":"positive","neg":"negative","neu":"neutral"})
    if set(s2.unique()) <= {"negative","neutral","positive"}:
        return s2
    raise SystemExit(f"Could not map labels to {{negative,neutral,positive}}. Found: {sorted(set(series.unique()))}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None,
                    help="Path to your frozen test CSV; if omitted, uses config.paths.test_frozen_file")
    ap.add_argument("--out", dest="outp", default="artifacts/training/frozen_test_for_eval.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    inp = args.inp or cfg.get("paths",{}).get("test_frozen_file")
    if not inp or not Path(inp).exists():
        raise SystemExit(f"Frozen test CSV not found: {inp}")

    df = pd.read_csv(inp)
    cols_lower = {c.lower(): c for c in df.columns}

    # find text column
    text_col = next((cols_lower[c] for c in TEXT_COLS if c in cols_lower), None)
    if text_col is None:
        raise SystemExit(f"No text column found. Tried: {TEXT_COLS}. Columns: {list(df.columns)}")

    # find label column
    gold_col = next((cols_lower[c] for c in GOLD_COLS if c in cols_lower), None)
    if gold_col is None:
        raise SystemExit(f"No label column found. Tried: {GOLD_COLS}. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "sentence": df[text_col].astype(str),
        "gold": normalize_labels(df[gold_col])
    })
    Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outp, index=False)
    print(f"Wrote {len(out)} rows → {args.outp}")
    print(out.head(3))

if __name__ == "__main__":
    main()
