#!/usr/bin/env python3
import re, random, argparse
import numpy as np, pandas as pd
from typing import List

POS_WORDS = ["great","excellent","love","amazing","perfect","awesome","fantastic","brilliant"]
NEG_WORDS = ["bad","terrible","awful","hate","trash","worst","broken","refund","return","slow","bug","issue","problem"]

def split_sentences(t: str):
    # Simple splitter; keep 20–300 chars
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', str(t).strip())
    return [s for s in parts if 20 <= len(s) <= 300]

def looks_neutral(s: str):
    s = s.lower()
    neg = sum(s.count(w) for w in NEG_WORDS)
    pos = sum(s.count(w) for w in POS_WORDS)
    return (neg == 0 and pos == 0)

def looks_positive(s: str):
    return re.search(r"\b(" + "|".join(POS_WORDS) + r")\b", s, flags=re.I) is not None

def looks_negative(s: str):
    return re.search(r"\b(" + "|".join(NEG_WORDS) + r")\b", s, flags=re.I) is not None

def detect_text_col(df: pd.DataFrame) -> str:
    for c in ["sentence","text","review","content","body","comment"]:
        if c in df.columns: return c
    # very defensive fallback
    for c in df.columns:
        if "text" in c.lower() or "review" in c.lower():
            return c
    raise ValueError("No suitable text column found in dataframe")

def load_exclude_sentences(paths: List[str]) -> set:
    excl = set()
    for p in paths:
        if not p: continue
        df = pd.read_csv(p)
        col = detect_text_col(df)
        excl |= set(df[col].astype(str).str.strip())
    return excl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", default="data/data_clean/reviews_clean.parquet")
    ap.add_argument("--out_csv", default="artifacts/labelstudio/tasks_batch1.csv")
    ap.add_argument("--sample_rows", type=int, default=40000, help="max reviews to scan")

    # class targets (sum ~= desired batch size)
    ap.add_argument("--n_neu", type=int, default=400)
    ap.add_argument("--n_pos", type=int, default=300)
    ap.add_argument("--n_neg", type=int, default=200)

    # NEW: exclusions
    ap.add_argument("--exclude_csv", nargs="*", default=[], help="One or more CSVs with labeled sentences to exclude")
    ap.add_argument("--exclude_test", type=str, default=None, help="CSV of frozen test set to exclude")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    df = pd.read_parquet(args.in_parquet)
    if len(df) > args.sample_rows:
        df = df.sample(args.sample_rows, random_state=args.seed)

    cand = []
    # Assume raw DataFrame has a 'text' column; if not, try to detect
    text_col = "text" if "text" in df.columns else detect_text_col(df)
    for _, r in df.iterrows():
        for s in split_sentences(r[text_col]):
            cand.append({"platform": r.get("platform",""), "source_id": r.get("source_id",""), "sentence": s})

    cand = pd.DataFrame(cand).drop_duplicates(subset=["sentence"])
    if cand.empty:
        raise SystemExit("No candidate sentences found. Check your input data.")

    # Build exclusion set from prior exports + frozen test
    excl_paths = list(args.exclude_csv or [])
    if args.exclude_test:
        excl_paths.append(args.exclude_test)
    exclude = load_exclude_sentences(excl_paths)

    if exclude:
        before = len(cand)
        cand = cand[~cand["sentence"].astype(str).str.strip().isin(exclude)]
        print(f"Excluded {before - len(cand)} previously seen/frozen sentences; {len(cand)} remain.")

    # Heuristic buckets
    neu = cand[cand["sentence"].map(looks_neutral)]
    pos = cand[cand["sentence"].map(looks_positive)]
    neg = cand[cand["sentence"].map(looks_negative)]

    # Sample per target counts (cap by availability)
    neu = neu.sample(min(args.n_neu, len(neu)), random_state=args.seed)
    pos = pos.sample(min(args.n_pos, len(pos)), random_state=args.seed)
    neg = neg.sample(min(args.n_neg, len(neg)), random_state=args.seed)

    batch = pd.concat([neu, pos, neg], ignore_index=True).drop_duplicates(subset=["sentence"])
    batch = batch.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out = batch[["sentence","platform","source_id"]].copy()
    out.to_csv(args.out_csv, index=False)  # UTF-8

    print("Saved:", args.out_csv)
    print("Counts approx → neutral-like:", len(neu), "positive-like:", len(pos), "negative-like:", len(neg))
    print("Total rows:", len(out))

if __name__ == "__main__":
    main()
