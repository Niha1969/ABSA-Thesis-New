
#!/usr/bin/env python3
"""
EDA for Amazon Electronics reviews.

- Reads: data/data_raw/amazon_electronics.csv
- Writes figures: artifacts/eda/*.png (ratings, buckets, time series, lengths, n-grams)
- Writes summary: artifacts/eda/summary.txt
- Skips balanced sampling unless --write-balanced is set
"""

import argparse
import os
from pathlib import Path
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_IN = "data/data_raw/amazon_electronics.csv"
EDA_DIR = Path("artifacts/eda")

STOPWORDS = set("""
a an the and or but if then than while of for to in on at by with from this that these those i me my we our you your he she it they them is am are was were be been being do does did doing have has had having not no nor
""".split())

def ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist."""
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as e:
        print(f"Failed to create directory {p}: {e}")
        raise

def token_count(s: str) -> int:
    """Count alphanumeric tokens in a string."""
    if not isinstance(s, str):
        return 0
    return len(re.findall(r"[A-Za-z0-9]+", s))

def clean_text_len(s: str) -> int:
    """Count characters in cleaned text."""
    if not isinstance(s, str):
        return 0
    return len(s.replace("\n", " ").strip())

def map_bucket(r):
    """Map rating to sentiment bucket."""
    try:
        r = float(r)
        if r >= 4:
            return "pos"
        if r <= 2:
            return "neg"
        return "neu"
    except:
        return np.nan

def month_floor(s):
    """Floor date to month start."""
    try:
        return pd.to_datetime(s, errors="coerce").to_period("M").to_timestamp()
    except:
        return pd.NaT

def barplot(series: pd.Series, title: str, fname: str, xlabel: str = "", ylabel: str = "count"):
    """Save bar plot."""
    try:
        plt.figure(figsize=(7, 4))
        series.plot(kind="bar")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(EDA_DIR / fname, dpi=200)
        plt.close()
        print(f"Saved plot: {EDA_DIR / fname}")
    except Exception as e:
        print(f"Failed to save barplot {fname}: {e}")

def histplot(series: pd.Series, bins: int, title: str, fname: str, xlabel: str = "", ylabel: str = "count"):
    """Save histogram."""
    try:
        plt.figure(figsize=(7, 4))
        plt.hist(series.dropna(), bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(EDA_DIR / fname, dpi=200)
        plt.close()
        print(f"Saved plot: {EDA_DIR / fname}")
    except Exception as e:
        print(f"Failed to save histplot {fname}: {e}")

def lineplot(index: pd.Series, values: pd.Series, title: str, fname: str, xlabel: str = "", ylabel: str = "count"):
    """Save line plot."""
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(index, values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(EDA_DIR / fname, dpi=200)
        plt.close()
        print(f"Saved plot: {EDA_DIR / fname}")
    except Exception as e:
        print(f"Failed to save lineplot {fname}: {e}")

def top_ngrams(texts: pd.Series, n=1, topk=20):
    """Get top n-grams, excluding stopwords."""
    try:
        counts = {}
        for s in texts.dropna():
            toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", s)]
            toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
            if n == 1:
                grams = toks
            else:
                grams = ["_".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:topk]
        return pd.Series(dict(items))
    except Exception as e:
        print(f"Failed to compute n-grams: {e}")
        return pd.Series()

def write_summary(path: Path, lines: list):
    """Write summary text file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Saved summary: {path}")
    except Exception as e:
        print(f"Failed to save summary {path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="EDA for Amazon Electronics reviews")
    parser.add_argument("--in-csv", default=DEFAULT_IN, help="Path to amazon_electronics.csv")
    parser.add_argument("--write-balanced", type=int, default=0, help="If >0, write balanced tasks CSV")
    parser.add_argument("--out-csv", default="artifacts/labelstudio/tasks_balanced.csv", help="Output path for balanced CSV")
    args = parser.parse_args()

    print(f"Starting EDA with input: {args.in_csv}")
    ensure_dir(EDA_DIR)

    try:
        df = pd.read_csv(args.in_csv, encoding="utf-8", low_memory=False)
        print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Failed to load CSV {args.in_csv}: {e}")
        return

    # Hardcode columns (adjust these after checking your CSV)
    text_col = "text"
    rating_col = "rating"
    date_col = "date"  # Set to None if no date column
    id_col = "source_id"

    needed = [text_col, rating_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Error: CSV missing columns {missing}. Found: {df.columns.tolist()}")
        return

    # Data cleaning
    try:
        df["__text"] = df[text_col].astype(str).replace("", np.nan)
        df["__len_chars"] = df["__text"].map(clean_text_len)
        df["__len_tokens"] = df["__text"].map(token_count)
        df["__rating"] = pd.to_numeric(df[rating_col], errors="coerce")
        df["__bucket"] = df["__rating"].map(map_bucket)
        if date_col and date_col in df.columns:
            df["__month"] = df[date_col].map(month_floor)
        else:
            df["__month"] = pd.NaT
            print("Warning: No date column found, skipping time series plot")
    except Exception as e:
        print(f"Failed during data cleaning: {e}")
        return

    # Stats
    try:
        total = len(df)
        missing_text = int(df["__text"].isna().sum())
        zero_len = int((df["__len_chars"] == 0).sum())
        dup_by_text = int(df["__text"].duplicated(keep=False).sum())

        rating_counts = df["__rating"].value_counts(dropna=False).sort_index()
        bucket_counts = df["__bucket"].value_counts(dropna=False)
        monthly = df.dropna(subset=["__month"]).groupby("__month")["__text"].count().sort_index()
        len_chars = df["__len_chars"]
        len_tokens = df["__len_tokens"]
        uni = top_ngrams(df["__text"], n=1, topk=20)
        bi = top_ngrams(df["__text"], n=2, topk=20)
    except Exception as e:
        print(f"Failed during stats computation: {e}")
        return

    # Save plots
    barplot(rating_counts, "Star Rating Distribution", "rating_distribution.png", xlabel="Stars")
    barplot(bucket_counts, "Sentiment Buckets (from Stars)", "bucket_distribution.png", xlabel="Bucket")
    if not monthly.empty:
        lineplot(monthly.index, monthly.values, "Reviews per Month", "reviews_per_month.png", xlabel="Month")
    histplot(len_chars, bins=40, title="Text Length (Characters)", fname="len_chars_hist.png", xlabel="Chars")
    histplot(len_tokens, bins=40, title="Text Length (Tokens)", fname="len_tokens_hist.png", xlabel="Tokens")
    barplot(uni, "Top Unigrams (Stopwords Removed)", "top_unigrams.png", xlabel="Unigram")
    barplot(bi, "Top Bigrams (Stopwords Removed)", "top_bigrams.png", xlabel="Bigram")

    # Write summary
    lines = [
        "=== Amazon Electronics EDA Summary ===",
        f"Total rows: {total}",
        f"Missing text: {missing_text} | Zero-length text: {zero_len} | Duplicate texts: {dup_by_text}",
        "",
        "Rating distribution (stars):"
    ] + [f"  {idx}: {int(val)}" for idx, val in rating_counts.items()] + [
        "Bucket distribution (weak labels):"
    ] + [f"  {idx}: {int(val)}" for idx, val in bucket_counts.items()]
    if not monthly.empty:
        lines.append(f"Time span: {monthly.index.min().date()} → {monthly.index.max().date()} | Months: {monthly.shape[0]}")
    lines += [
        f"Length (chars): mean={len_chars.mean():.1f}, median={len_chars.median():.0f}, p95={len_chars.quantile(0.95):.0f}",
        f"Length (tokens): mean={len_tokens.mean():.1f}, median={len_tokens.median():.0f}, p95={len_tokens.quantile(0.95):.0f}",
        "",
        "Top unigrams:"
    ] + [f"  {k}: {v}" for k, v in uni.items()] + [
        "Top bigrams:"
    ] + [f"  {k}: {v}" for k, v in bi.items()]

    write_summary(EDA_DIR / "summary.txt", lines)
    print("\n".join(lines))
    print(f"\nSaved figures to: {EDA_DIR.resolve()}")

    # Balanced sampling (skipped unless requested)
    if args.write_balanced > 0:
        try:
            target = {
                "neg": math.ceil(args.write_balanced * 0.35),
                "neu": math.floor(args.write_balanced * 0.30),
                "pos": args.write_balanced - (math.ceil(args.write_balanced * 0.35) + math.floor(args.write_balanced * 0.30))
            }
            rng = np.random.default_rng(13)
            parts = []
            for b, k in target.items():
                pool = df[df["__bucket"] == b]
                take = min(k, len(pool))
                parts.append(pool.sample(take, random_state=13))
            balanced = pd.concat(parts).sample(frac=1, random_state=13).reset_index(drop=True)
            out_path = Path(args.out_csv)
            ensure_dir(out_path.parent)
            balanced[["__text"]].rename(columns={"__text": "text"}).to_csv(out_path, index=False)
            print(f"Wrote balanced tasks CSV: {out_path} (rows={len(balanced)}, target={target})")
        except Exception as e:
            print(f"Failed to write balanced CSV: {e}")

if __name__ == "__main__":
    main()
