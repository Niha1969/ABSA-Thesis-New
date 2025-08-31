"""Compute SUS scores from a CSV with columns Q1..Q10 (1-5)."""
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

def compute_row(row):
    pos = [1,3,5,7,9]
    neg = [2,4,6,8,10]
    score = 0
    for i in pos:
        score += (row[f"Q{i}"] - 1)
    for i in neg:
        score += (5 - row[f"Q{i}"])
    return score * 2.5

def main(csv_path: str):
    df = pd.read_csv(csv_path)
    sus = df.apply(compute_row, axis=1)
    print(f"Mean SUS: {sus.mean():.1f}")
    print(f"n={len(sus)}; scores: {sus.tolist()}" )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.absa_thesis.study.compute_sus path/to/sus.csv")
        sys.exit(1)
    main(sys.argv[1])
