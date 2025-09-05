"""
Create a reproducible 50k slice from data/data_raw/amazon_electronics.csv
- Renames your text column to 'sentence'
- Drops dupes & extreme lengths so inference is stable
- Writes artifacts/preds/sample50k.csv
"""
import pandas as pd
from pathlib import Path

RAW = Path("data/data_raw/amazon_electronics.csv")
OUT = Path("artifacts/preds/sample50k.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

# try common text column names; rename to 'sentence'
for cand in ["sentence","text","reviewText","content","body"]:
    if cand in df.columns:
        df = df.rename(columns={cand: "sentence"})
        break
assert "sentence" in df.columns, "Couldn't find a text column in the CSV."

# basic clean + length filter + dedup
s = df["sentence"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df = df.assign(sentence=s)
df = df[df["sentence"].str.len().between(20, 300)]
df = df.drop_duplicates(subset=["sentence"])

# reproducible sample of up to 50k
SAMPLE_N = 50_000
if len(df) > SAMPLE_N:
    df = df.sample(n=SAMPLE_N, random_state=42)

df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows → {OUT}")
