"""
Add an 'aspect' column to a predictions CSV via keyword heuristics.
Input: artifacts/preds/preds_50k.csv (sentence, polarity_pred, ...)
Output: artifacts/preds/preds_50k_aspect.csv (adds 'aspect')
If multiple aspects match, pick the one with most hits; if none, 'none'.
"""
import re, sys
import pandas as pd
from collections import Counter
from pathlib import Path

ASPECT_LEX = {
    "battery":    [r"battery", r"charge", r"charging", r"drain", r"power\s?bank"],
    "screen":     [r"screen", r"display", r"panel", r"oled", r"lcd", r"resolution", r"pixel"],
    "performance":[r"lag", r"slow", r"snappy", r"freeze", r"fps", r"benchmark", r"speed"],
    "updates":    [r"update(s|d)?", r"firmware", r"patch", r"os\s?version", r"security update"],
    "price":      [r"price", r"expensive", r"cheap", r"value", r"worth"],
    "design":     [r"design", r"build quality", r"weight", r"thin", r"thick", r"bezel", r"material"],
    "usability":  [r"ui", r"ux", r"interface", r"navigation", r"easy to use", r"user[- ]?friendly"],
    "support":    [r"customer service", r"support", r"warranty", r"rma", r"helpdesk"],
    "privacy":    [r"privacy", r"tracking", r"data collection", r"spyware"],
    "ads":        [r"ad(s|vert|vertisement)", r"bloatware", r"preloaded app"],
}

def match_aspect(text: str) -> str:
    text = text.lower()
    hits = []
    for a, pats in ASPECT_LEX.items():
        count = sum(1 for p in pats if re.search(p, text))
        if count:
            hits.append((a, count))
    if not hits:
        return "none"
    # choose aspect with most matches; tie → first
    return sorted(hits, key=lambda x: -x[1])[0][0]

def main(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    assert "sentence" in df.columns and "polarity_pred" in df.columns
    df["aspect"] = df["sentence"].astype(str).map(match_aspect)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df):,} rows → {out_csv} with 'aspect'")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
