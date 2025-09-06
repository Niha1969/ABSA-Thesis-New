# tools/eval_baselines.py
import argparse, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from textblob import TextBlob
import yaml, json

# Prefer vaderSentiment (bundled lexicon, no NLTK download needed)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VSAnalyzer
    _VADER_IMPL = "vs"
except Exception:
    _VADER_IMPL = None

def load_cfg():
    return yaml.safe_load(open("config.yaml"))

def predict_vader(texts):
    if _VADER_IMPL == "vs":
        sia = VSAnalyzer()
        out=[]
        for t in texts:
            s = sia.polarity_scores(t or "")
            c = "positive" if s["compound"] >= 0.05 else "negative" if s["compound"] <= -0.05 else "neutral"
            out.append(c)
        return out
    # Fallback path (only if you insist on nltk)
    raise SystemExit("Install vaderSentiment: pip install vaderSentiment (recommended).")

def predict_textblob(texts):
    out=[]
    for t in texts:
        p = TextBlob(t or "").sentiment.polarity
        c = "positive" if p > 0.05 else "negative" if p < -0.05 else "neutral"
        out.append(c)
    return out

def report(y_true, y_pred):
    labels = ["negative","neutral","positive"]
    macro = f1_score(y_true, y_pred, labels=labels, average="macro")
    per = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return macro, per, cm, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default=None, help="Frozen test CSV with columns: sentence,gold")
    ap.add_argument("--out-dir", default="artifacts/training")
    args = ap.parse_args()

    cfg = load_cfg()
    in_csv = args.in_csv or cfg["paths"].get("test_frozen_file")
    if not in_csv or not Path(in_csv).exists():
        raise SystemExit(f"Frozen test CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    if not {"sentence","gold"}.issubset(df.columns):
        raise SystemExit("Test CSV must have columns: sentence,gold")

    y = df["gold"].astype(str).tolist()
    texts = df["sentence"].astype(str).tolist()

    preds = {}
    preds["vader"] = predict_vader(texts)
    preds["textblob"] = predict_textblob(texts)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, yp in preds.items():
        macro, per, cm, labels = report(y, yp)
        rows.append({"model": name, "macro_f1": round(macro,3),
                     "f1_negative": round(per["negative"]["f1-score"],3),
                     "f1_neutral":  round(per["neutral"]["f1-score"],3),
                     "f1_positive": round(per["positive"]["f1-score"],3)})
        pd.DataFrame({"sentence": texts, "gold": y, f"{name}_pred": yp}).to_csv(out_dir/f"{name}_preds.csv", index=False)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir/f"{name}_cm.csv")

    # Pull v1/v2 reports if present
    for tag in ["v1","v2"]:
        jf = out_dir/f"{tag}_report.json"
        if jf.exists():
            rep = json.loads(Path(jf).read_text())
            rows.append({
                "model": tag,
                "macro_f1": round(rep.get("macro_f1", np.nan),3),
                "f1_negative": round(rep.get("per_class",{}).get("negative",np.nan),3),
                "f1_neutral":  round(rep.get("per_class",{}).get("neutral",np.nan),3),
                "f1_positive": round(rep.get("per_class",{}).get("positive",np.nan),3),
            })

    pd.DataFrame(rows).to_csv(out_dir/"baselines_summary.csv", index=False)
    print("Wrote:", out_dir/"baselines_summary.csv")
    for k in preds:
        print("Saved raw preds:", out_dir/f"{k}_preds.csv")

if __name__ == "__main__":
    main()
