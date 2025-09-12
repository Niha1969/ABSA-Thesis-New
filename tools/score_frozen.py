#!/usr/bin/env python3
import argparse, os, re, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix

CAND_TEXT = ["sentence","text","review"]
CAND_GOLD = ["gold","label","polarity","polarity_gold","gold_label"]
CAND_PRED = ["pred","prediction","pred_label","baseline_pred","roberta_pred","vader_pred","textblob_pred"]

def pick(cols, cands):
    for c in cands:
        if c in cols: return c
    return None

def load_gold(path):
    df = pd.read_csv(path)
    t = pick(df.columns, CAND_TEXT)
    y = pick(df.columns, CAND_GOLD)
    if t is None or y is None:
        raise SystemExit(f"Gold needs text+gold columns. Got: {list(df.columns)}")
    df = df[[t,y]].rename(columns={t:"text", y:"gold"})
    df["text"] = df["text"].astype(str).str.strip()
    df["gold"] = df["gold"].str.lower().str.strip().replace(
        {"pos":"positive","neg":"negative","neu":"neutral"})
    return df

def infer_model_name(path, df):
    # prefer explicit pred column name
    for c in CAND_PRED:
        if c in df.columns:
            return c.replace("_pred","")
    # else use filename stem
    return Path(path).stem

def load_pred(path):
    df = pd.read_csv(path)
    t = pick(df.columns, CAND_TEXT)
    if t is None:
        # many preds still carry 'sentence'
        t = "sentence" if "sentence" in df.columns else None
    if t is None:
        raise SystemExit(f"Pred file {path} has no text/sentence column. Got: {list(df.columns)}")
    # find a single prediction column
    p = pick(df.columns, CAND_PRED)
    if p is None:
        # try any column that looks categorical (3 classes)
        for c in df.columns:
            if c==t: continue
            if df[c].astype(str).nunique()<=6:
                p=c; break
    if p is None:
        raise SystemExit(f"Pred file {path} has no prediction column. Got: {list(df.columns)}")
    out = df[[t,p]].rename(columns={t:"text", p:"pred"})
    out["text"] = out["text"].astype(str).str.strip()
    out["pred"] = out["pred"].astype(str).str.lower().str.strip().replace(
        {"pos":"positive","neg":"negative","neu":"neutral"})
    return out, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/training/frozen_metrics.csv")
    args = ap.parse_args()

    gold = load_gold(args.gold)

    rows=[]
    details=[]
    for pth in args.preds:
        pred_df, pred_col = load_pred(pth)
        name = infer_model_name(pth, pred_df)
        df = gold.merge(pred_df, on="text", how="inner")
        if len(df)==0:
            print(f"[WARN] Merge empty for {pth}")
            continue
        y_true = df["gold"].values
        y_pred = df["pred"].values
        macro = f1_score(y_true, y_pred, average="macro", labels=["negative","neutral","positive"])
        rows.append({"model":name, "n":len(df), "macro_f1":macro})
        # store per-class + confusion for appendix
        rep = classification_report(y_true, y_pred, labels=["negative","neutral","positive"], output_dict=True, zero_division=0)
        cm  = confusion_matrix(y_true, y_pred, labels=["negative","neutral","positive"])
        details.append((name, rep, cm))

    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))

    # optional: write detailed reports
    with open(Path(args.out).with_suffix(".txt"), "w") as f:
        for name, rep, cm in details:
            f.write(f"== {name} ==\n")
            f.write(pd.DataFrame(rep).T.to_string())
            f.write("\nConfusion (rows=gold):\n")
            f.write(pd.DataFrame(cm, index=["neg","neu","pos"], columns=["neg","neu","pos"]).to_string())
            f.write("\n\n")

if __name__=="__main__":
    main()
