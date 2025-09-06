import argparse, json, torch, pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative","neutral","positive"]

def load_model(model_dir):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return tok, mdl, device

def predict(texts, tok, mdl, device, max_len=256, batch=64):
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            p = logits.softmax(-1).cpu().numpy()
            preds.extend([LABELS[j] for j in p.argmax(1)])
            probs.extend(p.tolist())
    out = pd.DataFrame({
        "sentence": texts,
        "polarity_pred": preds,
        "prob_negative": [p[0] for p in probs],
        "prob_neutral":  [p[1] for p in probs],
        "prob_positive": [p[2] for p in probs],
    })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--in-csv", required=True, help="CSV with columns: sentence,gold")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out-dir", default="artifacts/training")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    assert {"sentence","gold"}.issubset(df.columns), "in-csv must have sentence,gold"
    texts = df["sentence"].astype(str).tolist()
    y = df["gold"].astype(str).tolist()

    tok, mdl, device = load_model(args.model_dir)
    preds = predict(texts, tok, mdl, device)
    merged = df[["sentence","gold"]].merge(preds, on="sentence", how="left").rename(columns={"polarity_pred": f"{args.tag}_pred"})

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f"{args.tag}_pred.csv", index=False)

    y_pred = merged[f"{args.tag}_pred"].astype(str).tolist()
    labels = LABELS
    macro = f1_score(y, y_pred, labels=labels, average="macro")
    per = classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)
    rep = {"macro_f1": float(macro),
           "per_class": {lab: float(per[lab]["f1-score"]) for lab in labels}}
    (out_dir / f"{args.tag}_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"{args.tag} macro_F1={macro:.3f}  per-class={rep['per_class']}")
    print(f"Wrote {out_dir / f'{args.tag}_pred.csv'} and {out_dir / f'{args.tag}_report.json'}")

if __name__ == "__main__":
    main()
