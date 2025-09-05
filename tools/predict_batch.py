
"""
Batch infer polarity on a CSV with 'sentence' column.
Reads model from config.yaml -> serving.model_path (and labels.txt if present).
Writes artifacts/preds/preds_50k.csv with:
  sentence, polarity_pred, prob_negative, prob_neutral, prob_positive
"""
import sys, yaml, torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def pick_device():
    if torch.backends.mps.is_available():  # Apple M-series GPU
        return torch.device("mps")
    return torch.device("cpu")

def load_labels_txt(model_dir: Path):
    f = model_dir / "labels.txt"
    if f.exists():
        # format: "0\tnegative" per line
        idx2lab = {}
        for line in f.read_text().splitlines():
            if not line.strip(): continue
            idx, lab = line.split("\t")
            idx2lab[int(idx)] = lab.strip()
        return [idx2lab[i] for i in sorted(idx2lab)]
    return ["negative","neutral","positive"]

def batched(xs, n=64):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def main(in_csv, out_csv, max_len=256, bs=64):
    cfg = load_cfg()
    model_dir = Path(cfg["serving"]["model_path"])
    labels = load_labels_txt(model_dir)
    device = pick_device()

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    df = pd.read_csv(in_csv)
    assert "sentence" in df.columns, "Input must have 'sentence'"
    texts = df["sentence"].astype(str).tolist()

    y_pred, y_prob = [], []

    with torch.no_grad():
        for chunk in batched(texts, bs):
            enc = tok(chunk, max_length=max_len, padding=True, truncation=True, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            p = logits.softmax(-1).cpu().numpy()
            y_prob.extend(p.tolist())
            y_pred.extend([labels[i] for i in p.argmax(1)])

    p = np.array(y_prob)
    out = pd.DataFrame({
        "sentence": texts,
        "polarity_pred": y_pred,
        "prob_negative": p[:,0],
        "prob_neutral":  p[:,1],
        "prob_positive": p[:,2],
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out):,} rows → {out_csv}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
