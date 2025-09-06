import argparse, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

LABELS = ["negative","neutral","positive"]

def load_pred_csv(path: str, pred_col: str):
    df = pd.read_csv(path)
    if not {"gold", pred_col}.issubset(df.columns):
        raise SystemExit(f"{path} must have columns: gold,{pred_col}")
    y_true = df["gold"].astype(str)
    y_pred = df[pred_col].astype(str)
    return y_true, y_pred

def confusion(y_true, y_pred, labels):
    idx = {l:i for i,l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t,p in zip(y_true, y_pred):
        if t in idx and p in idx: m[idx[t], idx[p]] += 1
    return m

def plot_cm(cm, labels, title, out_png):
    fig, ax = plt.subplots(figsize=(4,4), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="CSV with columns gold,<pred_col>")
    ap.add_argument("--pred-col", required=True, help="Prediction column name (e.g. v2_pred, vader_pred)")
    ap.add_argument("--tag", required=True, help="Tag for filenames (e.g. v2, v1, vader, textblob)")
    ap.add_argument("--out-dir", default="artifacts/training")
    args = ap.parse_args()

    y_true, y_pred = load_pred_csv(args.pred_csv, args.pred_col)
    cm = confusion(y_true, y_pred, LABELS)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"cm_{args.tag}.png"
    csv = out_dir / f"cm_{args.tag}.csv"
    plot_cm(cm, LABELS, f"Confusion Matrix ({args.tag})", png)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(csv)
    print("Wrote:", png, "and", csv)

if __name__ == "__main__":
    main()
