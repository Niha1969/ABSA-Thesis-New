# tools/predict_roberta.py
import sys, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_dir = sys.argv[1]   # e.g., artifacts/models/roberta_absa_v2
test_csv = sys.argv[2]    # e.g., artifacts/absa/test_frozen_2025-09-02.csv
out_csv  = sys.argv[3]    # e.g., artifacts/training/preds_roberta_v2_test.csv

df = pd.read_csv(test_csv).copy()
text_col = "sentence" if "sentence" in df.columns else [c for c in df.columns if "text" in c.lower()][0]
df[text_col] = df[text_col].astype(str)

tok = AutoTokenizer.from_pretrained(model_dir)
mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
mdl.eval()

def batch_predict(texts, bs=32):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], truncation=True, padding=True, max_length=256, return_tensors="pt")
            logits = mdl(**enc).logits
            pred_ids = logits.argmax(-1).cpu().tolist()
            preds.extend(pred_ids)
    return preds

id2label = mdl.config.id2label
y_pred = [id2label[int(i)] for i in batch_predict(df[text_col].tolist())]
df_out = pd.DataFrame({"sentence": df[text_col].tolist(),
                       "gold": df["polarity"].astype(str).str.lower().tolist(),
                       "roberta_pred": y_pred})
df_out.to_csv(out_csv, index=False)
print("Wrote", out_csv, len(df_out))
