import streamlit as st, yaml, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Inference", layout="wide")
st.title("Batch Inference")

cfg = yaml.safe_load(open("config.yaml"))
serv = cfg.get("serving", {})
model_dir = serv.get("model_path")
labels_path = serv.get("labels_path")

@st.cache_resource(show_spinner=False)
def load_model(md):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(md)
    mdl = AutoModelForSequenceClassification.from_pretrained(md).to(device).eval()
    return tok, mdl, device

def predict_sentences(texts, tok, mdl, device, max_len=256, batch=64, labels=("negative","neutral","positive")):
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], max_length=max_len, padding=True, truncation=True, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            p = logits.softmax(-1).cpu().numpy()
            preds.extend([labels[j] for j in p.argmax(1)])
            probs.extend(p.tolist())
    out = pd.DataFrame({
        "sentence": texts,
        "polarity_pred": preds,
        "prob_negative": [p[0] for p in probs],
        "prob_neutral":  [p[1] for p in probs],
        "prob_positive": [p[2] for p in probs],
    })
    return out

if not model_dir or not Path(model_dir).exists():
    st.error("serving.model_path missing or invalid in config.yaml")
    st.stop()

labels = ("negative","neutral","positive")
st.caption(f"Using model: {model_dir}")

tok, mdl, device = load_model(model_dir)

up = st.file_uploader("Upload CSV with a 'sentence' column", type=["csv"])
if up is not None:
    df = pd.read_csv(up)
    if "sentence" not in df.columns:
        st.error("CSV must contain a 'sentence' column.")
    else:
        texts = df["sentence"].astype(str).tolist()
        with st.spinner("Running inference..."):
            out = predict_sentences(texts, tok, mdl, device)
        st.success(f"Predicted {len(out)} rows.")
        st.dataframe(out.head(20), use_container_width=True, hide_index=True)
        st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
