import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..config import get_settings

st.title("Inference — Polarity Classifier (demo)")
cfg = get_settings()

try:
    tok = AutoTokenizer.from_pretrained(cfg.modeling["model_name"])
    mdl = AutoModelForSequenceClassification.from_pretrained(cfg.modeling["model_name"])
except Exception:
    st.warning("Using unfine-tuned base model for demo.")
    tok = AutoTokenizer.from_pretrained(cfg.modeling["model_name"])
    mdl = AutoModelForSequenceClassification.from_pretrained(cfg.modeling["model_name"], num_labels=3)

txt = st.text_area("Enter a sentence about a product", "Battery drains after update but the screen is gorgeous.")
if st.button("Predict"):
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=160)
    with torch.no_grad():
        logits = mdl(**enc).logits
    pred = logits.softmax(-1).argmax(-1).item()
    st.write(f"Predicted class id: {pred} (label mapping depends on training)")
