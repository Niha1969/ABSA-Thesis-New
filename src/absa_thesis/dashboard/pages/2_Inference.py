# src/absa_thesis/dashboard/pages/2_Inference.py
from pathlib import Path
import os, yaml, streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("🧪 Inference")

# Force the root config.yaml
REPO_ROOT = Path(__file__).resolve().parents[4]  # .../src/absa_thesis/dashboard/pages -> up 3
CFG_PATH  = REPO_ROOT / "config.yaml"
cfg = yaml.safe_load(open(CFG_PATH, "r"))
model_path = cfg["modeling"]["model_name"]  # this is your ABSOLUTE path now

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)      # fail loud if missing
    mdl = AutoModelForSequenceClassification.from_pretrained(path)# fail loud if missing
    mdl.eval()
    # prefer labels.txt
    id2label = {}
    lbl_fp = Path(path) / "labels.txt"
    if lbl_fp.exists():
        for line in open(lbl_fp, "r", encoding="utf-8"):
            i, lab = line.strip().split("\t")
            id2label[int(i)] = lab
    elif getattr(mdl.config, "id2label", None):
        tmp = {int(k): v for k, v in mdl.config.id2label.items()}
        if not any(str(v).startswith("LABEL_") for v in tmp.values()):
            id2label = tmp
    if not id2label:
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
    return tok, mdl, id2label

# Debug block so you can see what it’s actually loading
with st.expander("Debug"):
    st.write({"config_file": str(CFG_PATH.resolve()),
              "model_path": str(Path(model_path).resolve())})
    try:
        items = sorted(os.listdir(model_path))
        st.write({"dir_len": len(items), "sample": items[:10]})
    except Exception as e:
        st.error(f"Cannot list model dir: {e}")

# Optional: a reload button to clear the cached model
if st.button("🔄 Reload model (clear cache)"):
    load_model_and_tokenizer.clear()
    st.success("Cache cleared. Predict again to reload.")

tok, mdl, id2label = load_model_and_tokenizer(model_path)

txt = st.text_area("Paste a single review sentence (20–300 chars).", height=120)
if st.button("Predict", type="primary"):
    s = (txt or "").strip()
    if not s:
        st.warning("Give me a sentence.")
    else:
        with torch.no_grad():
            b = tok([s], padding=True, truncation=True, max_length=256, return_tensors="pt")
            out = mdl(**b)
            probs = torch.softmax(out.logits, dim=-1)[0].tolist()
            pred = int(torch.argmax(out.logits, dim=-1).item())
        st.success(f"Predicted: **{id2label.get(pred, pred)}**")
        st.caption("Probabilities → " + ", ".join(f"{id2label[i]}: {probs[i]:.3f}" for i in range(len(probs))))
