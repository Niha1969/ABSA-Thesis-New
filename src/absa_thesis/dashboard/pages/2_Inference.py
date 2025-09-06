import streamlit as st, yaml, torch, re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Inference", layout="wide")
st.title("Inference")

cfg = yaml.safe_load(open("config.yaml"))
serv = cfg.get("serving", {})
model_dir = serv.get("model_path")

# --- Aspect matcher (multi-hit) ---
ASPECT_LEX = {
    "battery":    [r"battery", r"charge", r"charging", r"drain", r"power\s?bank", r"overheat", r"thermal"],
    "screen":     [r"screen", r"display", r"panel", r"oled", r"lcd", r"resolution", r"pixel", r"brightness", r"glare"],
    "performance":[r"lag(gy)?", r"slow", r"snappy", r"freeze", r"fps", r"stutter", r"speed", r"benchmark"],
    "updates":    [r"update(s|d)?", r"firmware", r"patch", r"os\s?version", r"security update"],
    "price":      [r"price", r"expensive", r"cheap", r"value", r"worth"],
    "design":     [r"design", r"build quality", r"weight", r"thin", r"thick", r"bezel", r"material"],
    "usability":  [r"\bui\b", r"\bux\b", r"interface", r"navigation", r"easy to use", r"user[- ]?friendly", r"intuitive"],
    "support":    [r"customer service", r"support", r"warranty", r"rma", r"helpdesk"],
    "privacy":    [r"privacy", r"tracking", r"data collection", r"spyware"],
    "ads":        [r"\bad(s|vert|vertisement)\b", r"bloatware", r"preloaded app"],
}
def match_aspects(text: str):
    t = (text or "").lower()
    hits = []
    for a, pats in ASPECT_LEX.items():
        if any(re.search(p, t) for p in pats):
            hits.append(a)
    return sorted(set(hits)) or ["none"]

@st.cache_resource(show_spinner=False)
def load_model(md):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(md)
    mdl = AutoModelForSequenceClassification.from_pretrained(md).to(device).eval()
    return tok, mdl, device

def predict(texts, tok, mdl, device, max_len=256, batch=64, labels=("negative","neutral","positive")):
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
st.caption(f"Using model: {model_dir}")
tok, mdl, device = load_model(model_dir)

col1, col2 = st.columns([1,1])

# --- Single sentence ---
with col1:
    st.subheader("Single sentence")
    text = st.text_area("Enter a sentence", height=120, placeholder="e.g., The battery drains fast but the screen is gorgeous.")
    if st.button("Predict"):
        if not text.strip():
            st.warning("Type something first.")
        else:
            out = predict([text], tok, mdl, device)
            row = out.iloc[0]
            aspects = match_aspects(text)
            st.write({"polarity": row.polarity_pred,
                      "probabilities": {
                        "negative": round(row.prob_negative,3),
                        "neutral":  round(row.prob_neutral,3),
                        "positive": round(row.prob_positive,3),
                      },
                      "aspects": aspects})
            st.progress(float(max(row.prob_negative, row.prob_neutral, row.prob_positive)))

# --- Batch CSV ---
with col2:
    st.subheader("Batch CSV")
    up = st.file_uploader("Upload CSV with a 'sentence' column", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if "sentence" not in df.columns:
            st.error("CSV must contain a 'sentence' column.")
        else:
            texts = df["sentence"].astype(str).tolist()
            with st.spinner("Running inference..."):
                out = predict(texts, tok, mdl, device)
                out["aspect"] = out["sentence"].map(lambda s: ", ".join(match_aspects(s)))
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(20), use_container_width=True, hide_index=True)
            st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
