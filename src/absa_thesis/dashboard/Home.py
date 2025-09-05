import streamlit as st, yaml, os
from pathlib import Path

st.set_page_config(page_title="ABSA Dashboard", layout="wide")
st.title("ABSA Dashboard")

cfg = yaml.safe_load(open("config.yaml"))
serving = cfg.get("serving", {})
model_path = serving.get("model_path", "<not set>")
labels_path = serving.get("labels_path", "<not set>")

st.markdown("### Serving")
st.write({"model_path": model_path, "labels_path": labels_path})

artifacts = {
    "Rules CSV": "artifacts/rules/rules.csv",
    "Rules JSON": "artifacts/rules/rules.json",
    "Predictions 50k (aspect)": "artifacts/preds/preds_50k_aspect_for_rules.csv",
    "McNemar summary": "artifacts/training/mcnemar_summary.csv",
}
st.markdown("### Artifacts")
for k, p in artifacts.items():
    st.write(f"{k}: {'✅' if Path(p).exists() else '❌'} — {p}")
