# streamlit_app/Home.py
import streamlit as st, yaml
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="ABSA Dashboard", layout="wide")
st.markdown("<h1 style='margin-bottom:0'>ABSA Dashboard</h1>", unsafe_allow_html=True)
st.caption("Model status • Data health • Quick links")

cfg = yaml.safe_load(open("config.yaml"))
serv = cfg.get("serving", {})
art_paths = {
    "Rules CSV": Path("artifacts/rules/rules.csv"),
    "Preds 50k": Path("artifacts/preds/preds_50k.csv"),
    "Preds 50k (aspect)": Path("artifacts/preds/preds_50k_aspect_for_rules.csv"),
    "McNemar summary": Path("artifacts/training/mcnemar_summary.csv"),
    "Baselines summary": Path("artifacts/training/baselines_summary.csv"),
}

# Cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🧠 Serving Model")
    st.write(serv.get("model_path","<not set>"))
    st.write(f"Labels: {serv.get('labels_path','<not set>')}")
with c2:
    st.markdown("### 📦 Artifacts")
    for k,p in art_paths.items():
        st.write(f"{'✅' if p.exists() else '❌'} {k}")
with c3:
    st.markdown("### ⚙️ Config")
    st.json({"rules": cfg.get("rules",{}), "dashboard": cfg.get("dashboard",{})})

st.markdown("---")
st.subheader("Quick links")
st.write("➡️ Go to **Observability** for metrics and baselines.")
st.write("➡️ Go to **Insights** for Aspect↔Polarity rules and examples.")
st.write("➡️ Go to **Inference** for single sentence & CSV batch predictions.")
