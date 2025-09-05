import streamlit as st, yaml
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Observability", layout="wide")
st.title("Model Observability")

cfg = yaml.safe_load(open("config.yaml"))
serving = cfg.get("serving", {})
st.subheader("Serving")
st.write({"model_path": serving.get("model_path"), "labels_path": serving.get("labels_path")})

mc_path = Path("artifacts/training/mcnemar_summary.csv")
if mc_path.exists():
    st.subheader("McNemar (paired significance)")
    mc = pd.read_csv(mc_path)
    st.dataframe(mc, hide_index=True, use_container_width=True)
    # Friendly text for p-values that show as 0.0 due to underflow:
    if "p_value" in mc.columns and (mc["p_value"] == 0.0).any():
        st.caption("Note: p_value shown as 0.0 indicates p < 0.001 (floating underflow).")
else:
    st.info("No McNemar summary found at artifacts/training/mcnemar_summary.csv")

pred50k = Path("artifacts/preds/preds_50k.csv")
if pred50k.exists():
    st.subheader("50k batch — class distribution & uncertainty")
    d = pd.read_csv(pred50k)
    st.write(d["polarity_pred"].value_counts(normalize=True).rename("share").round(3))
    if {"prob_negative","prob_neutral","prob_positive"}.issubset(d.columns):
        maxp = d[["prob_negative","prob_neutral","prob_positive"]].max(axis=1)
        st.write({"uncertain_share(<0.6)": float((maxp<0.6).mean())})
else:
    st.info("No 50k prediction CSV found (artifacts/preds/preds_50k.csv)")
