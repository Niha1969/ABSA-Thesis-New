# streamlit_app/pages/01_Observability.py
import streamlit as st, yaml
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Observability", layout="wide")
st.title("Model Observability")

cfg = yaml.safe_load(open("config.yaml"))
serving = cfg.get("serving", {})
st.subheader("Serving")
st.write({"model_path": serving.get("model_path"), "labels_path": serving.get("labels_path")})

# --- McNemar
mc_path = Path("artifacts/training/mcnemar_summary.csv")
if mc_path.exists():
    st.subheader("Paired significance (McNemar)")
    mc = pd.read_csv(mc_path)
    st.dataframe(mc, hide_index=True, use_container_width=True)
    if "p_value" in mc.columns and (mc["p_value"] == 0.0).any():
        st.caption("Note: p_value shown as 0.0 indicates p < 0.001 (floating underflow).")
else:
    st.info("No McNemar summary at artifacts/training/mcnemar_summary.csv")

# --- 50k batch charts
pred50k = Path("artifacts/preds/preds_50k.csv")
if pred50k.exists():
    st.subheader("50k batch — distribution & uncertainty")
    d = pd.read_csv(pred50k)

    # Class distribution
    dist = d["polarity_pred"].value_counts().rename_axis("polarity").reset_index(name="count")
    st.bar_chart(dist.set_index("polarity"))

    # Uncertainty histogram (FIX: convert bin labels to str)
    if {"prob_negative","prob_neutral","prob_positive"}.issubset(d.columns):
        maxp = d[["prob_negative","prob_neutral","prob_positive"]].max(axis=1)
        st.markdown(f"Uncertain fraction (max prob < 0.6): **{(maxp<0.6).mean():.3f}**")
        bins = pd.cut(maxp, bins=[0,0.5,0.6,0.7,0.8,0.9,1.0], include_lowest=True)
        hist = bins.value_counts().sort_index()
        hist_df = pd.DataFrame({"bin": hist.index.astype(str), "count": hist.values})
        st.bar_chart(hist_df, x="bin", y="count")
else:
    st.info("No 50k prediction CSV found (artifacts/preds/preds_50k.csv)")

# --- Confusion matrices (auto-gallery) ---
st.subheader("Confusion matrices")

from pathlib import Path

def find_cm_pngs():
    base = Path.cwd()
    cand_dirs = [
        base / "artifacts" / "training",
        Path(__file__).resolve().parents[1] / "artifacts" / "training",  # repo root fallback
    ]
    pngs = []
    for d in cand_dirs:
        if d.exists():
            pngs += list(d.glob("cm_*.png"))                # our generator
            pngs += list(d.glob("*confusion*matrix*.png"))  # any older naming
    # de-dup + sort
    return sorted(set(pngs), key=lambda p: p.name)

pngs = find_cm_pngs()

st.caption("DEBUG found: " + ", ".join(p.name for p in pngs))

if not pngs:
    st.caption("No confusion matrix images found. Generate with tools/make_confusion_matrix.py.")
else:
    cols = st.columns(3)
    for i, png in enumerate(pngs):
        data = png.read_bytes()  # avoids browser cache issues
        caption = png.name.replace("cm_","").replace(".png","")
        with cols[i % 3]:
            st.image(data, caption=caption or png.name, use_column_width=True)

# --- Baseline comparison table
bcmp = Path("artifacts/training/baselines_summary.csv")
if bcmp.exists():
    st.subheader("Baselines vs ABSA models")
    st.dataframe(pd.read_csv(bcmp), use_container_width=True, hide_index=True)

