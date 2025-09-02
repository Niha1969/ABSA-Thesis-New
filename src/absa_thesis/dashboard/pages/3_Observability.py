import streamlit as st
import pandas as pd
from pathlib import Path

st.title("📊 Observability")

absa_dir = Path("artifacts/absa")
for fp in [absa_dir/"absa_train.csv", absa_dir/"absa_val.csv", absa_dir/"absa_test.csv"]:
    if fp.exists():
        df = pd.read_csv(fp)
        st.metric(fp.stem, len(df))
        st.write(df["polarity"].value_counts(normalize=True).round(2))
    else:
        st.warning(f"{fp} not found")

st.divider()
rep_fp = Path("artifacts/training/val_report.txt")
if rep_fp.exists():
    st.subheader("Validation report")
    st.code(rep_fp.read_text(encoding="utf-8"), language="text")
else:
    st.info("No val report yet. Run training.")
