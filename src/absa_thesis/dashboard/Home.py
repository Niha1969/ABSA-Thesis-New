import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Home", page_icon="🏠")
st.title("🏠 ABSA Dashboard")

clean_fp = Path("data/data_clean/reviews_clean.parquet")
if clean_fp.exists():
    df = pd.read_parquet(clean_fp)
    st.metric("Clean reviews loaded", len(df))
else:
    st.warning("Run `make etl` to generate `reviews_clean.parquet`.")

st.markdown("Use the sidebar to navigate: Insights → Inference → Observability → Admin.")
