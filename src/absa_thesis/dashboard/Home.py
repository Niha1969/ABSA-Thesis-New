import streamlit as st
from ..config import get_settings
from ..utils import ensure_dir
import pandas as pd
st.set_page_config(page_title="BrandPulse", page_icon="📊", layout="wide")

cfg = get_settings()
st.title("BrandPulse — Voice of Customer Dashboard")

st.markdown("""
This dashboard surfaces aspect-level sentiment and association rules for tech products.
Use the sidebar to navigate pages. If data is missing, pages will display setup steps.
""")

# Quick health check
clean_pq = ensure_dir(cfg.paths["clean_dir"]) / "reviews_clean.parquet"
if clean_pq.exists():
    df = pd.read_parquet(clean_pq)
    st.metric("Clean reviews", len(df))
else:
    st.warning("No cleaned data found. Run `make etl` to generate sample data.")
