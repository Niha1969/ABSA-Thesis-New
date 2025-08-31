import streamlit as st, pandas as pd
from ..config import get_settings
from ..utils import ensure_dir

st.title("Observability — Data & Performance")
cfg = get_settings()

clean_pq = ensure_dir(cfg.paths["clean_dir"]) / "reviews_clean.parquet"
if clean_pq.exists():
    df = pd.read_parquet(clean_pq)
    st.write(df.sample(min(10, len(df))))
else:
    st.info("No data yet. Run `make etl`.")
