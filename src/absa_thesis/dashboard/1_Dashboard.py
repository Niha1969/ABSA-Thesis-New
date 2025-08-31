import streamlit as st, pandas as pd, json
from ..config import get_settings
from ..utils import ensure_dir

st.title("Dashboard — Insights")
cfg = get_settings()

# Load rules if available
rules_json = ensure_dir(cfg.paths["rules_dir"]) / "rules.json"
if rules_json.exists():
    rules = json.loads(rules_json.read_text())
    st.subheader("Top Rules (by confidence)")
    st.dataframe(pd.DataFrame(rules)[["antecedents","consequents","support","confidence","lift"]].head(20))
else:
    st.info("No rules found. Run `make rules` after labeling some data.")
