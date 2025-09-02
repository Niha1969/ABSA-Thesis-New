import streamlit as st
import pandas as pd
from pathlib import Path
import json

st.title("🧭 Insights")

rules_fp = Path("artifacts/rules/rules.json")
if rules_fp.exists():
    rules = json.loads(rules_fp.read_text(encoding="utf-8"))
    if isinstance(rules, list) and rules:
        df = pd.DataFrame(rules)
        cols = [c for c in ["lhs","rhs","support","confidence","lift"] if c in df.columns]
        st.dataframe(df[cols].sort_values(["confidence","lift","support"], ascending=False).head(50), use_container_width=True)
    else:
        st.info("No rules yet. Run `make rules`.")
else:
    st.warning("`artifacts/rules/rules.json` not found. Run `make rules`.")
