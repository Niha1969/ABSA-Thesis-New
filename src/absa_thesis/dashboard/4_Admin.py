import streamlit as st
from ..config import get_settings
st.title("Admin — Setup & Checks")
cfg = get_settings()

st.markdown("""
**Setup checklist**
1. `make install`
2. `make etl`
3. Label sentences (Label Studio), then `merge_annotations`
4. `make train`
5. `make rules`
6. `make dashboard`
""")
