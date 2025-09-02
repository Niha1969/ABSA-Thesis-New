import streamlit as st

st.set_page_config(page_title="Admin", page_icon="🛠️")
st.title("🛠️ Admin checklist")

st.markdown("""
1. **Install deps**  
2. **ETL**  
3. **Generate labeling tasks** (CSV import, you’re already using your generator)  
4. **Label in Label Studio** → Export CSV (Completed only) to `artifacts/labelstudio/export_*.csv`
5. **Clean + Stratify**  
- Clean export → `artifacts/labelstudio/export_clean.csv`  
- Split → `artifacts/absa/absa_train.csv`, `absa_val.csv`, `absa_test.csv`
6. **Train** 
make train
7. **Mine rules**  
make rules
8. **Run app**  
""")

