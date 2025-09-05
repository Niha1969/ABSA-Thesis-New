import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Insights", layout="wide")
st.title("Insights: Aspect ↔ Polarity Rules")

RULES_CSV = Path("artifacts/rules/rules.csv")
PREDS = Path("artifacts/preds/preds_50k_aspect_for_rules.csv")

if not RULES_CSV.exists():
    st.warning("No rules.csv found. Run: python tools/make_rules_from_counts.py")
    st.stop()

rules = pd.read_csv(RULES_CSV)
if rules.empty:
    st.info("rules.csv is empty. Adjust thresholds or regenerate.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    direction = st.radio("Direction", ["All","Aspect→Positive (AP)","Negative→Aspect (PA)"], index=0)
    min_support = st.slider("Min support", 0.0, 0.05, 0.01, 0.001)
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.5, 0.01)
    min_lift = st.slider("Min lift", 1.0, 2.0, 1.05, 0.01)

df = rules.copy()
if direction != "All":
    df = df[df["direction"] == ("ap" if direction.startswith("Aspect") else "pa")]
df = df[(df["support"] >= min_support) & (df["confidence"] >= min_conf) & (df["lift"] >= min_lift)]

def to_aspect(row):
    if row["direction"] == "ap":
        return row["antecedents"].split("=",1)[-1]  # Aspect=xxx
    return row["consequents"].split("=",1)[-1]      # Aspect=xxx

if len(df):
    df["aspect"] = df.apply(to_aspect, axis=1)
    aspects = sorted(df["aspect"].unique())
    sel_aspects = st.sidebar.multiselect("Aspects", aspects, default=aspects)
    df = df[df["aspect"].isin(sel_aspects)]

st.subheader(f"Rules ({len(df)})")
st.dataframe(
    df[["direction","antecedents","consequents","support","confidence","lift"]]
      .sort_values(["direction","confidence","support","lift"], ascending=[True, False, False, False]),
    hide_index=True, use_container_width=True
)

st.download_button("Download rules.csv", RULES_CSV.read_bytes(), file_name="rules.csv", type="secondary")

st.markdown("---")
st.subheader("Examples")

if not PREDS.exists():
    st.info("No example corpus found (artifacts/preds/preds_50k_aspect_for_rules.csv).")
else:
    preds = pd.read_csv(PREDS)
    preds["polarity"] = preds["polarity"].astype(str)
    if len(df) == 0:
        st.info("No rules after filtering.")
    else:
        idx = st.number_input("Enter a rule row number (0-based)", min_value=0, max_value=max(len(df)-1,0), value=0, step=1)
        row = df.reset_index(drop=True).iloc[int(idx)]
        st.write(f"**Rule:** {row['antecedents']} ⇒ {row['consequents']}")
        if row["direction"] == "ap":
            aspect = row["antecedents"].split("=",1)[-1]
            pol = row["consequents"].split("=",1)[-1]
            ex = preds[(preds["aspect"]==aspect) & (preds["polarity"]==pol)].head(12)
        else:
            aspect = row["consequents"].split("=",1)[-1]
            ex = preds[(preds["aspect"]==aspect) & (preds["polarity"]=="negative")].head(12)
        if len(ex)==0:
            st.info("No matching examples found in the 50k batch.")
        else:
            for i, r in ex.reset_index(drop=True).iterrows():
                st.markdown(f"- {r['sentence']}")
