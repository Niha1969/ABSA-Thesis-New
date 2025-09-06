import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Insights", layout="wide")
st.title("Insights: Aspect ↔ Polarity")

RULES_CSV = Path("artifacts/rules/rules.csv")
PREDS = Path("artifacts/preds/preds_50k_aspect_for_rules.csv")

if not RULES_CSV.exists():
    st.warning("No rules.csv found. Run: python tools/make_rules_from_counts.py")
    st.stop()

rules = pd.read_csv(RULES_CSV)
if rules.empty:
    st.info("rules.csv is empty."); st.stop()

with st.sidebar:
    st.header("Filters")
    direction = st.radio("Direction", ["All","Aspect→Positive (AP)","Negative→Aspect (PA)"], index=0)
    min_support = st.slider("Min support", 0.0, 0.3, 0.01, 0.001)
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.5, 0.01)
    min_lift = st.slider("Min lift", 1.0, 2.0, 1.05, 0.01)

df = rules.copy()
if direction != "All":
    df = df[df["direction"] == ("ap" if direction.startswith("Aspect") else "pa")]
df = df[(df["support"] >= min_support) & (df["confidence"] >= min_conf) & (df["lift"] >= min_lift)]

def extract_aspect(row):
    if row["direction"] == "ap":
        return row["antecedents"].split("=",1)[-1]
    return row["consequents"].split("=",1)[-1]

if len(df):
    df["aspect"] = df.apply(extract_aspect, axis=1)
    options = [f"[{r.direction.upper()}] {r.antecedents} ⇒ {r.consequents}  "
               f"(supp={r.support:.3f}, conf={r.confidence:.3f}, lift={r.lift:.2f})"
               for _, r in df.iterrows()]
    st.subheader(f"Rules ({len(df)})")
    st.dataframe(
        df[["direction","antecedents","consequents","aspect","support","confidence","lift"]]
          .sort_values(["direction","confidence","support","lift"], ascending=[True, False, False, False]),
        use_container_width=True, hide_index=True
    )
else:
    options, df = [], df

st.markdown("---")
st.subheader("Examples")

if not PREDS.exists():
    st.info("No example corpus at artifacts/preds/preds_50k_aspect_for_rules.csv.")
else:
    preds = pd.read_csv(PREDS)
    preds["polarity"] = preds["polarity"].astype(str)

    if len(df) == 0:
        st.info("No rules after filtering.")
    else:
        sel = st.selectbox("Choose a rule", options, index=0)
        ridx = options.index(sel)
        r = df.reset_index(drop=True).iloc[ridx]
        st.write(f"**Rule:** {r['antecedents']} ⇒ {r['consequents']}")

        if r["direction"] == "ap":
            aspect = r["antecedents"].split("=",1)[-1]
            pol = r["consequents"].split("=",1)[-1]
            ex = preds[(preds["aspect"]==aspect) & (preds["polarity"]==pol)].head(12)
        else:  # pa: negative → Aspect
            aspect = r["consequents"].split("=",1)[-1]
            ex = preds[(preds["aspect"]==aspect) & (preds["polarity"]=="negative")].head(12)

        if len(ex)==0:
            st.info("No matching examples in 50k batch.")
        else:
            for i, row in ex.reset_index(drop=True).iterrows():
                st.markdown(f"- {row['sentence']}")
