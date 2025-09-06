import streamlit as st, yaml, pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="ABSA Dashboard", layout="wide")

# --- tiny style polish ---
st.markdown("""
<style>
.small { font-size:0.9rem; color:#9aa0a6 }
.card { padding:1rem; border:1px solid rgba(255,255,255,0.08); border-radius:14px; background:rgba(255,255,255,0.02); }
.kpi { font-size:2rem; font-weight:700; margin:0 }
.kpi-label { font-size:0.9rem; color:#9aa0a6; margin-top:-6px }
</style>
""", unsafe_allow_html=True)

st.markdown("## ✨ ABSA Admin")

cfg = yaml.safe_load(open("config.yaml"))
serv = cfg.get("serving", {})

# --- load artifacts if present ---
paths = {
    "rules": Path("artifacts/rules/rules.csv"),
    "pred50k": Path("artifacts/preds/preds_50k.csv"),
    "pred50k_aspect": Path("artifacts/preds/preds_50k_aspect_for_rules.csv"),
    "baselines": Path("artifacts/training/baselines_summary.csv"),
    "mcnemar": Path("artifacts/training/mcnemar_summary.csv"),
    "v2_report": Path("artifacts/training/v2_report.json"),
}

def exists(p): return "✅" if Path(p).exists() else "❌"

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Serving model**")
    st.caption(serv.get("model_path","<not set>"))
    st.caption(f"labels: {serv.get('labels_path','<not set>')}")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    # v2 macro-F1 chip
    try:
        rep = pd.read_json(paths["v2_report"]).to_dict()  # lazy trick; we just want a key check
        rep = __import__("json").loads(paths["v2_report"].read_text())
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi'>"+f"{rep.get('macro_f1', float('nan')):.3f}"+"</div>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>v2 macro-F1 (frozen test)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='kpi'>–</div>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-label'>v2 macro-F1 (missing)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with c3:
    # rules count
    try:
        r = pd.read_csv(paths["rules"])
        rc = len(r)
    except Exception:
        rc = 0
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'>{rc}</div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>mined rules</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Status")
colA,colB = st.columns(2)
with colA:
    st.write(f"{exists(paths['rules'])} Rules CSV — {paths['rules']}")
    st.write(f"{exists(paths['pred50k'])} 50k predictions — {paths['pred50k']}")
    st.write(f"{exists(paths['pred50k_aspect'])} 50k aspect preds — {paths['pred50k_aspect']}")
with colB:
    st.write(f"{exists(paths['baselines'])} Baselines summary — {paths['baselines']}")
    st.write(f"{exists(paths['mcnemar'])} McNemar summary — {paths['mcnemar']}")
    st.write(f"{'✅' if Path(serv.get('model_path','')).exists() else '❌'} Model dir — {serv.get('model_path','<not set>')}")

st.markdown("---")
st.subheader("Quick actions")

# Rebuild rules (runs your in-repo function)
def _rebuild_rules():
    # safe import and call; avoids subprocess
    from tools.make_rules_from_counts import main as build
    build()

if st.button("Rebuild rules from 50k"):
    try:
        _rebuild_rules()
        st.success("Rules rebuilt.")
    except Exception as e:
        st.error(f"Failed to rebuild rules: {e}")

st.caption(datetime.now().strftime("Last refresh: %Y-%m-%d %H:%M:%S"))
st.markdown("---")
st.write("**Go to** → Observability • Insights • Inference (see left sidebar).")
