"""
Derive Aspect↔Polarity rules directly from transactions.parquet using counts.
Outputs artifacts/rules/rules.csv and rules.json with direction column (ap|pa).
- Family AP: Aspect -> Polarity=positive (drivers of satisfaction)
- Family PA: Polarity=negative -> Aspect (over-indexed pain points)
"""
from pathlib import Path
import json
import pandas as pd

TX = Path("artifacts/transactions.parquet")
OUT_DIR = Path("artifacts/rules")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV = OUT_DIR / "rules.csv"
JSN = OUT_DIR / "rules.json"

# ---- knobs you can tweak safely ----
AP_MIN_SUP = 0.010   # >= ~216 rows
AP_MIN_LIFT = 1.05   # small but real uplift
AP_TOP_K   = 10

PA_MIN_SUP = 0.005   # >= ~108 rows
PA_MIN_LIFT = 1.12   # stronger over-indexing for pain points
PA_TOP_K   = 10
# ------------------------------------

def main():
    if not TX.exists():
        raise SystemExit(f"Missing {TX}; run build_transactions first.")
    df = pd.read_parquet(TX).astype(bool)
    n = len(df)

    aspects = [c for c in df.columns if c.startswith("Aspect=")]
    pols    = [c for c in df.columns if c.startswith("Polarity=")]

    # base rates
    p_pol = {p: df[p].mean() for p in pols}
    p_as  = {a: df[a].mean() for a in aspects}

    rows = []

    # Family AP: Aspect -> positive
    pos = "Polarity=positive"
    for a in aspects:
        supp = (df[a] & df[pos]).mean()
        if supp < AP_MIN_SUP: continue
        conf = supp / (df[a].mean() or 1e-9)
        lift = conf / (p_pol[pos] or 1e-9)
        if lift < AP_MIN_LIFT: continue
        rows.append({
            "direction": "ap",
            "antecedents": a,
            "consequents": pos,
            "support": round(supp, 6),
            "confidence": round(conf, 6),
            "lift": round(lift, 6),
        })

    # Family PA: negative -> Aspect
    neg = "Polarity=negative"
    for a in aspects:
        supp = (df[a] & df[neg]).mean()
        if supp < PA_MIN_SUP: continue
        conf = supp / (df[neg].mean() or 1e-9)       # P(aspect | negative)
        lift = conf / (p_as[a] or 1e-9)              # over-indexing relative to aspect base rate
        if lift < PA_MIN_LIFT: continue
        rows.append({
            "direction": "pa",
            "antecedents": neg,
            "consequents": a,
            "support": round(supp, 6),
            "confidence": round(conf, 6),
            "lift": round(lift, 6),
        })

    if not rows:
        Path(JSN).write_text("[]", encoding="utf-8")
        CSV.write_text("", encoding="utf-8")
        print("No rules after thresholds.")
        return

    rules = pd.DataFrame(rows)

    # Keep top-K per family to ensure a healthy but not spammy table
    out = []
    for fam, topk in [("ap", AP_TOP_K), ("pa", PA_TOP_K)]:
        sub = rules[rules["direction"] == fam].sort_values(
            ["confidence", "support", "lift"], ascending=[False, False, False]
        ).head(topk)
        out.append(sub)
    rules = pd.concat(out, ignore_index=True)

    # add leverage/conviction (optional; simple, not exact MLxtend)
    rules["leverage"] = rules["support"] - rules["confidence"] * (
        rules.apply(lambda r: p_pol["Polarity=positive"] if r["direction"]=="ap" else p_as[r["consequents"]], axis=1)
    )
    # save
    rules.to_csv(CSV, index=False)
    Path(JSN).write_text(json.dumps(rules.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"Wrote {len(rules)} rules -> {CSV}")
    print(rules)

if __name__ == "__main__":
    main()
