# src/absa_thesis/rules/mine_rules.py
"""Mine Aspect↔Polarity rules into one table with direction labels (ap / pa)."""

import json
from pathlib import Path
import argparse
import pandas as pd
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules

from ..config import get_settings
from ..utils import ensure_dir

def _itemset_to_str(fs):
    if isinstance(fs, (set, frozenset, list, tuple)):
        return ", ".join(sorted(map(str, fs)))
    return str(fs)

def _has_prefix(fs, prefix: str) -> bool:
    it = fs if isinstance(fs, (set, frozenset, list, tuple)) else [fs]
    return any(str(x).startswith(prefix) for x in it)

def _is_singleton(fs) -> bool:
    try:
        return len(fs) == 1
    except TypeError:
        return True

def _apriori(df_bool, min_support, max_len=2):
    return apriori(df_bool, min_support=min_support, use_colnames=True, max_len=max_len)

def _assoc(freq, min_conf):
    if freq.empty:
        return pd.DataFrame()
    return association_rules(freq, metric="confidence", min_threshold=min_conf)

def _filter_cross_type_singletons(rules, antecedent_prefix, consequent_prefix):
    if rules.empty:
        return rules
    mask = (
        rules["antecedents"].apply(lambda fs: _has_prefix(fs, antecedent_prefix)) &
        rules["consequents"].apply(lambda fs: _has_prefix(fs, consequent_prefix)) &
        rules["antecedents"].apply(_is_singleton) &
        rules["consequents"].apply(_is_singleton)
    )
    return rules[mask]

def _format_rules(rules):
    if rules.empty:
        return rules
    rules = rules.sort_values(
        ["confidence", "support", "lift"], ascending=[False, False, False]
    ).reset_index(drop=True)
    rules["antecedents"] = rules["antecedents"].map(_itemset_to_str)
    rules["consequents"] = rules["consequents"].map(_itemset_to_str)
    cols = ["direction","antecedents","consequents","support","confidence","lift","leverage","conviction"]
    return rules[ [c for c in cols if c in rules.columns] ]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx-path", default="artifacts/transactions.parquet")
    # Family A (Aspect -> Positive)
    ap.add_argument("--ap-support", type=float, default=None)
    ap.add_argument("--ap-conf", type=float, default=None)
    ap.add_argument("--ap-lift", type=float, default=None)
    # Family B (Negative -> Aspect)
    ap.add_argument("--pa-support", type=float, default=None)
    ap.add_argument("--pa-conf", type=float, default=None)
    ap.add_argument("--pa-lift", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=2)
    return ap.parse_args()

def main():
    cfg = get_settings()
    args = parse_args()

    tx_path = Path(args.tx_path)
    if not tx_path.exists():
        logger.error(f"Transactions not found at {tx_path}. Run build_transactions first.")
        return

    df = pd.read_parquet(tx_path)
    for c in df.columns:
        df[c] = df[c].astype(bool)

    # ---- Thresholds (defaults chosen from your printed supports) ----
    # You can still keep global config rules for baseline numbers,
    # but here we set per-family defaults that make sense for your data.
    ap_support = args.ap_support if args.ap_support is not None else max(0.01, float(cfg.rules.get("min_support", 0.02)))
    ap_conf    = args.ap_conf    if args.ap_conf    is not None else max(0.60, float(cfg.rules.get("min_confidence", 0.6)))
    ap_lift    = args.ap_lift    if args.ap_lift    is not None else max(1.05, float(cfg.rules.get("min_lift", 1.0)))
    pa_support = args.pa_support if args.pa_support is not None else 0.005   # lower; negatives are rarer per aspect
    pa_conf    = args.pa_conf    if args.pa_conf    is not None else 0.18    # 0.15–0.30 works; you printed ~0.21 for design→neg
    pa_lift    = args.pa_lift    if args.pa_lift    is not None else 1.20    # require over-indexing

    max_len = args.max_len

    # ---- Mine Family A: Aspect -> Polarity=positive ----
    logger.info(f"[AP] support>={ap_support}, conf>={ap_conf}, lift>={ap_lift}, max_len={max_len}")
    freq = _apriori(df, ap_support, max_len=max_len)
    rules_ap = _assoc(freq, ap_conf)
    rules_ap = _filter_cross_type_singletons(rules_ap, "Aspect=", "Polarity=")
    if not rules_ap.empty and ap_lift > 1.0:
        rules_ap = rules_ap[rules_ap["lift"] >= ap_lift]
    if not rules_ap.empty:
        rules_ap = rules_ap.copy()
        rules_ap["direction"] = "ap"

    # ---- Mine Family B: Polarity=negative -> Aspect ----
    logger.info(f"[PA] support>={pa_support}, conf>={pa_conf}, lift>={pa_lift}, max_len={max_len}")
    freq_b = _apriori(df, pa_support, max_len=max_len)
    rules_pa = _assoc(freq_b, pa_conf)
    rules_pa = _filter_cross_type_singletons(rules_pa, "Polarity=", "Aspect=")
    # keep only antecedent = Polarity=negative
    if not rules_pa.empty:
        rules_pa = rules_pa[rules_pa["antecedents"].apply(lambda fs: _has_prefix(fs, "Polarity=negative"))]
    if not rules_pa.empty and pa_lift > 1.0:
        rules_pa = rules_pa[rules_pa["lift"] >= pa_lift]
    if not rules_pa.empty:
        rules_pa = rules_pa.copy()
        rules_pa["direction"] = "pa"

    # ---- Combine, format & write ----
    if rules_ap is None or rules_ap.empty:
        rules_ap = pd.DataFrame(columns=["direction","antecedents","consequents","support","confidence","lift","leverage","conviction"])
    if rules_pa is None or rules_pa.empty:
        rules_pa = pd.DataFrame(columns=["direction","antecedents","consequents","support","confidence","lift","leverage","conviction"])

    rules = pd.concat([rules_ap, rules_pa], ignore_index=True)
    rules = _format_rules(rules)

    out_dir = ensure_dir(Path("artifacts/rules"))
    out_json = out_dir / "rules.json"
    out_csv  = out_dir / "rules.csv"

    rules_list = [] if rules.empty else rules.to_dict(orient="records")
    if rules_list:
        rules.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rules_list, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(rules_list)} rules to {out_json} (and CSV if not empty)")

if __name__ == "__main__":
    main()
