"""Mine association rules and write JSON/CSV."""
import json
from pathlib import Path
import pandas as pd
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules

from ..config import get_settings
from ..utils import ensure_dir

def _itemset_to_str(fs):
    # fs could be frozenset({'Aspect=battery','Polarity=negative'})
    if isinstance(fs, (set, frozenset, list, tuple)):
        return ", ".join(sorted(map(str, fs)))
    return str(fs)

def main():
    cfg = get_settings()
    tx_path = Path("artifacts/transactions.parquet")
    if not tx_path.exists():
        logger.error("Transactions not found. Run build_transactions first.")
        return

    df = pd.read_parquet(tx_path)
    # ensure boolean dataframe for mlxtend
    for c in df.columns:
        df[c] = df[c].astype(bool)

    min_support = float(cfg.rules["min_support"])
    min_conf = float(cfg.rules["min_confidence"])
    min_lift = float(cfg.rules.get("min_lift", 1.0))

    logger.info(f"Apriori: min_support={min_support}, min_confidence={min_conf}, min_lift={min_lift}")
    freq = apriori(df, min_support=min_support, use_colnames=True)
    if freq.empty:
        logger.warning("No frequent itemsets found at this support.")
        rules = pd.DataFrame()
    else:
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        if not rules.empty and min_lift > 1.0:
            rules = rules[rules["lift"] >= min_lift]
        # prettify columns
        if not rules.empty:
            rules = rules.sort_values(["confidence", "support", "lift"], ascending=[False, False, False]).reset_index(drop=True)
            rules["antecedents"] = rules["antecedents"].map(_itemset_to_str)
            rules["consequents"] = rules["consequents"].map(_itemset_to_str)

    out_dir = ensure_dir(Path("artifacts/rules"))
    out_json = out_dir / "rules.json"
    out_csv = out_dir / "rules.csv"

    if rules.empty:
        rules_list = []
    else:
        cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
        rules_list = rules[cols].to_dict(orient="records")
        rules[cols].to_csv(out_csv, index=False)

    out_json.write_text(json.dumps(rules_list, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(rules_list)} rules to {out_json} (and CSV if not empty)")

if __name__ == "__main__":
    main()
