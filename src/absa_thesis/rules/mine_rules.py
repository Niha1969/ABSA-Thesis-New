"""Mine association rules and filter by support/confidence thresholds."""
import pandas as pd
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules
from ..config import get_settings
from ..utils import ensure_dir

def main():
    cfg = get_settings()
    trans_fp = ensure_dir(cfg.paths["artifacts_dir"]) / "transactions.parquet"
    if not trans_fp.exists():
        logger.warning("Transactions not found. Run build_transactions first.")
        return
    X = pd.read_parquet(trans_fp)

    freq = apriori(X, min_support=cfg.rules["min_support"], use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=cfg.rules["min_confidence"])
    rules = rules.sort_values(["confidence","support"], ascending=False)

    out_json = ensure_dir(cfg.paths["rules_dir"]) / "rules.json"
    rules_out = rules.to_dict(orient="records")
    out_json.write_text(json.dumps(rules_out, indent=2), encoding="utf-8")

    logger.info(f"Mined {len(rules)} rules. Saved to {out_json}")

if __name__ == "__main__":
    import json
    main()
