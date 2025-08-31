"""Convert labeled ABSA data into transactions for association-rule mining."""
import os, pandas as pd
from loguru import logger
from ..config import get_settings
from ..utils import ensure_dir

def main():
    cfg = get_settings()
    absa_fp = cfg.modeling["absa_train_file"]
    if not os.path.exists(absa_fp):
        logger.warning("ABSA train file missing; cannot build transactions.")
        return
    df = pd.read_csv(absa_fp)
    # Each transaction is a set of tokens like Aspect=battery, Polarity=negative
    transactions = df.apply(lambda r: {f"Aspect={r['aspect']}", f"Polarity={r['polarity']}"}, axis=1)
    # Multi-hot encode
    items = sorted({item for t in transactions for item in t})
    rows = []
    for t in transactions:
        row = {i: (1 if i in t else 0) for i in items}
        rows.append(row)
    X = pd.DataFrame(rows)
    out = ensure_dir(cfg.paths["artifacts_dir"]) / "transactions.parquet"
    X.to_parquet(out, index=False)
    logger.info(f"Wrote transactions to {out} with shape {X.shape}")

if __name__ == "__main__":
    main()
