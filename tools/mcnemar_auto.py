# tools/mcnemar_auto.py  (overwrite)
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

PAIR_SPECS = [
    ("artifacts/training/preds_baseline_test.csv",
     "artifacts/training/preds_roberta_v2_test.csv",
     "baseline_vs_v2"),
    ("artifacts/training/preds_roberta_v1_test.csv",
     "artifacts/training/preds_roberta_v2_test.csv",
     "v1_vs_v2"),
]

SAFE_GOLD = "gold"
SAFE_SENT = "sentence"

def choose_pred_col(df: pd.DataFrame) -> str:
    # pick first non sentence/gold col that contains "pred", else first other col
    candidates = [c for c in df.columns if c.lower() not in (SAFE_SENT, SAFE_GOLD)]
    for c in candidates:
        if "pred" in c.lower():
            return c
    if not candidates:
        raise ValueError("No prediction column found")
    return candidates[0]

def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if SAFE_SENT not in df.columns: raise ValueError(f"{path} missing '{SAFE_SENT}'")
    if SAFE_GOLD not in df.columns: raise ValueError(f"{path} missing '{SAFE_GOLD}'")
    df[SAFE_GOLD] = df[SAFE_GOLD].astype(str).str.lower().str.strip()
    return df

def main():
    rows = []
    for a_path, b_path, name in PAIR_SPECS:
        a = load_preds(a_path)
        b = load_preds(b_path)
        a_pred = choose_pred_col(a)
        b_pred = choose_pred_col(b)

        print(f"[{name}] A uses '{a_pred}' | B uses '{b_pred}'")
        print(" A cols:", list(a.columns))
        print(" B cols:", list(b.columns))

        # If pred col names clash, pandas will suffix both; detect final names
        clash = (a_pred == b_pred)
        merged = a[[SAFE_SENT, SAFE_GOLD, a_pred]].merge(
            b[[SAFE_SENT, b_pred]],
            on=SAFE_SENT,
            how="inner",
            suffixes=("_a","_b") if clash else ("","_b")
        )

        # Resolve final column names post-merge
        a_col_final = f"{a_pred}_a" if clash else a_pred
        b_col_final = f"{b_pred}_b"  # B always gets _b when clash or when we forced suffixes

        # If no clash, pandas won’t append _b to B’s col; handle that:
        if not clash and b_col_final not in merged.columns:
            b_col_final = b_pred

        print(" MERGED cols:", list(merged.columns))
        print(f" Using A='{a_col_final}'  B='{b_col_final}'")

        correct_a = (merged[a_col_final].astype(str).str.lower() == merged[SAFE_GOLD])
        correct_b = (merged[b_col_final].astype(str).str.lower() == merged[SAFE_GOLD])

        b01 = int((~correct_a &  correct_b).sum())  # A wrong, B right
        b10 = int(( correct_a & ~correct_b).sum())  # A right, B wrong

        table = pd.DataFrame([[0, b01],[b10, 0]],
                             index=["A right","A wrong"],
                             columns=["B right","B wrong"])
        res = mcnemar(table, exact=False, correction=True)

        print(f"[{name}]")
        print(table)
        print(" p-value:", res.pvalue)
        print("-"*60)

        rows.append({"pair": name, "b01": b01, "b10": b10, "p_value": float(res.pvalue)})

    out = pd.DataFrame(rows)
    out.to_csv("artifacts/training/mcnemar_summary.csv", index=False)
    print("Saved artifacts/training/mcnemar_summary.csv")
    print(out)

if __name__ == "__main__":
    main()
