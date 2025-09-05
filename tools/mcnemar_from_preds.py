# tools/mcnemar_from_preds.py
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def load_pair(a_path, b_path, a_col, b_col):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    df = a.merge(b[["sentence", b_col]], on="sentence", how="inner")
    correct_a = (df[a_col] == df["gold"])
    correct_b = (df[b_col] == df["gold"])
    b01 = ((~correct_a) & (correct_b)).sum()  # A wrong, B right
    b10 = ((correct_a) & (~correct_b)).sum()  # A right, B wrong
    return b01, b10

pairs = [
    ("artifacts/training/preds_baseline_test.csv",
     "artifacts/training/preds_roberta_v2_test.csv",
     "baseline_pred","roberta_pred","baseline_vs_v2"),
    ("artifacts/training/preds_roberta_v1_test.csv",
     "artifacts/training/preds_roberta_v2_test.csv",
     "roberta_pred","roberta_pred","v1_vs_v2"),
]
rows=[]
for a,b,acol,bcol,name in pairs:
    b01,b10 = load_pair(a,b,acol,bcol)
    from pandas import DataFrame
    table = DataFrame([[0,b01],[b10,0]], index=["A right","A wrong"], columns=["B right","B wrong"])
    res = mcnemar(table, exact=False, correction=True)
    rows.append({"pair":name, "b01":int(b01), "b10":int(b10), "p_value":float(res.pvalue)})
    print(name, "\n", table, "\n", "p =", res.pvalue, "\n")

pd.DataFrame(rows).to_csv("artifacts/training/mcnemar_summary.csv", index=False)
print("Saved artifacts/training/mcnemar_summary.csv")
