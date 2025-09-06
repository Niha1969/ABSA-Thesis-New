import argparse, pandas as pd, math
from pathlib import Path

def load_pair(a_csv, a_col, b_csv, b_col):
    a = pd.read_csv(a_csv)[["sentence","gold",a_col]].rename(columns={a_col:"A_pred"})
    b = pd.read_csv(b_csv)[["sentence","gold",b_col]].rename(columns={b_col:"B_pred"})
    df = a.merge(b, on=["sentence","gold"], how="inner")
    return df

def mcnemar_counts(df):
    # Following your existing convention: 
    # b01 = A right & B wrong; b10 = A wrong & B right
    a_right = df["A_pred"] == df["gold"]
    b_right = df["B_pred"] == df["gold"]
    b01 = int((a_right & ~b_right).sum())  # A correct, B wrong
    b10 = int((~a_right & b_right).sum())  # A wrong, B correct
    return b01, b10

def exact_binom_p(b01, b10):
    # two-sided exact p-value under H0 p=0.5 for discordant pairs
    n = b01 + b10
    if n == 0: 
        return 1.0
    k = min(b01, b10)
    # cumulative prob of <=k successes in Binom(n,0.5)
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k+1)) * (0.5 ** n)
    p = 2 * tail
    return min(1.0, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-csv", required=True)
    ap.add_argument("--a-col", required=True)
    ap.add_argument("--b-csv", required=True)
    ap.add_argument("--b-col", required=True)
    ap.add_argument("--pair-name", required=True)
    ap.add_argument("--out-csv", default="artifacts/training/mcnemar_summary.csv")
    args = ap.parse_args()

    df = load_pair(args.a_csv, args.a_col, args.b_csv, args.b_col)
    b01, b10 = mcnemar_counts(df)
    p = exact_binom_p(b01, b10)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        res = pd.read_csv(out)
    else:
        res = pd.DataFrame(columns=["pair","b01","b10","p_value"])
    row = {"pair": args.pair-name if False else args.pair_name, "b01": b01, "b10": b10, "p_value": round(p, 6)}
    res = pd.concat([res, pd.DataFrame([row])], ignore_index=True)
    res.to_csv(out, index=False)
    print(f"{args.pair_name}: b01={b01} (A right, B wrong), b10={b10} (A wrong, B right), p={p:.6g}")
    print("Wrote", out)

if __name__ == "__main__":
    main()
