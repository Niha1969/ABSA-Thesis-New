#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, uuid, pathlib, sys

def detect_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    raise ValueError(f"Missing required column; looked for any of {candidates} in {list(df.columns)}")

def load_pool(paths):
    dfs=[]
    for p in paths:
        df=pd.read_csv(p)
        df.columns=[c.strip() for c in df.columns]
        dfs.append(df)
    pool=pd.concat(dfs, ignore_index=True)
    return pool

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="CSV files with your *gold* labels (e.g., artifacts/absa/absa_train.csv artifacts/absa/absa_val.csv)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="CSV files to exclude by sentence (e.g., artifacts/absa/test_frozen_2025-09-02.csv)")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--p_pos", type=float, default=0.34)
    ap.add_argument("--p_neu", type=float, default=0.33)
    ap.add_argument("--p_neg", type=float, default=0.33)
    ap.add_argument("--out", required=True, help="Output CSV for Label Studio import")
    args=ap.parse_args()

    pool=load_pool(args.inputs)

    # Robust column detection
    text_col = detect_col(pool, ["sentence","text","review"])
    pol_col  = detect_col(pool, ["polarity","label","polarity_label"])
    asp_col  = next((c for c in ["aspect","aspect_category","aspect_label"] if c in pool.columns), None)
    id_col   = next((c for c in ["source_id","id","example_id"] if c in pool.columns), None)

    # Normalise labels
    pool[pol_col]=pool[pol_col].str.strip().str.lower().map({
        "pos":"positive","positive":"positive",
        "neu":"neutral","neutral":"neutral",
        "neg":"negative","negative":"negative"
    })

    # Basic clean
    pool[text_col]=pool[text_col].astype(str).str.strip()
    pool=pool.dropna(subset=[text_col, pol_col])
    pool=pool.drop_duplicates(subset=[text_col])

    # Exclude any sentences that appear in provided exclude files (e.g., frozen test)
    if args.exclude:
        ex_text=set()
        for p in args.exclude:
            try:
                ex=pd.read_csv(p)
                ex_text.update(ex.get(text_col, ex.iloc[:,0]).astype(str).str.strip().tolist())
            except Exception:
                pass
        pool=pool[~pool[text_col].isin(ex_text)]

    # Stratified sampling (approximate counts)
    targets={
        "positive": int(round(args.n*args.p_pos)),
        "neutral":  int(round(args.n*args.p_neu)),
        "negative": args.n - int(round(args.n*args.p_pos)) - int(round(args.n*args.p_neu)),
    }

    rng=np.random.default_rng(7)
    parts=[]
    for cls,k in targets.items():
        dfc=pool[pool[pol_col]==cls]
        if dfc.empty:
            print(f"[WARN] No rows for class '{cls}'. Reducing target to 0.", file=sys.stderr); continue
        take=min(k, len(dfc))
        parts.append(dfc.sample(n=take, random_state=42))
    sub=pd.concat(parts, ignore_index=True)

    # Shuffle
    sub=sub.sample(frac=1.0, random_state=123).reset_index(drop=True)

    # Carry original labels/ids as meta; LS wants a 'text' column
    out=pd.DataFrame({
        "sample_id": [uuid.uuid4().hex for _ in range(len(sub))],
        "text": sub[text_col],
        "polarity_original": sub[pol_col],
        **({"aspect_original": sub[asp_col]} if asp_col else {})
    })
    if id_col: out["source_id"]=sub[id_col]

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows → {args.out}")
    print("Class counts:", out["polarity_original"].value_counts(dropna=False).to_dict())

if __name__=="__main__":
    main()
