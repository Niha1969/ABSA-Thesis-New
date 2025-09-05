# re_split_gold.py
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 42

def pick_text(df):
    for c in ["sentence","text","review","content","body","comment"]:
        if c in df.columns: return c
    raise ValueError("No text column found")

# 1) Load cumulative labels & frozen test
lab  = pd.read_csv("artifacts/labelstudio/export_cumulative_2025-09-03.csv")
test = pd.read_csv("artifacts/absa/test_frozen_2025-09-02.csv")

txt_lab  = pick_text(lab)
txt_test = pick_text(test)

# 2) Normalise and exclude frozen-test sentences
lab = lab.rename(columns={txt_lab: "sentence"}).copy()
lab["sentence"] = lab["sentence"].astype(str).str.strip()
test_sents = set(test[txt_test].astype(str).str.strip().tolist())
lab = lab[~lab["sentence"].isin(test_sents)].copy()

# 3) Normalise label column → "polarity"
for c in ["polarity","label","sentiment"]:
    if c in lab.columns:
        lab["polarity"] = lab[c].astype(str).str.lower().str.strip()
        break
valid = {"negative","neutral","positive"}
lab = lab[lab["polarity"].isin(valid)].drop_duplicates(subset=["sentence"]).reset_index(drop=True)

# 4) Stratified 90/10 split
X = lab["sentence"].tolist()
y = lab["polarity"].tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.10,
    random_state=RANDOM_SEED,
    stratify=y,
)

train = pd.DataFrame({"sentence": X_train, "polarity": y_train})
val   = pd.DataFrame({"sentence": X_val,   "polarity": y_val})

# 5) Save
train.to_csv("artifacts/absa/absa_train.csv", index=False)
val.to_csv("artifacts/absa/absa_val.csv", index=False)

# 6) Sanity prints
print("Train/Val sizes:", len(train), len(val))
print("Train class counts:\n", train["polarity"].value_counts())
print("Val class counts:\n",   val["polarity"].value_counts())
