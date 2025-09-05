# tools/baseline_tfidf.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def pick_text(df):
    for c in ["sentence","text","review","content","body","comment"]:
        if c in df.columns: return c
    raise ValueError("No text column")

test = pd.read_csv("artifacts/absa/test_frozen_2025-09-02.csv")
train = pd.read_csv("artifacts/absa/absa_train.csv")
val   = pd.read_csv("artifacts/absa/absa_val.csv")

for d in (train,val,test):
    if "polarity" not in d.columns:
        for c in ["label","sentiment"]:
            if c in d.columns: d.rename(columns={c:"polarity"}, inplace=True)

xt, yt = pick_text(train), "polarity"
xv, yv = pick_text(val),   "polarity"
xe, ye = pick_text(test),  "polarity"

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000)
Xtr = vec.fit_transform(train[xt].astype(str))
Xva = vec.transform(val[xv].astype(str))
Xte = vec.transform(test[xe].astype(str))

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(Xtr, train[yt].str.lower())

from sklearn.metrics import classification_report
pred_val = clf.predict(Xva); pred_test = clf.predict(Xte)
print("BASELINE macro-F1 (val): ", f1_score(val[yv].str.lower(), pred_val, average="macro"))
print("BASELINE macro-F1 (test):", f1_score(test[ye].str.lower(), pred_test, average="macro"))

pd.DataFrame({"sentence": test[xe].astype(str),
              "gold": test[ye].str.lower(),
              "baseline_pred": pred_test}).to_csv(
    "artifacts/training/preds_baseline_test.csv", index=False)
