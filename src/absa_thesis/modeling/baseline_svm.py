"""Simple baseline: sentence-level sentiment using weak labels from ratings.
Not ABSA, but provides a baseline to beat in speed and accuracy.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from loguru import logger
from ..config import get_settings
from ..utils import ensure_dir

def main():
    cfg = get_settings()
    clean_pq = ensure_dir(cfg.paths["clean_dir"]) / "reviews_clean.parquet"
    if not clean_pq.exists():
        logger.warning("Cleaned data missing. Run ETL first.")
        return

    df = pd.read_parquet(clean_pq).copy()
    # Weak label: rating >=4 positive, <=2 negative, else neutral; drop neutral for baseline
    def weak_label(r):
        if r >= 4:
            return "positive"
        if r <= 2:
            return "negative"
        return None
    df["label"] = df["rating"].map(weak_label)
    df = df.dropna(subset=["label"])

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)

    macro_f1 = f1_score(y_test, preds, average="macro")
    logger.info(f"Baseline macro-F1: {macro_f1:.3f}")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
