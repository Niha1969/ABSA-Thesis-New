# tools/peek_labels.py
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

val = pd.read_csv("artifacts/absa/absa_val.csv")
print("val size:", val.shape)
print(val["polarity"].value_counts())

# If you saved predictions later, you can compare; for now this shows class skew.
