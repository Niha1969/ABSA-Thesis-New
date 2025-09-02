import pandas as pd, numpy as np

raw = pd.read_csv("data/data_raw/amazon_electronics.csv")

def bucket(r):
    if r >= 4: return "pos"
    if r <= 2: return "neg"
    return "neu"

raw["bucket"] = raw["rating"].map(bucket)

TARGET = {"neg": 35, "neu": 30, "pos": 35}
rng = np.random.default_rng(13)

samples = []
for b, k in TARGET.items():
    pool = raw[raw["bucket"] == b]
    take = min(k, len(pool))
    samples.append(pool.sample(take, random_state=13))
balanced = pd.concat(samples).sample(frac=1, random_state=13).reset_index(drop=True)

# Write CSV with only 'text' column for Label Studio
balanced[["text"]].to_csv("artifacts/labelstudio/tasks_balanced.csv", index=False)

print("Wrote", len(balanced), "rows -> artifacts/labelstudio/tasks_balanced.csv")
print(balanced["bucket"].value_counts())
