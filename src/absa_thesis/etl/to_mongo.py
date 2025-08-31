"""Optional: load cleaned reviews into MongoDB (requires local Mongo instance)."""
import pandas as pd
from pymongo import MongoClient
from loguru import logger
from ..config import get_settings
from ..utils import ensure_dir, env

def main():
    cfg = get_settings()
    clean_pq = ensure_dir(cfg.paths["clean_dir"]) / "reviews_clean.parquet"
    if not clean_pq.exists():
        logger.error("Cleaned parquet not found. Run clean_normalize first.")
        return

    mongo_uri = env("MONGO_URI", "mongodb://localhost:27017")
    db_name = env("MONGO_DB", "absa_thesis")
    client = MongoClient(mongo_uri)
    db = client[db_name]

    df = pd.read_parquet(clean_pq)
    records = df.to_dict(orient="records")
    res = db.reviews.insert_many(records)
    logger.info(f"Inserted {len(res.inserted_ids)} docs into {db_name}.reviews")

if __name__ == "__main__":
    main()
