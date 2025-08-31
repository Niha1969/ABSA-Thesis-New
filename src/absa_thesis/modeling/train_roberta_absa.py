"""Train RoBERTa for ABSA on annotated data (aspect + polarity)."""
import os
from dataclasses import dataclass
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from ..config import get_settings
from ..utils import ensure_dir

@dataclass
class Encoders:
    aspect: LabelEncoder
    polarity: LabelEncoder

def encode_labels(df):
    enc_aspect = LabelEncoder().fit(df["aspect"])
    enc_pol = LabelEncoder().fit(df["polarity"])
    df["y_aspect"] = enc_aspect.transform(df["aspect"])
    df["y_polarity"] = enc_pol.transform(df["polarity"])
    return df, Encoders(enc_aspect, enc_pol)

def main():
    cfg = get_settings()
    train_fp  = cfg.modeling["absa_train_file"]
    val_fp    = cfg.modeling["absa_val_file"]
    if not (os.path.exists(train_fp) and os.path.exists(val_fp)):
        logger.warning("No labeled ABSA data found. Run merge_annotations after labeling.")
        return

    train = pd.read_csv(train_fp)
    val = pd.read_csv(val_fp)

    # For simplicity, train a polarity classifier conditioned on sentence only.
    # In a full ABSA, you'd model (sentence, aspect) pairs.
    train, encs = encode_labels(train)
    val["polarity"] = encs.polarity.transform(val["polarity"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.modeling.model_name)  

    def tok(df):
        toks = tokenizer(df["sentence"].tolist(), max_length=cfg.modeling.max_len, truncation=True, padding=True)  # Changed from cfg["modeling"]["max_len"]
        toks["labels"] = df["y_polarity" if "y_polarity" in df else "polarity"].tolist()
        return Dataset.from_dict(toks)

    dtrain = tok(train)
    dval = tok(val)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.modeling["model_name"], num_labels=len(encs.polarity.classes_))  # Changed from cfg["modeling"]["model_name"]

    args = TrainingArguments(
        output_dir="runs/roberta_absa",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.modeling.lr,              # Changed from cfg["modeling"]["lr"]
        per_device_train_batch_size=cfg.modeling["batch_size"],  # Changed from cfg["modeling"]["batch_size"]
        per_device_eval_batch_size=cfg.modeling["batch_size"],   # Changed from cfg["modeling"]["batch_size"]
        num_train_epochs=cfg.modeling["epochs"],      # Changed from cfg["modeling"]["epochs"]
        weight_decay=0.01,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    from sklearn.metrics import f1_score
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        f1 = f1_score(labels, preds, average="macro")
        return {"f1": f1}

    trainer = Trainer(model=model, args=args, train_dataset=dtrain, eval_dataset=dval, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    f1 = metrics.get("eval_f1", 0.0)
    logger.info(f"Validation macro-F1: {f1:.3f}")

if __name__ == "__main__":
    main()