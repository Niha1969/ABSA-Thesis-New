"""Train RoBERTa for ABSA polarity on annotated data (one label per sentence)."""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from loguru import logger
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..config import get_settings
from ..utils import ensure_dir


# ---------------------------
# Helpers
# ---------------------------

def normalize_polarity(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip().lower()
    mapping = {
        "pos": "positive", "positive": "positive", "+": "positive",
        "neg": "negative", "negative": "negative", "-": "negative",
        "neu": "neutral",  "neutral": "neutral",  "0": "neutral",
    }
    return mapping.get(t, t)

def ensure_sentence_col(df: pd.DataFrame) -> pd.DataFrame:
    if "sentence" in df.columns:
        df["sentence"] = df["sentence"].astype(str)
        return df
    for c in ["text", "review", "content", "body", "comment"]:
        if c in df.columns:
            df = df.rename(columns={c: "sentence"})
            df["sentence"] = df["sentence"].astype(str)
            return df
    raise ValueError("No text column found (expected one of: sentence/text/review/content/body/comment)")

@dataclass
class Encoders:
    polarity: LabelEncoder

def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, Encoders]:
    df = df.copy()
    # Tolerate alternative label column names
    if "polarity" not in df.columns:
        for c in ["label", "sentiment"]:
            if c in df.columns:
                df = df.rename(columns={c: "polarity"})
                break
    if "polarity" not in df.columns:
        raise ValueError("No 'polarity' (or label/sentiment) column found in training data")

    # Normalise polarity and encode
    df["polarity"] = df["polarity"].astype(str).map(normalize_polarity)
    enc_pol = LabelEncoder().fit(df["polarity"])
    df["y_polarity"] = enc_pol.transform(df["polarity"])

    # Ensure text column exists as 'sentence'
    df = ensure_sentence_col(df)

    return df, Encoders(enc_pol)


# ---------------------------
# Trainer with class weights
# ---------------------------

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is None:
            loss_fct = CrossEntropyLoss()
        else:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------
# Main
# ---------------------------

def main():
    cfg = get_settings()
    train_fp = cfg.modeling["absa_train_file"]
    val_fp   = cfg.modeling["absa_val_file"]

    if not (os.path.exists(train_fp) and os.path.exists(val_fp)):
        logger.warning("No labeled ABSA data found. Run merge_annotations after labeling.")
        return

    # Load splits
    train = pd.read_csv(train_fp)
    val   = pd.read_csv(val_fp)

    # Encode labels (polarity is the supervised target)
    train, encs = encode_labels(train)

    # Prepare val: normalise, encode labels, ensure text column
    val = val.copy()
    if "polarity" not in val.columns:
        for c in ["label", "sentiment"]:
            if c in val.columns:
                val = val.rename(columns={c: "polarity"})
                break
    if "polarity" not in val.columns:
        raise ValueError("Validation file missing 'polarity' (or label/sentiment) column")
    val["polarity"] = val["polarity"].astype(str).map(normalize_polarity)
    val = ensure_sentence_col(val)
    val["polarity"] = encs.polarity.transform(val["polarity"])

    num_labels = len(encs.polarity.classes_)

    # Tokenizer/model from base (never from your v1 folder)
    base_model = cfg.modeling.get("base_model", "roberta-base")
    max_len = int(cfg.modeling["max_len"])
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def to_hf(ds_df: pd.DataFrame, use_encoded: bool) -> Dataset:
        texts = ds_df["sentence"].astype(str).tolist()
        toks = tokenizer(texts, max_length=max_len, truncation=True, padding=True)
        labels = (ds_df["y_polarity"] if use_encoded else ds_df["polarity"]).astype(int).tolist()
        toks["labels"] = labels
        return Dataset.from_dict(toks)

    dtrain = to_hf(train, use_encoded=True)
    dval   = to_hf(val,   use_encoded=False)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)

    # Label names in model config (for dashboard)
    id2label = {int(i): lab for i, lab in enumerate(encs.polarity.classes_)}
    label2id = {lab: int(i) for i, lab in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Class weights from train distribution (inverse frequency, normalised)
    counts = train["y_polarity"].value_counts().sort_index().to_numpy()
    weights = (1.0 / (counts + 1e-9))
    weights = weights / weights.sum() * len(counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float)

    out_dir = ensure_dir(Path(cfg.modeling.get("output_dir", "runs/roberta_absa")))
    args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(cfg.modeling.get("lr", 2e-5)),
        per_device_train_batch_size=int(cfg.modeling.get("batch_size", 8)),
        per_device_eval_batch_size=int(cfg.modeling.get("batch_size", 8)),
        num_train_epochs=int(cfg.modeling.get("epochs", 6)),
        weight_decay=0.01,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],  # disable WandB/TensorBoard unless configured
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):  # HF sometimes returns (logits,)
            preds = preds[0]
        pred_ids = np.argmax(preds, axis=-1)
        return {"f1": f1_score(labels, pred_ids, average="macro")}

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=dtrain,
        eval_dataset=dval,
        compute_metrics=compute_metrics,
        class_weights=weights_tensor,
    )

    trainer.train()
    metrics = trainer.evaluate()
    f1 = float(metrics.get("eval_f1", 0.0))
    logger.info(f"Validation macro-F1: {f1:.3f}")

    # Detailed per-class report on the val set
    preds = trainer.predict(dval)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)
    labels = [label2id[k] for k in sorted(label2id, key=label2id.get)]
    report = classification_report(
        y_true, y_pred, target_names=[id2label[i] for i in labels], digits=3, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_dir = ensure_dir(Path("artifacts/training"))
    (report_dir / "val_report.txt").write_text(
        report + "\n" + np.array2string(cm), encoding="utf-8"
    )
    logger.info(f"Wrote per-class report and confusion matrix to {report_dir/'val_report.txt'}")


if __name__ == "__main__":
    main()
