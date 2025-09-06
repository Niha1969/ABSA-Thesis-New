import argparse, json, re
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["negative","neutral","positive"]

def latest_checkpoint(run_dir: Path) -> Path:
    cks = sorted(
        [p for p in run_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(re.findall(r"\d+", p.name)[-1]) if re.findall(r"\d+", p.name) else -1
    )
    return cks[-1] if cks else run_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Training run dir with checkpoint-* (or a saved model dir)")
    ap.add_argument("--out", required=True, help="Export dir, e.g. artifacts/models/roberta_absa_v2")
    ap.add_argument("--base-model", default=None, help="Tokenizer source (e.g. roberta-base). If omitted, read from config.yaml->modeling.base_model or fall back to roberta-base.")
    args = ap.parse_args()

    run = Path(args.run)
    src = latest_checkpoint(run)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Load model weights from checkpoint/src
    model = AutoModelForSequenceClassification.from_pretrained(str(src))

    # Load tokenizer: try src first (rarely exists), else base model name
    base_model = args.base_model
    if base_model is None:
        # try to read config.yaml->modeling.base_model
        try:
            import yaml
            base_model = yaml.safe_load(open("config.yaml"))["modeling"]["base_model"]
        except Exception:
            base_model = "roberta-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(src))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Save both
    tokenizer.save_pretrained(str(out))
    model.save_pretrained(str(out))
    (out / "labels.txt").write_text("\n".join(LABELS), encoding="utf-8")

    # Ensure id2label/label2id in config
    cfgp = out / "config.json"
    try:
        cfg = json.loads(cfgp.read_text())
    except Exception:
        cfg = {}
    label2id = {l:i for i,l in enumerate(LABELS)}
    id2label = {str(i):l for l,i in label2id.items()}
    cfg.update({"label2id": label2id, "id2label": id2label, "num_labels": len(LABELS)})
    cfgp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"Exported model from {src} → {out}")
    print("Files:", [p.name for p in out.iterdir()])

if __name__ == "__main__":
    main()
