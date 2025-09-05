#tiny utility to promote a trained checkpoint ,v2

import os, pathlib, shutil, sys

RUNS_DIR = pathlib.Path("runs/roberta_absa")
DEST = pathlib.Path("artifacts/models/roberta_absa_v2")

def main():
    cks = sorted([p for p in RUNS_DIR.glob("checkpoint-*") if p.is_dir()], key=lambda p: p.name)
    best = cks[-1] if cks else RUNS_DIR
    DEST.mkdir(parents=True, exist_ok=True)
    # copy files (avoid copying trainer_state if you want smaller footprint)
    for item in best.iterdir():
        if item.is_file():
            shutil.copy2(item, DEST / item.name)
        elif item.is_dir():
            shutil.copytree(item, DEST / item.name, dirs_exist_ok=True)
    (DEST / "labels.txt").write_text("0\tnegative\n1\tneutral\n2\tpositive\n", encoding="utf-8")
    print("Promoted:", best, "->", DEST)

if __name__ == "__main__":
    sys.exit(main())
