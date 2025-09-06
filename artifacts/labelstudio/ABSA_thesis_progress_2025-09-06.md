# ABSA Thesis — Daily Progress Summary
**Date:** 2025-09-06 (Europe/London)  
**Author:** Nihal  
**Project:** Aspect-Based Sentiment Analysis (ABSA) Dashboard & Evaluation

---

## 1) Executive Summary
- Exported and registered **v2** model to `artifacts/models/roberta_absa_v2` and wired it into the app.
- Evaluated v2 on the **frozen test** (90 rows; file timestamp: **2025-09-02**).
- Added **lexicon baselines** (VADER via `vaderSentiment`, TextBlob) and an Observability table that compares macro-F1 and per-class F1 across models.
- Ran **paired significance tests (McNemar)**: v2 vs VADER/TextBlob (significant).
- Generated **confusion matrices** for v2, VADER, TextBlob; fixed Observability to auto-discover and render them.
- Implemented a robust **single-sentence inference** UI (multi-aspect detection) + **batch CSV** inference.
- Rebuilt **Insights** (click a rule → show examples) and **Observability** (class distribution, uncertainty histogram, baselines table, McNemar, CM gallery).
- Stabilized **rule mining** (AP and PA families) and provided a deterministic **counts-based** rules shortcut.
- Fixed multiple engineering issues (config/schema/paths, NLTK SSL, Streamlit interval bins, tokenizer-at-checkpoint).

**Today’s headline metric (frozen test, 90 rows):**
- **v2 macro-F1 = 0.795**  
  Per-class F1: **neg 0.769 · neu 0.706 · pos 0.911**

**Baselines on the same frozen test:**
- VADER macro-F1 **0.577** (neg 0.486 · neu 0.465 · pos 0.780)
- TextBlob macro-F1 **0.447** (neg 0.378 · neu 0.255 · pos 0.708)

**McNemar (paired on frozen test):**
- v2 vs VADER: **b01=19, b10=3, p≈0.0009**
- v2 vs TextBlob: **b01=29, b10=2, p≈0.000…** (rendered as 0.0 → p < 0.001 in app)

---

## 2) Data & Splits
- Frozen test (locked): `artifacts/absa/test_frozen_2025-09-02.csv` → normalized to `artifacts/training/frozen_test_for_eval.csv` with **columns: `sentence,gold`**.
- 50k unlabeled batch (inferred with v2) for insights/rules:
  - `artifacts/preds/preds_50k.csv` (polarity + probabilities)
  - `artifacts/preds/preds_50k_aspect.csv` and `..._for_rules.csv` (after aspect tagging & filtering; ~21,602 rows for rules)

**Uncertainty (on 50k):** fraction with max prob < 0.6 ≈ **0.170** (observability chart).

---

## 3) Modeling & Serving
- Base model: **roberta-base** (from `config.yaml:modeling.base_model`).
- Exported v2 checkpoint using fallback tokenizer (base model) + fine-tuned weights.
- `config.yaml` updated with:
  ```yaml
  serving:
    model_path: "artifacts/models/roberta_absa_v2"
    labels_path: "artifacts/models/roberta_absa_v2/labels.txt"
  paths:
    test_frozen_file: artifacts/absa/test_frozen_2025-09-02.csv
  ```

---

## 4) Inference & Insights
- **Inference page** now supports:
  - Single sentence → polarity + probabilities + **multi-aspect hits** (regex lexicon).
  - Batch CSV upload with `'sentence'` column → downloadable predictions.
- **Insights page**:
  - Clickable rules (no row index input), filters (support/confidence/lift, direction AP/PA), and example sentences from 50k predictions.

**Rules (example set from counts-based shortcut):**
```
direction,antecedents,consequents,support,confidence,lift,leverage
ap,Aspect=price,Polarity=positive,0.2446,0.8483,1.1399,-0.3867
ap,Aspect=usability,Polarity=positive,0.0970,0.8161,1.0967,-0.5103
pa,Polarity=negative,Aspect=design,0.0457,0.2560,1.1894,-0.0094
pa,Polarity=negative,Aspect=battery,0.0436,0.2438,1.2623,-0.0035
pa,Polarity=negative,Aspect=screen,0.0157,0.0878,1.2626,0.0096
pa,Polarity=negative,Aspect=support,0.0098,0.0547,1.6612,0.0080
pa,Polarity=negative,Aspect=performance,0.0089,0.0500,1.3905,0.0071
```
Interpretation:
- **AP (Aspect→Positive)**: satisfaction drivers (e.g., **price**, **usability** skew positive).
- **PA (Negative→Aspect)**: over-indexed pain points when **negative**, especially **support** and **screen**.

---

## 5) Evaluation
### 5.1 Metrics (frozen test)
- v2 macro-F1 **0.795** (neg **0.769**, neu **0.706**, pos **0.911**).
- Baselines in Observability table: VADER **0.577**, TextBlob **0.447**.

### 5.2 McNemar (paired significance)
- `v2_vs_vader`: b01=19 (v2 right, VADER wrong), b10=3 (v2 wrong, VADER right), **p≈0.0009** → significant.
- `v2_vs_textblob`: b01=29, b10=2, **p < 0.001** → significant.
- (Legacy rows `baseline_vs_v2`, `v1_vs_v2` present from earlier runs; kept for completeness.)

### 5.3 Confusion matrices
- Generated PNGs in `artifacts/training/`: `cm_v2.png`, `cm_vader.png`, `cm_textblob.png`.
- Observability now auto-discovers `cm_*.png` and renders a gallery.

---

## 6) Engineering Issues & Fixes
- **Config KeyError (`serving`)**: added `serving:` section to `config.yaml`.
- **`Settings.__init__` unexpected keyword**: dataclass updated to include `serving` or loader filtered keys.
- **`predict_batch.py` pathing**: aligned to `config.yaml` and relative paths; produced `preds_50k.csv` OK.
- **`tag_aspect_keywords.py` IndexError**: added CLI args (input/output) & usage docs.
- **Rule mining too few rules**: introduced dual families AP/PA with distinct thresholds; added **counts-based** fallback (`tools/make_rules_from_counts.py`) giving 7 stable rules from 50k counts.
- **Streamlit interval bins error**: converted Pandas Interval bins to **string labels** before `st.bar_chart`.
- **NLTK VADER SSL/cert failure**: replaced with **`vaderSentiment`** package (bundled lexicon).
- **Tokenizer missing in checkpoint**: exporter now loads tokenizer from base model and weights from checkpoint.
- **Confusion matrices not rendering**: Observability switched from single hardcoded filename to an **auto-gallery** scanning `cm_*.png`/`*confusion*matrix*.png` across candidate dirs.

---

## 7) Files & Scripts Added/Modified Today
**New tools/**
- `tools/prepare_test_csv.py` — normalize frozen test to `sentence,gold`.
- `tools/eval_baselines.py` — VADER/TextBlob eval + summary CSV (uses `vaderSentiment`).
- `tools/export_model.py` — export fine-tuned checkpoint to `artifacts/models/...` with labels.
- `tools/eval_model.py` — evaluate any exported model on frozen test; writes preds + report.
- `tools/make_confusion_matrix.py` — generate `cm_<tag>.png` and CSV.
- `tools/mcnemar_compare.py` — exact-binomial McNemar; appends to `mcnemar_summary.csv`.

**Dashboard/**
- `streamlit_app/pages/01_Observability.py` — baselines table, McNemar table, uncertainty fix, **CM auto-gallery**.
- `streamlit_app/pages/02_Insights.py` — clickable rules + examples.
- `streamlit_app/pages/03_Inference.py` — single sentence + CSV; multi-aspect detection.
- `streamlit_app/Home.py` — polished Admin-style landing (cards, KPIs, health checks).

**Artifacts produced/**
- `artifacts/models/roberta_absa_v2/` (exported model + labels).
- `artifacts/training/frozen_test_for_eval.csv` (normalized test).
- `artifacts/training/v2_pred.csv`, `v2_report.json`.
- `artifacts/training/vader_preds.csv`, `textblob_preds.csv`, `baselines_summary.csv`.
- `artifacts/training/cm_v2.png`, `cm_vader.png`, `cm_textblob.png`.
- `artifacts/training/mcnemar_summary.csv` (rows incl. `v2_vs_vader`, `v2_vs_textblob`).
- `artifacts/preds/preds_50k*.csv` and `artifacts/rules/rules.csv`/`.json`.

---

## 8) Results & Analysis — How to Present (tomorrow)
Include in the dissertation:
1. **Table: Overall performance** (macro-F1 + per-class for v2, VADER, TextBlob).
2. **Figure: Confusion matrices** (v2 vs baselines) — discuss major error modes.
3. **Statistical significance** (McNemar) — report b01/b10 and p-values; interpret.
4. **Uncertainty analysis** (share of low-confidence predictions; threshold 0.6).
5. **Insight rules** (top AP/PA with support/confidence/lift) + 2–3 example sentences per rule.
6. **Scalability**: 50k batch inference and mining results; class imbalance note (≈74% positive in 50k).

---

## 9) Critical Evaluation, Conclusions & Recommendations — Outline
- **Strengths**: solid evaluation hygiene (frozen test; macro-F1; McNemar), clear interpretability (AP/PA rules with examples), production-friendly dashboard + serving.
- **Limitations**:
  - Aspects for 50k come from heuristics (regex lexicon), not human labels → insight layer is exploratory.
  - Class imbalance constrains lift; neutral remains hardest class (see per-class F1 and CM).
  - No calibrated probabilities; threshold-dependent uncertainty.
- **Threats to validity**: labeling noise, domain shift between labeled set and Amazon electronics 50k, sample size of frozen test (n=90).
- **Recommendations**:
  - Train a supervised **aspect classifier**; allow **multi-aspect per sentence**.
  - Add **probability calibration** (temperature scaling) and display calibrated uncertainty.
  - Expand frozen test size; stratify by aspect & polarity.
  - Add **model confidence filtering** to Insights; contrast slices (e.g., battery vs design).
  - Include a lightweight **logistic regression** baseline over TF‑IDF for completeness (optional).

---

## 10) Tomorrow’s Checklist (Actionable)
**Results & Analysis (20 marks)**
- [ ] Export table from `baselines_summary.csv` and insert into the chapter.
- [ ] Insert CM images and write 2–3 paragraphs on error modes.
- [ ] Write McNemar subsection (method + interpretation for v2 vs VADER/TextBlob).
- [ ] Add uncertainty paragraph (0.170 fraction with max‑prob < 0.6).

**Critical Evaluation, Conclusions & Recommendations (20 marks)**
- [ ] Draft limitations & threats to validity (see §9 bullets).
- [ ] Summarize **answer to research question**: v2 significantly outperforms lexicon baselines with interpretable rules at scale.
- [ ] Recommendations roadmap (aspect classifier, calibration, bigger frozen test, slice analysis).

**Packaging**
- [ ] Commit new scripts & updated pages.
- [ ] Freeze `config.yaml` into appendix.
- [ ] Screenshot Observability tables and Insights examples for figures.

---

## 11) Reproducibility Notes
- Run order:
  1) `python tools/prepare_test_csv.py --in artifacts/absa/test_frozen_2025-09-02.csv`
  2) `python tools/export_model.py --run runs/roberta_absa --out artifacts/models/roberta_absa_v2`
  3) `python tools/eval_model.py --model-dir artifacts/models/roberta_absa_v2 --in-csv artifacts/training/frozen_test_for_eval.csv --tag v2`
  4) `python tools/eval_baselines.py --in-csv artifacts/training/frozen_test_for_eval.csv`
  5) `python tools/make_confusion_matrix.py --pred-csv artifacts/training/v2_pred.csv --pred-col v2_pred --tag v2` (repeat for baselines)
  6) `python tools/mcnemar_compare.py --a-csv artifacts/training/v2_pred.csv --a-col v2_pred --b-csv artifacts/training/vader_preds.csv --b-col vader_pred --pair-name v2_vs_vader` (and textblob)
  7) `streamlit run streamlit_app/Home.py`

- All paths are **relative to repo root**; quote paths with spaces on macOS.
