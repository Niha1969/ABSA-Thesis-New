# Description of Practical Research Work Undertaken

## Requirements
**Stakeholders:** tech product teams who need timely, actionable feedback from public reviews.  
**Non‑functional:** <10 min refresh latency; reproducible ETL; audit trails; WCAG‑AA dashboard; SUS ≥ 70.  
**Data:** ≥50k public reviews (research‑licensed datasets), stripped of PII and stored as canonical schema.

## Design
**Architecture:** batch ETL → cleaned Parquet → optional Mongo for serving → ABSA training/eval → rule mining → Streamlit dashboard.  
**ABSA task:** sentence‑level labeling of aspect category and polarity; domain adaptation using RoBERTa.  
**Evaluation plan:** macro‑F1, throughput; McNemar vs SVM baseline; rule quality (support, confidence, relevance); SUS.

## Implementation
- **ETL.** `load_raw.py` ingests JSONL/CSV into a canonical schema; `clean_normalize.py` de‑duplicates and trims; optional `to_mongo.py` loads to Mongo. Paths and thresholds in `config.yaml`.
- **Annotation.** `prepare_tasks.py` splits reviews into sentences and writes Label Studio tasks; `guidelines.md` and `labelstudio_config.xml` define consistent labeling; `merge_annotations.py` merges exports and creates train/val/test.
- **Baselines.** `baseline_svm.py` trains a weakly‑labeled TF‑IDF + Logistic model for sentiment to establish timing/accuracy baselines.
- **Transformer.** `train_roberta_absa.py` fine‑tunes RoBERTa for polarity (extendable to aspect‑conditioned inputs). Target macro‑F1 ≥ 0.85 on validation.
- **Association rules.** `build_transactions.py` converts ABSA labels to transactions; `mine_rules.py` runs Apriori with thresholds (support ≥ 0.02, confidence ≥ 0.6) and saves JSON for the dashboard.
- **Dashboard.** Streamlit pages: Home (health), Insights (rules table), Inference (demo classifier), Observability (data snapshots), Admin (setup).
- **Usability.** SUS materials in `study/` with scorer script.

## Testing
- Unit tests for cleaning functions (`tests/test_cleaning.py`).
- Run‑books via `Makefile`; deterministic splits (fixed seeds).

## Tools & Technologies
Python, Pandas, scikit‑learn, Transformers, mlxtend, Streamlit, PyArrow; optional MongoDB. Development on macOS (M1) with `.venv`; Docker for services if used.

## Alternatives Considered and Rationale
- **Data store:** chose Parquet + optional Mongo over full Hadoop/Spark due to dataset size and simplicity; can scale later.  
- **Annotation:** Label Studio over bespoke UI for speed and auditability.  
- **Modeling:** RoBERTa for strong ABSA benchmarks; SVM baseline retained to support statistical significance testing.  
- **Rules:** Apriori for interpretability; can swap to FP‑Growth if latency becomes an issue.

## Legal, Social, Security, Ethical & Professional
- Use research‑licensed datasets; document licenses and sources; no direct scraping of ToS‑restricted content.  
- Strip PII, encrypt at rest where needed; purge data after 12 months.  
- Bias monitoring by error breakdown per aspect and polarity; retraining if disparities observed.  
- Accessibility checks on the dashboard; provide methodological caveats for end‑users.

## Testing and Validation
- Macro‑F1 reported with confidence intervals; McNemar test vs baseline.  
- Rule sets filtered and manually rated by two product managers (relevance scale 1–5).  
- SUS scores summarized with bootstrapped 95% CI; median task completion times reported.

## Limitations
- Polarity model initially sentence‑only; future work adds explicit aspect‑sentence pairs.  
- Sample data provided; performance numbers illustrative until full label set is created.

## Summary
The implemented pipeline is reproducible, auditable, and aligned to evaluation targets. It establishes the scaffolding to collect evidence for the research question and to meet the module’s practical‑work marking criteria.
