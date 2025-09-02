# Simple quality-of-life commands
.PHONY: install etl annotate baseline train rules dashboard test format

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python -m nltk.downloader punkt

etl:
	python -m src.absa_thesis.etl.load_raw
	python -m src.absa_thesis.etl.clean_normalize

annotate:
	python -m src.absa_thesis.annotation.prepare_tasks --n 2000

baseline:
	python -m src.absa_thesis.modeling.baseline_svm

train:
	python -m src.absa_thesis.modeling.train_roberta_absa

serve:
	CONFIG=config_infer.yaml streamlit run src/absa_thesis/dashboard/Home.py

rules:
	python -m src.absa_thesis.rules.build_transactions
	python -m src.absa_thesis.rules.mine_rules

dashboard:
	streamlit run src/absa_thesis/dashboard/Home.py

test:
	pytest -q
