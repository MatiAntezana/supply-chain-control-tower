# All targets run inside the `AIEnv` conda environment (project requirement).
PY := conda run -n AIEnv python
SHELL := /bin/bash

.PHONY: data forecast optimize simulate pipeline api dashboard test lint drift retrain-check freeze

data:            ## M1: raw M5 CSVs -> panel + features (deterministic)
	$(PY) -m src.ingest.load_m5
	$(PY) -m src.ingest.features

forecast:        ## M2: train quantile models + rolling-origin backtest
	$(PY) -m src.forecast.backtest
	$(PY) -m src.forecast.train

optimize:        ## M3: (s,S)/newsvendor + MILP policies per SKU
	$(PY) -m src.optimize.run

simulate:        ## M4: SimPy DES + Monte-Carlo validation + cost/service curve
	$(PY) -m src.simulate.run

pipeline: data forecast optimize simulate  ## full end-to-end rebuild

api:             ## M5: serve FastAPI on :8000
	conda run -n AIEnv uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

dashboard:       ## M5: Streamlit dashboard on :8501
	conda run -n AIEnv streamlit run app/dashboard.py

drift:           ## M6: Evidently drift report
	$(PY) -m src.monitor.drift

retrain-check:   ## M6: exit 1 if retrain should trigger (drift or fill-rate drop)
	$(PY) -m src.monitor.retrain_check

test:
	conda run -n AIEnv pytest -q

lint:
	conda run -n AIEnv ruff check src tests app

freeze:
	conda run -n AIEnv pip freeze > requirements.txt
