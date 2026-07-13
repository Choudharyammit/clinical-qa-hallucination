# ============================================================
# Clinical QA System — Makefile
# LJMU MSc ML&AI | Amit Choudhary | Student No: 1187247
# ============================================================

.PHONY: install install-dev setup-env download-data ingest-data \
        run-api run-ui test test-cov lint format type-check \
        docker-build docker-up docker-down clean docs \
        halluc-stats halluc-ab-test halluc-eval-all train-nli verify-setup

PYTHON := python3.11
PIP := pip
SRC := src
TESTS := tests

# ── Setup ────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	pre-commit install

setup-env:
	cp .env.example .env
	@echo "Edit .env with your settings before running"

# ── Data Pipeline ─────────────────────────────────────────────
download-data:
	$(PYTHON) -m src.data.downloader --dataset medqa
	$(PYTHON) -m src.data.downloader --dataset pubmedqa
	$(PYTHON) -m src.data.downloader --dataset pubmed_abstracts

ingest-data:
	$(PYTHON) -m src.data.vectorstore --action ingest

# ── Services ─────────────────────────────────────────────────
run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run src/ui/app.py --server.port 8501

run-mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ── Evaluation ────────────────────────────────────────────────
evaluate:
	$(PYTHON) -m src.evaluation.medqa_eval --condition all --max-samples 500

evaluate-retrieval:
	$(PYTHON) -m src.evaluation.retrieval_eval --top-k 1 3 5 10 --max-samples 200

evaluate-baseline:
	$(PYTHON) -m src.evaluation.baseline_eval --model biogpt --max-samples 100

evaluate-hallucination:
	$(PYTHON) -m src.evaluation.hallucination_eval

evaluate-ragas:
	$(PYTHON) -m src.evaluation.ragas_eval

evaluate-all: evaluate-baseline evaluate-retrieval evaluate evaluate-ragas evaluate-hallucination

# ── Day 4: Hallucination A/B Evaluation ───────────────────────
halluc-stats:
	$(PYTHON) scripts/run_hallucination_eval.py --mode stats

halluc-ab-test:
	$(PYTHON) scripts/run_hallucination_eval.py --mode ab-test --n 100

halluc-ab-test-small:
	$(PYTHON) scripts/run_hallucination_eval.py --mode ab-test --n 20

halluc-eval-all:
	$(PYTHON) scripts/run_hallucination_eval.py --mode both --n 100

halluc-summary:
	$(PYTHON) scripts/run_hallucination_eval.py --summary

# ── Day 5: Full Benchmark ──────────────────────────────────────
benchmark:
	$(PYTHON) scripts/run_full_benchmark.py --n 500

benchmark-quick:
	$(PYTHON) scripts/run_full_benchmark.py --n 50

benchmark-charts:
	$(PYTHON) scripts/run_full_benchmark.py --skip-inference --phase charts

benchmark-ragas:
	$(PYTHON) scripts/run_full_benchmark.py --phase ragas --ragas-n 100

benchmark-retrieval:
	$(PYTHON) scripts/run_full_benchmark.py --phase retrieval --retrieval-n 200

train-nli:
	$(PYTHON) -m src.models.nli_trainer --epochs 3 --batch-size 16

train-nli-dry:
	$(PYTHON) -m src.models.nli_trainer --dry-run

verify-setup:
	$(PYTHON) scripts/verify_setup.py

verify-setup-quick:
	$(PYTHON) scripts/verify_setup.py --quick

pipeline-test:
	$(PYTHON) scripts/run_pipeline.py --n-questions 100 --condition rag_with_hallucination_filter

pipeline-test-all:
	$(PYTHON) scripts/run_pipeline.py --n-questions 100 --condition all

pipeline-smoke:
	$(PYTHON) scripts/run_pipeline.py --fast

# ── Testing ───────────────────────────────────────────────────
test:
	pytest $(TESTS)/ -v

test-cov:
	pytest $(TESTS)/ -v --cov=$(SRC) --cov-report=html --cov-report=term

test-fast:
	pytest $(TESTS)/ -v -m "not slow"

# ── Code Quality ──────────────────────────────────────────────
lint:
	flake8 $(SRC)/ $(TESTS)/ --max-line-length=100
	pylint $(SRC)/

format:
	black $(SRC)/ $(TESTS)/
	isort $(SRC)/ $(TESTS)/

type-check:
	mypy $(SRC)/

# ── Docker ────────────────────────────────────────────────────
docker-build:
	docker build -t clinical-qa-system:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ── Documentation ─────────────────────────────────────────────
docs:
	pdoc3 --html --output-dir docs/api $(SRC)/
	@echo "Docs generated at docs/api/"

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage .pytest_cache/
	@echo "Cleaned."

clean-data:
	rm -rf data/raw/* data/processed/* data/chroma_db/
	@echo "Data directories cleaned."

# ── Git ───────────────────────────────────────────────────────
git-init:
	git init
	git add .
	git commit -m "feat: initial project scaffold — Day 1 complete"

tag-release:
	git tag -a v1.0.0 -m "Production release — LJMU MSc Final Submission"
