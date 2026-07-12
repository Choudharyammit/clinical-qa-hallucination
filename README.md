# Hallucination-Resistant Clinical QA System

**LJMU MSc Machine Learning & AI — Masters Research Project**  
**Student:** Amit Choudhary | **Student No:** 1187247  
**Supervisor:** TBC | **Deadline:** 7-day sprint

---

## Overview

A production-grade clinical question-answering system that combines Retrieval-Augmented Generation (RAG) with post-generation hallucination detection to provide evidence-grounded, citation-backed answers to medical questions.

**Key Innovation:** Every factual claim in the generated response is verified against retrieved evidence using DeBERTa-v3 NLI, achieving ≥50% reduction in hallucination rate versus the base LLM baseline.

---

## System Architecture

```
User Query
    │
    ▼
OrchestratorAgent (ReAct loop, max 5 iterations)
    │
    ├─► RetrieverAgent    ── BioBERT embeddings + ChromaDB + MMR
    │
    ├─► RerankAgent       ── cross-encoder/ms-marco-MiniLM-L-6-v2
    │
    ├─► GeneratorAgent    ── BioMistral-7B-DARE (4-bit quantized)
    │
    ├─► HallucinationDetectorAgent ── DeBERTa-v3 NLI on MedNLI
    │
    └─► CitationAgent     ── inline [N] citations + reference list
            │
            ▼
    REST API (FastAPI) ◄──► Streamlit UI
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- 8GB+ RAM (16GB recommended for BioMistral-7B)
- GPU optional (falls back to CPU with quantization)

### Setup

```bash
# 1. Clone and install
git clone <repo>
cd clinical-qa-system
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work without API keys)

# 3. Download datasets
make download-data

# 4. Preprocess and ingest into vector store
make ingest-data

# 5. Start the API
make run-api

# 6. Start the UI (new terminal)
make run-ui
```

### Docker (recommended)

```bash
docker-compose up -d
# API:    http://localhost:8000
# UI:     http://localhost:8501
# MLflow: http://localhost:5000
```

---

## API Reference

### POST /query
```json
{
  "question": "What is the first-line treatment for type 2 diabetes?",
  "use_rag": true,
  "use_hallucination_filter": true,
  "mcq_options": {"A": "Metformin", "B": "Insulin", "C": "Surgery", "D": "Aspirin"}
}
```

**Response:**
```json
{
  "answer": "Metformin is the first-line treatment... [1]",
  "mcq_prediction": "A",
  "confidence": "High",
  "hallucination_rate": 0.05,
  "response_time_s": 1.87,
  "citations": [{"source": "PubMed:12345", "pmid": "12345", ...}],
  "hallucination_report": {"total_claims": 4, "hallucinated_claims": 0, ...}
}
```

### GET /health
Returns system health status, vector store count, model status.

### GET /metrics
Returns latest benchmark evaluation metrics.

---

## Datasets

| Dataset | Source | Size | Use |
|---------|--------|------|-----|
| MedQA (USMLE 4-opt) | HuggingFace: GBaker/MedQA-USMLE-4-options | 11,723 QA pairs | Accuracy evaluation |
| PubMedQA | HuggingFace: qiaojin/PubMedQA | 211K QA pairs | RAG corpus + RAGAS eval |
| PubMed Abstracts | NCBI E-utilities (no auth) | Up to 50K abstracts | Retrieval corpus |

All datasets are freely available without institutional access.

---

## Evaluation

Run the full 3-condition comparison study:

```bash
# All conditions: base LLM / RAG only / RAG + hallucination filter
make evaluate

# Hallucination rate analysis + charts
make evaluate-hallucination

# RAGAS faithfulness + answer relevance
make evaluate-ragas
```

**Target Metrics:**
- MedQA Accuracy: ≥75%
- Hallucination Rate Reduction vs Base LLM: ≥50%
- API P95 Response Time: <3s

---

## Running Tests

```bash
# Full test suite with coverage
make test-cov

# Fast tests only (exclude slow model-loading tests)
make test-fast
```

---

## Project Structure

```
clinical-qa-system/
├── src/
│   ├── agents/         # 7 specialist agents
│   ├── data/           # data download, preprocessing, vectorstore
│   ├── models/         # LLM, embeddings, NLI wrappers
│   ├── evaluation/     # MedQA, hallucination, RAGAS evaluators
│   ├── api/            # FastAPI application
│   └── ui/             # Streamlit chat interface
├── tests/              # pytest test suite (>80% coverage target)
├── notebooks/          # Jupyter exploration + results analysis
├── docs/               # Auto-generated API docs + architecture
├── report/             # LaTeX/Markdown academic report
├── data/               # Raw + processed datasets (git-ignored)
└── mlruns/             # MLflow experiment tracking
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11, FastAPI, uvicorn |
| LLM | BioMistral-7B-DARE (4-bit GPTQ) |
| Embeddings | BioBERT (sentence-transformers) |
| Vector DB | ChromaDB + FAISS fallback |
| NLI | DeBERTa-v3-base (MedNLI) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RAG | LangChain + LlamaIndex |
| Evaluation | RAGAS, MLflow |
| Frontend | Streamlit |
| Testing | pytest + >80% coverage |
| Deploy | Docker + docker-compose |

---

## Reproducibility

Random seeds are fixed at 42 (Python, NumPy, PyTorch).  
All dependencies are pinned in `requirements.txt`.  
To reproduce results exactly:

```bash
pip install -r requirements.txt
make download-data && make ingest-data
python -m src.evaluation.medqa_eval --condition all --max-samples 500
```

---

## Academic Report

The full 15,000–20,000 word LJMU-standard research report is in `report/masters_report.md`.

---

## License

MIT — for academic purposes only.  
Dataset licenses: MedQA (MIT), PubMedQA (MIT), PubMed abstracts (NLM terms of service).
