# Hallucination-Resistant Clinical Question Answering

Source code for the MSc dissertation *Hallucination-Resistant Clinical Question Answering: A Multi-Agent
Retrieval-Augmented Pipeline with NLI-Based Verification on Consumer Hardware*
(MSc Machine Learning and Artificial Intelligence, Liverpool John Moores University).

**Author:** Amit Choudhary · Student No. 1187247

## What the system does

A six-agent pipeline answers a free-text clinical question and checks every generated claim against
retrieved evidence before the answer is returned:

| Agent | Role |
|---|---|
| OrchestratorAgent | ReAct-style loop (at most 5 iterations) over a shared `PipelineState` |
| RetrieverAgent | BioBERT embeddings (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`), ChromaDB cosine search, 30 candidates → MMR (λ = 0.5) → top 10 |
| RerankAgent | Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, keeps the top 5 passages |
| GeneratorAgent | `microsoft/BioGPT` (347M), greedy decoding, up to 200 new tokens |
| HallucinationDetectorAgent | Splits the answer into claims and scores each against the 5 passages with `cross-encoder/nli-deberta-v3-base`; claims with entailment below 0.35 are removed |
| CitationAgent | Links each retained claim to its best-supporting passage |

A FastAPI service (`/query`, `/health`, `/metrics`) and a Streamlit demonstration interface wrap the pipeline.
Everything runs locally on an Apple M1 laptop with 8 GB of unified memory (PyTorch MPS backend).

## Four-condition evaluation (n = 50 PubMedQA questions)

| Condition | Description | Hallucination rate | Decision accuracy | P95 latency |
|---|---|---|---|---|
| A | Base model, no retrieval | 0.52 | 20% | 1.38 s |
| B | + BioBERT retrieval | 0.94 | 20% | 12.92 s |
| C | + cross-encoder reranking | 0.98 | 18% | 10.78 s |
| D (raw) | + NLI filter, as recorded | 0.98 | 18% | 9.95 s |
| D (corrected) | + NLI filter, fallback bug corrected | 0.10 | 0% | 9.95 s |

Hallucination is measured per claim against the gold PubMedQA abstract (entailment threshold 0.35).
The raw Condition D records were affected by a fallback in the evaluation script that returned the unfiltered
answer when the filter removed every claim; `scripts/analyze_eval_results.py` applies the documented correction.
Corrected D versus A: Welch's t = 5.045, p < 0.001, Cohen's d = 1.01 (80.77% relative reduction).

## Repository layout

```
src/agents/       orchestrator, retriever, reranker, generator, hallucination detector, citation, evaluator
src/models/       LLM wrappers (BioGPT, BioMistral, Ollama), NLI model, embeddings, model manager, NLI trainer
src/data/         downloader, preprocessor (cleaning, deduplication, chunking), ChromaDB vector store
src/evaluation/   retrieval, baseline, hallucination and RAGAS-style evaluation utilities
src/api/          FastAPI application and schemas
src/ui/           Streamlit demonstration interface
scripts/          run_four_condition_eval.py, analyze_eval_results.py, benchmark and load-test scripts
results/          four_condition_eval.json (per-question records), supplementary_analysis.json
tests/            pytest suite (215 test functions in 11 modules)
```

## Datasets (public)

The datasets are not stored in this repository; the scripts download and rebuild them.

- **PubMedQA** (Jin et al., 2019): https://huggingface.co/datasets/qiaojin/PubMedQA
  The expert-labelled subset (PQA-L) supplies the evaluation questions and gold abstracts; the artificially
  generated subset (PQA-A) supplies most of the retrieval corpus.
- **PubMed abstracts** via NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/

The persisted ChromaDB index used in the dissertation contains 668,909 chunks (about 5.5 GB including raw data).

## Reproducing the results

```bash
pip install -r requirements.txt
cp .env.example .env                      # set PUBMED_EMAIL for NCBI requests

python -m src.data.downloader --dataset all
python -m src.data.preprocessor           # clean, deduplicate and chunk (512 words, 64-word overlap)
python -m src.data.vectorstore --action ingest

python scripts/run_four_condition_eval.py --n 50 --seed 42 --output results/four_condition_eval.json
python scripts/analyze_eval_results.py --results results/four_condition_eval.json
```

The evaluation in the dissertation was run with Python 3.13 on macOS (Apple M1, 8 GB). The pins in
`requirements.txt` predate that environment and may need updating for Python 3.13.

## Running the demo

```bash
make run-api      # FastAPI on http://localhost:8000
make run-ui       # Streamlit on http://localhost:8501
```

## Tests

```bash
make test
```

## Disclaimer

This is a research prototype. It has not been validated with clinicians and must not be used for clinical
decision-making.
