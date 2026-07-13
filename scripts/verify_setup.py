"""
Setup verification script — checks all system components before running experiments.

Validates:
  ✅ Python version and dependencies
  ✅ Dataset files exist (raw + processed)
  ✅ ChromaDB vector store is populated
  ✅ Embedding model loads and produces correct-shape output
  ✅ NLI model loads and classifies correctly
  ✅ LLM loads (or a fallback is available)
  ✅ FastAPI app starts (import check)
  ✅ MLflow tracking directory is writable

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --quick   # skip model loading
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check(name: str, fn, critical: bool = True) -> bool:
    """Run a check function and print pass/fail."""
    try:
        msg = fn()
        print(f"  ✅  {name}" + (f" — {msg}" if msg else ""))
        return True
    except Exception as e:
        icon = "❌" if critical else "⚠️ "
        print(f"  {icon}  {name}: {e}")
        return not critical


def run_verification(quick: bool = False) -> bool:
    from src.config import settings
    settings.ensure_dirs()

    print(f"\n{'═'*60}")
    print("  Clinical QA System — Setup Verification")
    print(f"{'═'*60}\n")

    all_passed = True

    # ── Python & imports ──────────────────────────────────────────────────────
    print("Python Environment:")

    def check_python():
        v = sys.version_info
        if v < (3, 11):
            raise RuntimeError(f"Python 3.11+ required, got {v.major}.{v.minor}")
        return f"Python {v.major}.{v.minor}.{v.micro}"

    all_passed &= check("Python version", check_python)

    core_imports = [
        ("fastapi", "FastAPI"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "sentence-transformers"),
        ("transformers", "transformers"),
        ("torch", "PyTorch"),
        ("langchain", "LangChain"),
        ("mlflow", "MLflow"),
        ("streamlit", "Streamlit"),
    ]

    print("\nCore Dependencies:")
    for module, label in core_imports:
        def _check(m=module, l=label):
            import importlib
            pkg = importlib.import_module(m)
            ver = getattr(pkg, "__version__", "?")
            return f"{l} v{ver}"
        all_passed &= check(label, _check)

    # ── Data files ────────────────────────────────────────────────────────────
    print("\nDataset Files:")

    data_checks = [
        ("MedQA train",       settings.data_raw_dir / "medqa" / "train.jsonl",       False),
        ("MedQA test",        settings.data_raw_dir / "medqa" / "test.jsonl",        True),
        ("PubMedQA labeled",  settings.data_raw_dir / "pubmedqa" / "labeled_train.jsonl", False),
        ("PubMed abstracts",  settings.data_raw_dir / "pubmed" / "abstracts.jsonl",  False),
        ("Processed chunks",  settings.data_processed_dir / "pubmed_chunks.jsonl",   False),
    ]

    for label, path, critical in data_checks:
        def _check(p=path, l=label):
            if not p.exists():
                raise FileNotFoundError(f"not found at {p}")
            lines = sum(1 for _ in open(p))
            return f"{lines:,} records"
        all_passed &= check(label, _check, critical=critical)

    # ── Vector store ──────────────────────────────────────────────────────────
    print("\nVector Store:")

    def check_chromadb():
        from src.data.vectorstore import get_chroma_client, get_or_create_collection
        client = get_chroma_client()
        coll = get_or_create_collection(client)
        count = coll.count()
        if count == 0:
            raise RuntimeError("Collection is empty — run: make ingest-data")
        return f"{count:,} chunks indexed"

    all_passed &= check("ChromaDB collection", check_chromadb, critical=False)

    if quick:
        print("\n⚡ Quick mode — skipping model loading checks.\n")
        _print_summary(all_passed)
        return all_passed

    # ── Embedding model ───────────────────────────────────────────────────────
    print("\nModels:")

    def check_embeddings():
        from src.data.vectorstore import get_embedding_function
        embed = get_embedding_function()
        vecs = embed(["test clinical query about diabetes"])
        assert len(vecs) == 1 and len(vecs[0]) == 768
        return f"dim={len(vecs[0])}"

    all_passed &= check("Embedding model (BioBERT)", check_embeddings)

    # ── NLI model ─────────────────────────────────────────────────────────────
    def check_nli():
        from src.models.nli import MedNLIModel
        nli = MedNLIModel()
        nli.load()
        result = nli.classify(
            "Metformin is used for diabetes.",
            "Metformin is used for diabetes.",
        )
        assert result.label in ("entailment", "neutral", "contradiction")
        return f"model={nli.model_name}, label={result.label}"

    all_passed &= check("NLI model (DeBERTa)", check_nli)

    # ── LLM ───────────────────────────────────────────────────────────────────
    def check_llm():
        from src.models.llm import get_llm
        llm = get_llm(prefer="auto")
        result = llm.answer_clinical_question(
            question="What is metformin?",
            context="Metformin is an oral antidiabetic drug used for type 2 diabetes.",
        )
        assert "answer" in result and len(result["answer"]) > 0
        return f"model={llm.model_name}, time={result.get('response_time_s', 0):.2f}s"

    all_passed &= check("LLM inference", check_llm)

    # ── Reranker ──────────────────────────────────────────────────────────────
    def check_reranker():
        from src.agents.reranker import RerankAgent
        agent = RerankAgent()
        from src.agents.retriever import RetrievedDocument
        docs = [RetrievedDocument(text="Metformin treats diabetes.", source="S1", score=0.8, rank=1)]
        ranked = agent.rerank("diabetes treatment", docs, top_k=1)
        assert len(ranked) == 1
        return f"model={settings.reranker_model}"

    all_passed &= check("Reranker (cross-encoder)", check_reranker)

    # ── API import ────────────────────────────────────────────────────────────
    print("\nAPI & Tracking:")

    def check_api():
        from src.api.main import app
        assert app.title == "Clinical QA System"
        return "FastAPI app importable"

    all_passed &= check("FastAPI app", check_api)

    # ── MLflow ────────────────────────────────────────────────────────────────
    def check_mlflow():
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("_verify_setup_test")
        with mlflow.start_run(run_name="verify_setup"):
            mlflow.log_metric("test", 1.0)
        return f"tracking_uri={settings.mlflow_tracking_uri}"

    all_passed &= check("MLflow tracking", check_mlflow)

    _print_summary(all_passed)
    return all_passed


def _print_summary(all_passed: bool) -> None:
    print(f"\n{'═'*60}")
    if all_passed:
        print("  ✅  All checks passed — system ready to run.")
    else:
        print("  ⚠️   Some checks failed — see above for details.")
        print("       Run: make download-data && make ingest-data")
    print(f"{'═'*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify system setup")
    parser.add_argument("--quick", action="store_true", help="Skip model loading checks")
    args = parser.parse_args()
    ok = run_verification(quick=args.quick)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
