"""
Text preprocessing pipeline for the Clinical QA System.

Handles:
  - Loading raw JSONL datasets (MedQA, PubMedQA, PubMed abstracts)
  - Text cleaning (HTML stripping, whitespace normalization, deduplication)
  - Document chunking with overlap (sliding window)
  - Output: processed JSONL ready for vector store ingestion

Usage:
    python -m src.data.preprocessor --source pubmed --input data/raw/pubmed/abstracts.jsonl
    python -m src.data.preprocessor --source medqa  --input data/raw/medqa/train.jsonl
    python -m src.data.preprocessor --source all
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Generator, Optional

from loguru import logger
from tqdm import tqdm

from src.config import settings


# ── Text Cleaning ─────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")       # [1], [1,2,3]
_SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z\s]{2,}:\s*") # "METHODS: ..."


def clean_text(text: str) -> str:
    """Normalize and clean raw medical text."""
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Strip HTML tags
    text = _HTML_TAG_RE.sub(" ", text)
    # Remove citation brackets [1], [2,3]
    text = _CITATION_RE.sub("", text)
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def content_hash(text: str) -> str:
    """MD5 fingerprint for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    chunk_size and chunk_overlap are measured in tokens (words as proxy).
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - chunk_overlap

    return [c for c in chunks if len(c.split()) >= 20]  # Drop tiny tail chunks


# ── Loaders ───────────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Generator[dict, None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def process_pubmed_abstracts(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    """
    Process PubMed abstract JSONL into chunks.
    Each output record: {chunk_id, text, source, pmid, title, metadata}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    written = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in tqdm(_iter_jsonl(input_path), desc="Processing PubMed abstracts"):
            raw_text = f"{record.get('title', '')}. {record.get('abstract', '')}"
            text = clean_text(raw_text)
            if not text or len(text) < 50:
                continue

            chunks = chunk_text(text, chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                h = content_hash(chunk)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                out_f.write(json.dumps({
                    "chunk_id": f"pubmed_{record.get('pmid', 'unk')}_{i}",
                    "text": chunk,
                    "source": record.get("source", f"PubMed:{record.get('pmid', '')}"),
                    "pmid": record.get("pmid", ""),
                    "title": record.get("title", ""),
                    "year": record.get("year", ""),
                    "journal": record.get("journal", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }) + "\n")
                written += 1

    logger.info(f"PubMed processed: {written} chunks → {output_path}")
    return written


def process_pubmedqa(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    """
    Process PubMedQA JSONL into chunks.
    Uses context passages as retrieval corpus and preserves QA pairs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    written = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in tqdm(_iter_jsonl(input_path), desc="Processing PubMedQA"):
            pubid = record.get("pubid", "unk")
            contexts = record.get("context", {}).get("contexts", [])
            question = clean_text(record.get("question", ""))
            long_answer = clean_text(record.get("long_answer", ""))
            final_decision = record.get("final_decision", "")

            # Index context passages as retrieval corpus
            for j, ctx in enumerate(contexts):
                text = clean_text(ctx)
                if not text or len(text) < 30:
                    continue
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                for i, chunk in enumerate(chunks):
                    h = content_hash(chunk)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    out_f.write(json.dumps({
                        "chunk_id": f"pubmedqa_{pubid}_ctx{j}_{i}",
                        "text": chunk,
                        "source": f"PubMedQA:{pubid}",
                        "pubid": pubid,
                        "question": question,
                        "final_decision": final_decision,
                        "chunk_type": "context",
                        "chunk_index": i,
                    }) + "\n")
                    written += 1

            # Also index the long answer if present
            if long_answer and len(long_answer) > 50:
                for i, chunk in enumerate(chunk_text(long_answer, chunk_size, chunk_overlap)):
                    h = content_hash(chunk)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    out_f.write(json.dumps({
                        "chunk_id": f"pubmedqa_{pubid}_ans_{i}",
                        "text": chunk,
                        "source": f"PubMedQA:{pubid}",
                        "pubid": pubid,
                        "question": question,
                        "final_decision": final_decision,
                        "chunk_type": "answer",
                        "chunk_index": i,
                    }) + "\n")
                    written += 1

    logger.info(f"PubMedQA processed: {written} chunks → {output_path}")
    return written


def process_medqa(input_path: Path, output_path: Path) -> int:
    """
    Process MedQA JSONL for evaluation (no chunking needed — QA format).
    Cleans question + option text; stores structured for EvaluatorAgent.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in tqdm(_iter_jsonl(input_path), desc="Processing MedQA"):
            question = clean_text(record.get("question", ""))
            options = record.get("options", {})
            if isinstance(options, dict):
                options = {k: clean_text(v) for k, v in options.items()}
            elif isinstance(options, list):
                options = {chr(65 + i): clean_text(v) for i, v in enumerate(options)}

            if not question:
                continue

            out_f.write(json.dumps({
                "question": question,
                "options": options,
                "answer": record.get("answer", ""),
                "answer_idx": record.get("answer_idx", ""),
            }) + "\n")
            written += 1

    logger.info(f"MedQA processed: {written} records → {output_path}")
    return written


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_all_preprocessing(
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict[str, int]:
    """Run full preprocessing pipeline for all available raw data."""
    raw = settings.data_raw_dir
    proc = settings.data_processed_dir
    results = {}

    # PubMed abstracts
    pubmed_raw = raw / "pubmed" / "abstracts.jsonl"
    if pubmed_raw.exists():
        results["pubmed"] = process_pubmed_abstracts(
            pubmed_raw,
            proc / "pubmed_chunks.jsonl",
            chunk_size,
            chunk_overlap,
        )
    else:
        logger.warning(f"PubMed raw file not found: {pubmed_raw}")

    # PubMedQA
    for fname in (raw / "pubmedqa").glob("*.jsonl") if (raw / "pubmedqa").exists() else []:
        key = f"pubmedqa_{fname.stem}"
        results[key] = process_pubmedqa(
            fname,
            proc / f"pubmedqa_{fname.stem}_chunks.jsonl",
            chunk_size,
            chunk_overlap,
        )

    # MedQA splits
    for split in ["train", "validation", "test"]:
        mq_raw = raw / "medqa" / f"{split}.jsonl"
        if mq_raw.exists():
            results[f"medqa_{split}"] = process_medqa(
                mq_raw,
                proc / f"medqa_{split}.jsonl",
            )

    logger.success(f"Preprocessing complete. Summary: {results}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess clinical datasets")
    parser.add_argument(
        "--source",
        choices=["pubmed", "pubmedqa", "medqa", "all"],
        default="all",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    settings.ensure_dirs()

    if args.source == "all":
        run_all_preprocessing(args.chunk_size, args.chunk_overlap)
    elif args.source == "pubmed":
        inp = args.input or settings.data_raw_dir / "pubmed" / "abstracts.jsonl"
        out = args.output or settings.data_processed_dir / "pubmed_chunks.jsonl"
        process_pubmed_abstracts(inp, out, args.chunk_size, args.chunk_overlap)
    elif args.source == "pubmedqa":
        inp = args.input or settings.data_raw_dir / "pubmedqa" / "labeled_train.jsonl"
        out = args.output or settings.data_processed_dir / "pubmedqa_chunks.jsonl"
        process_pubmedqa(inp, out, args.chunk_size, args.chunk_overlap)
    elif args.source == "medqa":
        for split in ["train", "validation", "test"]:
            inp = args.input or settings.data_raw_dir / "medqa" / f"{split}.jsonl"
            if inp.exists():
                out = args.output or settings.data_processed_dir / f"medqa_{split}.jsonl"
                process_medqa(inp, out)


if __name__ == "__main__":
    main()
