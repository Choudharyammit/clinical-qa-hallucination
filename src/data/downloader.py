"""
Dataset downloader for the Clinical QA System.

Downloads:
  - MedQA (USMLE-style 4-option MCQ) from HuggingFace datasets
  - PubMedQA (biomedical QA) from HuggingFace datasets
  - PubMed abstracts via NCBI E-utilities (no API key required for <3 req/s)

Usage:
    python -m src.data.downloader --dataset medqa
    python -m src.data.downloader --dataset pubmedqa
    python -m src.data.downloader --dataset pubmed_abstracts --max-abstracts 10000
    python -m src.data.downloader --dataset all
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import requests
from loguru import logger
from tqdm import tqdm

from src.config import settings


# ── MedQA ─────────────────────────────────────────────────────────────────────

def download_medqa(output_dir: Optional[Path] = None) -> Path:
    """
    Download MedQA (USMLE 4-option) dataset from HuggingFace.
    Returns path to output directory.

    Dataset: GBaker/MedQA-USMLE-4-options
    Splits: train (10178), validation (1272), test (1273)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install datasets: pip install datasets")

    out = output_dir or settings.data_raw_dir / "medqa"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MedQA (USMLE 4-option) from HuggingFace…")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)

    for split_name, split_data in ds.items():
        out_file = out / f"{split_name}.jsonl"
        records = []
        for item in split_data:
            records.append({
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "answer_idx": item.get("answer_idx", ""),
                "metamap_phrases": item.get("metamap_phrases", []),
            })
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.info(f"  Saved {len(records)} records → {out_file}")

    logger.success(f"MedQA download complete → {out}")
    return out


# ── PubMedQA ──────────────────────────────────────────────────────────────────

def download_pubmedqa(output_dir: Optional[Path] = None) -> Path:
    """
    Download PubMedQA labeled + unlabeled from HuggingFace.
    Returns path to output directory.

    Dataset: qiaojin/PubMedQA
    Configs: pqa_labeled (1000), pqa_unlabeled (61k), pqa_artificial (211k)
    We download pqa_labeled (gold standard) + pqa_artificial (training).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install datasets: pip install datasets")

    out = output_dir or settings.data_raw_dir / "pubmedqa"
    out.mkdir(parents=True, exist_ok=True)

    configs = [
        ("pqa_labeled", "labeled"),
        ("pqa_artificial", "artificial"),
    ]

    for hf_config, file_prefix in configs:
        logger.info(f"Downloading PubMedQA config: {hf_config}…")
        try:
            ds = load_dataset("qiaojin/PubMedQA", hf_config, trust_remote_code=True)
            for split_name, split_data in ds.items():
                out_file = out / f"{file_prefix}_{split_name}.jsonl"
                records = []
                for item in split_data:
                    records.append({
                        "pubid": item.get("pubid", ""),
                        "question": item["question"],
                        "context": {
                            "contexts": item["context"].get("contexts", []),
                            "labels": item["context"].get("labels", []),
                            "meshes": item["context"].get("meshes", []),
                        },
                        "long_answer": item.get("long_answer", ""),
                        "final_decision": item.get("final_decision", ""),
                    })
                with open(out_file, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                logger.info(f"  Saved {len(records)} records → {out_file}")
        except Exception as e:
            logger.warning(f"Failed to download {hf_config}: {e}")

    logger.success(f"PubMedQA download complete → {out}")
    return out


# ── PubMed Abstracts via NCBI E-utilities ────────────────────────────────────

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Medical search queries for a broad clinical corpus
PUBMED_SEARCH_TERMS = [
    "clinical decision making[MeSH] AND diagnosis[MeSH]",
    "drug therapy[MeSH] AND evidence-based medicine[MeSH]",
    "clinical guidelines[MeSH] AND treatment outcome[MeSH]",
    "disease[MeSH] AND pathophysiology[MeSH]",
    "symptoms[MeSH] AND differential diagnosis[MeSH]",
    "pharmacology[MeSH] AND adverse effects[MeSH]",
    "surgery[MeSH] AND complications[MeSH]",
    "infectious disease[MeSH] AND antimicrobial therapy",
    "cardiology[MeSH] AND heart failure[MeSH]",
    "oncology[MeSH] AND cancer treatment[MeSH]",
    "neurology[MeSH] AND neurological disorders[MeSH]",
    "emergency medicine[MeSH] AND acute care",
    "pediatrics[MeSH] AND childhood disease[MeSH]",
    "psychiatry[MeSH] AND mental health[MeSH]",
    "internal medicine[MeSH] AND chronic disease[MeSH]",
]


def _pubmed_search(term: str, max_results: int, email: str) -> list[str]:
    """Run ESearch and return list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": min(max_results, 10000),
        "retmode": "json",
        "email": email,
    }
    resp = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _pubmed_fetch_abstracts(pmids: list[str], email: str) -> list[dict]:
    """Fetch abstract text for a batch of PMIDs via EFetch."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        "email": email,
    }
    resp = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()

    records = []
    try:
        root = ET.fromstring(resp.content)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            year_el = article.find(".//PubDate/Year")
            journal_el = article.find(".//Journal/Title")

            pmid = pmid_el.text if pmid_el is not None else ""
            title = title_el.text if title_el is not None else ""
            abstract = abstract_el.text if abstract_el is not None else ""
            year = year_el.text if year_el is not None else ""
            journal = journal_el.text if journal_el is not None else ""

            if abstract and len(abstract) > 50:
                records.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "journal": journal,
                    "source": f"PubMed:{pmid}",
                })
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")

    return records


def download_pubmed_abstracts(
    max_abstracts: int = 50000,
    output_dir: Optional[Path] = None,
    email: str = "",
) -> Path:
    """
    Download PubMed abstracts via NCBI E-utilities (no API key needed).
    Respects the 3 requests/second rate limit for unauthenticated access.
    Returns path to output JSONL file.
    """
    email = email or settings.pubmed_email
    out = output_dir or settings.data_raw_dir / "pubmed"
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "abstracts.jsonl"

    batch_size = settings.pubmed_batch_size
    total_collected = 0
    all_pmids: list[str] = []

    logger.info(f"Searching PubMed for up to {max_abstracts} abstracts…")

    # Collect PMIDs across search terms
    per_term = max(max_abstracts // len(PUBMED_SEARCH_TERMS), 500)
    for term in PUBMED_SEARCH_TERMS:
        if total_collected >= max_abstracts:
            break
        logger.info(f"  Searching: {term[:60]}…")
        try:
            pmids = _pubmed_search(term, per_term, email)
            all_pmids.extend(pmids)
            total_collected += len(pmids)
            time.sleep(0.4)  # Stay under 3 req/s
        except Exception as e:
            logger.warning(f"  Search failed: {e}")

    # Deduplicate
    all_pmids = list(dict.fromkeys(all_pmids))[:max_abstracts]
    logger.info(f"Fetching abstracts for {len(all_pmids)} unique PMIDs…")

    written = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(all_pmids), batch_size), desc="Fetching PubMed batches"):
            batch = all_pmids[i : i + batch_size]
            try:
                records = _pubmed_fetch_abstracts(batch, email)
                for r in records:
                    f.write(json.dumps(r) + "\n")
                written += len(records)
                time.sleep(0.4)
            except Exception as e:
                logger.warning(f"  Batch {i//batch_size} failed: {e}")

    logger.success(f"PubMed abstracts complete: {written} saved → {out_file}")
    return out


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download clinical QA datasets")
    parser.add_argument(
        "--dataset",
        choices=["medqa", "pubmedqa", "pubmed_abstracts", "all"],
        required=True,
        help="Dataset to download",
    )
    parser.add_argument(
        "--max-abstracts",
        type=int,
        default=settings.pubmed_max_abstracts,
        help="Maximum PubMed abstracts to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    settings.ensure_dirs()

    if args.dataset in ("medqa", "all"):
        download_medqa(args.output_dir)
    if args.dataset in ("pubmedqa", "all"):
        download_pubmedqa(args.output_dir)
    if args.dataset in ("pubmed_abstracts", "all"):
        download_pubmed_abstracts(
            max_abstracts=args.max_abstracts,
            output_dir=args.output_dir,
            email=settings.pubmed_email,
        )


if __name__ == "__main__":
    main()
