"""
Shared pytest fixtures for the Clinical QA System test suite.
Heavy models (LLM, embeddings, NLI) are mocked by default so the suite
runs fast without GPU. Tests that need real models are marked @pytest.mark.slow.
"""

import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── Paths ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    (tmp_path / "chroma_db").mkdir()
    return tmp_path


# ── Sample data ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_medqa_records() -> list[dict]:
    return [
        {
            "question": "A 45-year-old patient presents with polyuria and polydipsia. Fasting glucose is 210 mg/dL. What is the first-line pharmacological treatment?",
            "options": {"A": "Metformin", "B": "Glipizide", "C": "Insulin glargine", "D": "Sitagliptin"},
            "answer": "Metformin",
            "answer_idx": "A",
        },
        {
            "question": "Which antibiotic is first-line for community-acquired pneumonia in a healthy outpatient?",
            "options": {"A": "Vancomycin", "B": "Amoxicillin", "C": "Meropenem", "D": "Linezolid"},
            "answer": "Amoxicillin",
            "answer_idx": "B",
        },
        {
            "question": "A patient with hypertension and diabetes should receive which antihypertensive as first-line?",
            "options": {"A": "Thiazide diuretic", "B": "ACE inhibitor", "C": "Beta blocker", "D": "Calcium channel blocker"},
            "answer": "ACE inhibitor",
            "answer_idx": "B",
        },
    ]


@pytest.fixture
def sample_pubmedqa_records() -> list[dict]:
    return [
        {
            "pubid": "12345678",
            "question": "Does metformin reduce cardiovascular mortality in type 2 diabetes?",
            "context": {
                "contexts": [
                    "Metformin has been shown to reduce all-cause mortality in type 2 diabetic patients.",
                    "The UKPDS trial demonstrated significant reduction in diabetes-related endpoints with metformin.",
                ],
                "labels": ["yes", "yes"],
                "meshes": ["Metformin", "Diabetes Mellitus, Type 2"],
            },
            "long_answer": "Yes, metformin reduces cardiovascular mortality in type 2 diabetes based on multiple RCTs.",
            "final_decision": "yes",
        },
    ]


@pytest.fixture
def sample_pubmed_abstracts() -> list[dict]:
    return [
        {
            "pmid": "11111111",
            "title": "Metformin therapy in type 2 diabetes",
            "abstract": "Metformin is the cornerstone of pharmacological therapy for type 2 diabetes mellitus. It reduces hepatic glucose production and improves insulin sensitivity. Clinical trials including the UKPDS have demonstrated reduction in diabetes-related endpoints and mortality.",
            "source": "PubMed:11111111",
            "year": "2020",
            "journal": "New England Journal of Medicine",
        },
        {
            "pmid": "22222222",
            "title": "Antibiotics for community-acquired pneumonia",
            "abstract": "Community-acquired pneumonia is commonly treated with amoxicillin or amoxicillin-clavulanate as first-line therapy in outpatient settings. Macrolides may be added for atypical coverage. Fluoroquinolones are reserved for patients with comorbidities.",
            "source": "PubMed:22222222",
            "year": "2021",
            "journal": "Lancet Infectious Diseases",
        },
    ]


# ── Mock embedding function ────────────────────────────────────────────────────

@pytest.fixture
def mock_embed_fn():
    """Returns deterministic embeddings (unit vectors) for any text list."""
    rng = np.random.default_rng(42)

    def _embed(texts: list[str]) -> list[list[float]]:
        vecs = rng.standard_normal((len(texts), 768)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-8)
        return vecs.tolist()

    return _embed


# ── Mock ChromaDB collection ────────────────────────────────────────────────────

@pytest.fixture
def mock_collection(sample_pubmed_abstracts):
    """ChromaDB collection pre-populated with sample abstracts."""
    collection = MagicMock()
    collection.count.return_value = len(sample_pubmed_abstracts)
    collection.query.return_value = {
        "documents": [["Metformin is first-line for type 2 diabetes.", "ACE inhibitors are used for hypertension."]],
        "metadatas": [
            [
                {"source": "PubMed:11111111", "pmid": "11111111", "title": "Metformin therapy", "year": "2020", "journal": "NEJM"},
                {"source": "PubMed:22222222", "pmid": "22222222", "title": "Pneumonia antibiotics", "year": "2021", "journal": "Lancet"},
            ]
        ],
        "distances": [[0.08, 0.25]],
    }
    return collection


# ── Mock NLI model ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_nli():
    """NLI model that supports long claims and flags very short ones."""
    nli = MagicMock()

    def score_claim(claim, evidence_chunks):
        entailment = 0.88 if len(claim.split()) > 5 else 0.15
        return {
            "claim": claim,
            "max_entailment": entailment,
            "is_supported": entailment >= 0.7,
            "best_evidence": evidence_chunks[0] if evidence_chunks else "",
            "label": "entailment" if entailment >= 0.7 else "neutral",
            "all_scores": [entailment] * len(evidence_chunks) if evidence_chunks else [],
        }

    nli.score_claim_against_evidence.side_effect = score_claim
    return nli


# ── Mock LLM ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.is_loaded = True
    llm.model_name = "mock-biomistral"

    def answer_clinical_question(question, context, mcq_options=None):
        if mcq_options:
            return {
                "answer": "Based on the context, the answer is A. Metformin is the first-line treatment.",
                "response_time_s": 0.1,
                "model": "mock-biomistral",
            }
        return {
            "answer": "Metformin is the first-line treatment for type 2 diabetes mellitus. It reduces hepatic glucose production. Confidence: High",
            "response_time_s": 0.1,
            "model": "mock-biomistral",
        }

    llm.answer_clinical_question.side_effect = answer_clinical_question
    return llm


# ── Temporary JSONL files ──────────────────────────────────────────────────────

@pytest.fixture
def medqa_jsonl(tmp_path, sample_medqa_records) -> Path:
    f = tmp_path / "medqa_test.jsonl"
    with open(f, "w") as fp:
        for r in sample_medqa_records:
            fp.write(json.dumps(r) + "\n")
    return f


@pytest.fixture
def pubmed_jsonl(tmp_path, sample_pubmed_abstracts) -> Path:
    f = tmp_path / "abstracts.jsonl"
    with open(f, "w") as fp:
        for r in sample_pubmed_abstracts:
            fp.write(json.dumps(r) + "\n")
    return f


@pytest.fixture
def pubmedqa_jsonl(tmp_path, sample_pubmedqa_records) -> Path:
    f = tmp_path / "labeled_train.jsonl"
    with open(f, "w") as fp:
        for r in sample_pubmedqa_records:
            fp.write(json.dumps(r) + "\n")
    return f


# ── Pytest markers ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that load real models (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks end-to-end integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require a CUDA GPU")
