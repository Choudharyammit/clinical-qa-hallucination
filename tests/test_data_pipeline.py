"""
Unit tests for data pipeline: preprocessor and vectorstore.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.preprocessor import (
    chunk_text,
    clean_text,
    content_hash,
    process_medqa,
    process_pubmed_abstracts,
    process_pubmedqa,
)


# ── clean_text ─────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_strips_html(self):
        assert clean_text("<b>hello</b> world") == "hello world"

    def test_strips_citations(self):
        assert "[1]" not in clean_text("Metformin is effective [1,2] in treatment.")

    def test_normalizes_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_unicode_normalization(self):
        # NFKC should normalize ligatures and special chars
        result = clean_text("café")
        assert result == "café"

    def test_returns_stripped(self):
        assert clean_text("  hello  ") == "hello"


# ── chunk_text ─────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short medical text."
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        text = " ".join(["word"] * 1200)
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
        assert len(chunks) >= 2

    def test_chunks_have_overlap(self):
        words = [f"word{i}" for i in range(600)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
        if len(chunks) >= 2:
            # The end of chunk 0 should partially overlap with start of chunk 1
            end_words = chunks[0].split()[-30:]
            start_words = chunks[1].split()[:30]
            overlap = set(end_words) & set(start_words)
            assert len(overlap) > 0

    def test_empty_string(self):
        assert chunk_text("", 512, 64) == []

    def test_drops_tiny_chunks(self):
        # Tail chunk with < 20 words should be dropped
        text = " ".join(["word"] * 525)  # 512 + 13 leftover
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.split()) >= 20


# ── content_hash ───────────────────────────────────────────────────────────────

class TestContentHash:
    def test_same_text_same_hash(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_text_different_hash(self):
        assert content_hash("hello") != content_hash("world")

    def test_returns_string(self):
        assert isinstance(content_hash("test"), str)


# ── process_pubmed_abstracts ───────────────────────────────────────────────────

class TestProcessPubmedAbstracts:
    def test_basic_ingestion(self, tmp_path):
        input_file = tmp_path / "abstracts.jsonl"
        records = [
            {
                "pmid": "12345",
                "title": "Diabetes Treatment",
                "abstract": " ".join(["word"] * 100),
                "source": "PubMed:12345",
                "year": "2023",
                "journal": "NEJM",
            }
        ]
        with open(input_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        output_file = tmp_path / "chunks.jsonl"
        count = process_pubmed_abstracts(input_file, output_file, chunk_size=50, chunk_overlap=10)

        assert count > 0
        assert output_file.exists()
        with open(output_file) as f:
            chunk = json.loads(f.readline())
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "source" in chunk

    def test_deduplication(self, tmp_path):
        input_file = tmp_path / "dup.jsonl"
        record = {
            "pmid": "99999",
            "title": "Test",
            "abstract": " ".join(["unique"] * 80),
            "source": "PubMed:99999",
        }
        # Write same record twice
        with open(input_file, "w") as f:
            f.write(json.dumps(record) + "\n")
            f.write(json.dumps(record) + "\n")

        output_file = tmp_path / "dedup.jsonl"
        count = process_pubmed_abstracts(input_file, output_file, chunk_size=50, chunk_overlap=0)

        with open(output_file) as f:
            lines = f.readlines()
        # Should not have duplicates
        ids = [json.loads(l)["chunk_id"] for l in lines]
        assert len(ids) == len(set(ids))

    def test_skips_short_abstracts(self, tmp_path):
        input_file = tmp_path / "short.jsonl"
        with open(input_file, "w") as f:
            f.write(json.dumps({"pmid": "1", "title": "T", "abstract": "short", "source": "S"}) + "\n")

        output_file = tmp_path / "out.jsonl"
        count = process_pubmed_abstracts(input_file, output_file)
        assert count == 0


# ── process_medqa ──────────────────────────────────────────────────────────────

class TestProcessMedQA:
    def test_basic_processing(self, tmp_path):
        input_file = tmp_path / "train.jsonl"
        records = [
            {
                "question": "What is the treatment for type 2 diabetes?",
                "options": {"A": "Metformin", "B": "Insulin", "C": "Surgery", "D": "Aspirin"},
                "answer": "Metformin",
                "answer_idx": "A",
            }
        ]
        with open(input_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        output_file = tmp_path / "processed.jsonl"
        count = process_medqa(input_file, output_file)

        assert count == 1
        with open(output_file) as f:
            rec = json.loads(f.readline())
        assert "question" in rec
        assert "options" in rec
        assert isinstance(rec["options"], dict)

    def test_skips_empty_questions(self, tmp_path):
        input_file = tmp_path / "empty.jsonl"
        with open(input_file, "w") as f:
            f.write(json.dumps({"question": "", "options": {}, "answer": ""}) + "\n")

        output_file = tmp_path / "out.jsonl"
        count = process_medqa(input_file, output_file)
        assert count == 0
