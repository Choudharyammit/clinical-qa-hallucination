"""
CitationAgent — attaches source citations to every factual claim in the response.

For each verified claim, links it to the most supporting evidence chunk
and formats citations in a clinical/academic style.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.agents.hallucination import ClaimVerification, HallucinationReport


@dataclass
class Citation:
    source: str
    pmid: str
    title: str
    year: str
    journal: str
    relevance_score: float
    supported_claims: list[str] = field(default_factory=list)

    def to_reference_string(self) -> str:
        """Format as Harvard-style reference."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.journal:
            parts.append(self.journal)
        if self.year:
            parts.append(f"({self.year})")
        if self.pmid:
            parts.append(f"PMID: {self.pmid}")
        return ". ".join(parts) if parts else self.source


@dataclass
class CitedResponse:
    text: str                    # Response text with inline citation markers [1], [2]…
    references: list[Citation]   # Ordered list of cited sources
    citation_map: dict[int, str] # {citation_number: source_id}

    def formatted_references(self) -> str:
        lines = ["\n\nReferences:"]
        for i, ref in enumerate(self.references, start=1):
            lines.append(f"[{i}] {ref.to_reference_string()}")
        return "\n".join(lines)

    def full_response(self) -> str:
        return self.text + self.formatted_references()


class CitationAgent:
    """
    Attaches inline citations to a verified response.

    Strategy:
    - For each supported claim, find the best-matching evidence chunk
    - Assign it a citation number [N]
    - Deduplicate citations pointing to the same source
    - Append a formatted reference list
    """

    def attach_citations(
        self,
        response: str,
        hallucination_report: HallucinationReport,
        retrieved_docs: list,  # list[RetrievedDocument | RankedDocument]
    ) -> CitedResponse:
        """
        Add inline citations to a response based on claim verification results.

        Args:
            response: The (optionally flagged) verified response text
            hallucination_report: Output from HallucinationDetectorAgent
            retrieved_docs: Documents retrieved for this query (for metadata)

        Returns:
            CitedResponse with inline [N] markers and reference list
        """
        # Build source metadata lookup from retrieved docs
        source_meta: dict[str, dict] = {}
        for doc in retrieved_docs:
            if hasattr(doc, "source"):
                src = doc.source
                meta = getattr(doc, "metadata", {})
            else:
                src = doc.get("source", "")
                meta = doc.get("metadata", {})
            source_meta[src] = meta

        # Assign citation numbers to unique sources
        citation_map: dict[str, int] = {}  # source → citation number
        citations: list[Citation] = []
        cit_counter = 1

        cited_text = response
        for claim_ver in hallucination_report.claims:
            if claim_ver.is_hallucinated or not claim_ver.best_evidence:
                continue

            src = claim_ver.evidence_source or claim_ver.best_evidence[:50]
            if src not in citation_map:
                citation_map[src] = cit_counter
                meta = source_meta.get(src, {})
                citations.append(Citation(
                    source=src,
                    pmid=meta.get("pmid", ""),
                    title=meta.get("title", ""),
                    year=meta.get("year", ""),
                    journal=meta.get("journal", ""),
                    relevance_score=claim_ver.entailment_score,
                    supported_claims=[claim_ver.claim],
                ))
                cit_counter += 1
            else:
                # Add this claim to existing citation's list
                cit_num = citation_map[src]
                citations[cit_num - 1].supported_claims.append(claim_ver.claim)

            # Insert inline citation marker after the claim sentence
            cit_num = citation_map[src]
            cited_text = cited_text.replace(
                claim_ver.claim,
                f"{claim_ver.claim} [{cit_num}]",
                1,
            )

        return CitedResponse(
            text=cited_text,
            references=citations,
            citation_map={v: k for k, v in citation_map.items()},
        )

    def attach_simple_citations(
        self,
        response: str,
        retrieved_docs: list,
    ) -> CitedResponse:
        """
        Simpler citation attachment when no hallucination report is available.
        Appends all retrieved sources as references without inline markers.
        """
        citations = []
        for i, doc in enumerate(retrieved_docs, start=1):
            if hasattr(doc, "source"):
                src = doc.source
                meta = getattr(doc, "metadata", {})
                # RankedDocument uses cross_encoder_score; RetrievedDocument uses score
                score = getattr(doc, "cross_encoder_score",
                                getattr(doc, "score", 0.0))
            else:
                src = doc.get("source", "")
                meta = doc.get("metadata", {})
                score = doc.get("cross_encoder_score", doc.get("score", 0.0))

            citations.append(Citation(
                source=src,
                pmid=meta.get("pmid", ""),
                title=meta.get("title", ""),
                year=meta.get("year", ""),
                journal=meta.get("journal", ""),
                relevance_score=score,
            ))

        return CitedResponse(
            text=response,
            references=citations,
            citation_map={i: c.source for i, c in enumerate(citations, start=1)},
        )
