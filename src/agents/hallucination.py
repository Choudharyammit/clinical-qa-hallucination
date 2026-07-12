"""
HallucinationDetectorAgent — post-generation fact verification.

Pipeline per generated response:
  1. Extract factual claims (sentence segmentation)
  2. Score each claim against retrieved evidence via NLI (DeBERTa-v3)
  3. Flag claims where max entailment score < threshold
  4. Return per-sentence hallucination scores + overall hallucination rate

This is the core research contribution of the system.
"""

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.config import settings
from src.models.nli import MedNLIModel, extract_claims, get_nli_model


@dataclass
class ClaimVerification:
    claim: str
    entailment_score: float
    is_hallucinated: bool
    best_evidence: str
    evidence_source: str = ""
    label: str = "neutral"


@dataclass
class HallucinationReport:
    original_response: str
    claims: list[ClaimVerification] = field(default_factory=list)
    hallucination_rate: float = 0.0
    overall_confidence: float = 0.0
    verified_response: str = ""  # Response with hallucinated claims removed/flagged
    total_claims: int = 0
    hallucinated_count: int = 0
    supported_count: int = 0

    def summary(self) -> dict:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_count,
            "hallucinated_claims": self.hallucinated_count,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "overall_confidence": round(self.overall_confidence, 4),
        }


class HallucinationDetectorAgent:
    """
    Detects hallucinations in LLM-generated clinical responses.

    For each generated answer:
    - Splits into individual factual claims
    - Uses DeBERTa NLI to check each claim against retrieved evidence
    - Claims not entailed by any evidence chunk are flagged as hallucinations
    - Returns a structured HallucinationReport with per-claim scores
    """

    def __init__(
        self,
        nli_model: Optional[MedNLIModel] = None,
        entailment_threshold: Optional[float] = None,
        flag_mode: str = "bracket",  # "bracket", "remove", or "score_only"
    ):
        self._nli = nli_model
        self.threshold = entailment_threshold or settings.nli_entailment_threshold
        self.flag_mode = flag_mode

    def _ensure_nli(self) -> MedNLIModel:
        if self._nli is None:
            self._nli = get_nli_model()
        return self._nli

    def verify(
        self,
        response: str,
        evidence_chunks: list[str],
        evidence_sources: Optional[list[str]] = None,
    ) -> HallucinationReport:
        """
        Verify all claims in a generated response against evidence.

        Args:
            response: The LLM-generated text to verify
            evidence_chunks: Retrieved document texts used as evidence
            evidence_sources: Optional source labels for each chunk

        Returns:
            HallucinationReport with per-claim scores and overall rate
        """
        nli = self._ensure_nli()
        claims = extract_claims(response)
        sources = evidence_sources or [""] * len(evidence_chunks)

        if not claims:
            return HallucinationReport(
                original_response=response,
                verified_response=response,
                hallucination_rate=0.0,
                overall_confidence=1.0,
            )

        if not evidence_chunks:
            # No evidence → cannot verify, mark everything as uncertain
            claim_verifications = [
                ClaimVerification(
                    claim=c,
                    entailment_score=0.0,
                    is_hallucinated=True,
                    best_evidence="",
                    label="neutral",
                )
                for c in claims
            ]
            return HallucinationReport(
                original_response=response,
                claims=claim_verifications,
                hallucination_rate=1.0,
                overall_confidence=0.0,
                verified_response=self._flag_response(response, claim_verifications),
                total_claims=len(claims),
                hallucinated_count=len(claims),
                supported_count=0,
            )

        # Score each claim against all evidence chunks
        claim_verifications = []
        for claim in claims:
            result = nli.score_claim_against_evidence(claim, evidence_chunks)
            best_idx = (
                result["all_scores"].index(max(result["all_scores"]))
                if result.get("all_scores")
                else 0
            )
            best_source = sources[best_idx] if best_idx < len(sources) else ""

            claim_verifications.append(ClaimVerification(
                claim=claim,
                entailment_score=result["max_entailment"],
                is_hallucinated=not result["is_supported"],
                best_evidence=result["best_evidence"],
                evidence_source=best_source,
                label=result["label"],
            ))

        hallucinated = [c for c in claim_verifications if c.is_hallucinated]
        supported = [c for c in claim_verifications if not c.is_hallucinated]

        hallucination_rate = len(hallucinated) / len(claims) if claims else 0.0
        avg_confidence = (
            sum(c.entailment_score for c in claim_verifications) / len(claim_verifications)
            if claim_verifications else 0.0
        )

        verified_response = self._flag_response(response, claim_verifications)

        report = HallucinationReport(
            original_response=response,
            claims=claim_verifications,
            hallucination_rate=hallucination_rate,
            overall_confidence=avg_confidence,
            verified_response=verified_response,
            total_claims=len(claims),
            hallucinated_count=len(hallucinated),
            supported_count=len(supported),
        )

        logger.debug(
            f"Hallucination check: {len(claims)} claims | "
            f"{len(hallucinated)} hallucinated ({hallucination_rate:.1%}) | "
            f"avg confidence: {avg_confidence:.3f}"
        )
        return report

    def _flag_response(
        self,
        response: str,
        claim_verifications: list[ClaimVerification],
    ) -> str:
        """
        Modify the response text to flag hallucinated claims.
        Modes:
          bracket: Wrap flagged claims in [UNVERIFIED: ...]
          remove:  Remove flagged claims entirely
          score_only: Return original (scores are in report)
        """
        if self.flag_mode == "score_only":
            return response

        result = response
        for cv in claim_verifications:
            if not cv.is_hallucinated:
                continue
            if self.flag_mode == "bracket":
                result = result.replace(cv.claim, f"[UNVERIFIED: {cv.claim}]")
            elif self.flag_mode == "remove":
                result = result.replace(cv.claim, "")

        return result.strip()

    def compute_hallucination_rate(
        self,
        responses: list[str],
        evidence_list: list[list[str]],
    ) -> float:
        """
        Compute aggregate hallucination rate over a batch of responses.
        Used by EvaluatorAgent for system-level benchmarking.
        """
        if not responses:
            return 0.0

        total_rate = 0.0
        for response, evidence in zip(responses, evidence_list):
            report = self.verify(response, evidence)
            total_rate += report.hallucination_rate

        return total_rate / len(responses)
