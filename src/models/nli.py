"""
NLI (Natural Language Inference) model for hallucination detection.

Uses DeBERTa-v3-base fine-tuned on MedNLI to classify:
  - ENTAILMENT: premise supports hypothesis (claim is grounded)
  - NEUTRAL: premise is unrelated to hypothesis
  - CONTRADICTION: premise contradicts hypothesis (potential hallucination)

Used by HallucinationDetectorAgent to score each generated claim
against retrieved evidence chunks.
"""

import re
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings


@dataclass
class NLIResult:
    premise: str
    hypothesis: str
    label: str          # "entailment", "neutral", "contradiction"
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    is_supported: bool  # True if entailment_score >= threshold

    def confidence(self) -> float:
        """Confidence that the hypothesis IS supported by the premise."""
        return self.entailment_score


class MedNLIModel:
    """
    DeBERTa-v3-base for medical NLI.
    Falls back to cross-encoder/nli-deberta-v3-base if MedNLI fine-tune unavailable.
    """

    # Label order varies by model — we detect dynamically
    _LABEL_MAP_STANDARD = {0: "contradiction", 1: "neutral", 2: "entailment"}
    _LABEL_MAP_INVERTED = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def __init__(self, model_name: Optional[str] = None, threshold: Optional[float] = None):
        self.model_name = model_name or settings.nli_model
        self.threshold = threshold or settings.nli_entailment_threshold
        self.batch_size = settings.nli_batch_size
        self._model = None
        self._tokenizer = None
        self._label_map = self._LABEL_MAP_STANDARD

    def load(self) -> None:
        """Load model and tokenizer. Detects label ordering automatically."""
        logger.info(f"Loading NLI model: {self.model_name}")

        # Try MedNLI-specific model first, then fall back to general NLI
        model_candidates = [
            self.model_name,
            "cross-encoder/nli-deberta-v3-base",
            "typeform/distilbert-base-uncased-mnli",
        ]

        for model_id in model_candidates:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                # Detect label ordering from model config
                self._detect_label_map()
                logger.success(f"NLI model loaded: {model_id}")
                self.model_name = model_id
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_id}: {e}")

        raise RuntimeError("Could not load any NLI model.")

    def _detect_label_map(self) -> None:
        """Infer label→index mapping from model config id2label."""
        if hasattr(self._model.config, "id2label"):
            id2label = self._model.config.id2label
            self._label_map = {int(k): v.lower() for k, v in id2label.items()}
        else:
            self._label_map = self._LABEL_MAP_STANDARD

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _infer_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
    ) -> list[list[float]]:
        """Run a batch of NLI inferences. Returns list of [ent, neut, contra] probabilities."""
        if not self.is_loaded:
            self.load()

        inputs = self._tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for prob_row in probs:
            # Map to [entailment, neutral, contradiction] order
            scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
            for idx, label in self._label_map.items():
                if idx < len(prob_row):
                    scores[label] = float(prob_row[idx])
            results.append([scores["entailment"], scores["neutral"], scores["contradiction"]])

        return results

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify a single premise-hypothesis pair."""
        probs = self._infer_batch([premise], [hypothesis])[0]
        ent, neut, contra = probs
        label = ["entailment", "neutral", "contradiction"][probs.index(max(probs))]
        return NLIResult(
            premise=premise,
            hypothesis=hypothesis,
            label=label,
            entailment_score=ent,
            neutral_score=neut,
            contradiction_score=contra,
            is_supported=ent >= self.threshold,
        )

    def classify_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
    ) -> list[NLIResult]:
        """Batch NLI classification."""
        if not premises:
            return []

        results = []
        for i in range(0, len(premises), self.batch_size):
            batch_p = premises[i : i + self.batch_size]
            batch_h = hypotheses[i : i + self.batch_size]
            probs_list = self._infer_batch(batch_p, batch_h)
            for j, probs in enumerate(probs_list):
                ent, neut, contra = probs
                label = ["entailment", "neutral", "contradiction"][probs.index(max(probs))]
                results.append(NLIResult(
                    premise=batch_p[j],
                    hypothesis=batch_h[j],
                    label=label,
                    entailment_score=ent,
                    neutral_score=neut,
                    contradiction_score=contra,
                    is_supported=ent >= self.threshold,
                ))

        return results

    def score_claim_against_evidence(
        self,
        claim: str,
        evidence_chunks: list[str],
    ) -> dict:
        """
        Score a single claim against multiple evidence chunks.
        Returns the max entailment score across all chunks (best evidence match).
        """
        if not evidence_chunks:
            return {
                "claim": claim,
                "max_entailment": 0.0,
                "is_supported": False,
                "best_evidence": "",
                "label": "neutral",
            }

        premises = evidence_chunks
        hypotheses = [claim] * len(evidence_chunks)
        results = self.classify_batch(premises, hypotheses)

        best = max(results, key=lambda r: r.entailment_score)
        return {
            "claim": claim,
            "max_entailment": best.entailment_score,
            "is_supported": best.is_supported,
            "best_evidence": best.premise[:200],
            "label": best.label,
            "all_scores": [r.entailment_score for r in results],
        }


# ── Claim Extraction Utility ──────────────────────────────────────────────────

_SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+")


def extract_claims(text: str) -> list[str]:
    """
    Split generated text into individual factual claims (sentences).
    Filters out trivially short or non-factual fragments.
    """
    sentences = _SENTENCE_SPLITTER.split(text.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        # Keep sentences that look like factual claims (>20 chars, no pure conjunctions)
        if len(s) > 20 and not s.lower().startswith(("however,", "therefore,", "in summary,")):
            claims.append(s)
    return claims


# ── Singleton ─────────────────────────────────────────────────────────────────

_nli_instance: MedNLIModel | None = None


def get_nli_model() -> MedNLIModel:
    global _nli_instance
    if _nli_instance is None:
        _nli_instance = MedNLIModel()
        _nli_instance.load()
    return _nli_instance
