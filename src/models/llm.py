"""
LLM loading and inference for the Clinical QA System.

Supports:
  - BioMistral-7B-DARE via HuggingFace (4-bit quantized, bitsandbytes)
  - Ollama local inference (llama3-med42 / llava-med) as fallback
  - CPU-only mode with smaller models (Phi-2, BioGPT) for low-resource environments

Priority: BioMistral → Ollama → BioGPT fallback
"""

import os
import time
from typing import Optional

import torch
from loguru import logger

from src.config import settings


def _best_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' — best available for this hardware."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Clinical QA prompt template following chain-of-thought style
CLINICAL_PROMPT_TEMPLATE = """You are a clinical decision support assistant with expertise in evidence-based medicine. Answer the following medical question using ONLY the provided context. If the context does not contain sufficient information, explicitly state that you cannot answer based on available evidence.

Context:
{context}

Question: {question}

Instructions:
- Base your answer strictly on the provided context
- Cite specific details from the context
- If multiple options are given, reason through each
- State your confidence level (High/Medium/Low)
- Flag any information gaps

Answer:"""

CLINICAL_MCQ_TEMPLATE = """You are a medical expert taking a USMLE-style examination. Use the provided context to answer the multiple choice question.

Context:
{context}

Question: {question}

Options:
{options}

Reasoning: Think step by step through each option using the context provided.

Answer (letter only):"""


class LLMInferenceError(Exception):
    """Raised when LLM inference fails after all retries."""
    pass


class BioMistralLLM:
    """
    BioMistral-7B-DARE with 4-bit quantization via bitsandbytes.
    Requires ~6GB VRAM (4-bit) or ~12GB RAM (CPU offload).
    """

    def __init__(
        self,
        model_name: str = "",
        quantization: str = "4bit",
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.model_name = model_name or settings.llm_model_name
        self.quantization = quantization or settings.llm_quantization
        self.device = device or settings.llm_device
        self.max_new_tokens = max_new_tokens or settings.llm_max_new_tokens
        self.temperature = temperature or settings.llm_temperature
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load model and tokenizer. Call once before inference."""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        logger.info(f"Loading {self.model_name} ({self.quantization} quantization)…")

        bnb_config = None
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            token=settings.huggingface_hub_token or None,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            token=settings.huggingface_hub_token or None,
            torch_dtype=torch.float16 if bnb_config is None else None,
        )
        self._model.eval()

        # Set seeds for reproducibility
        torch.manual_seed(settings.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.torch_seed)

        logger.success(f"Model loaded: {self.model_name}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> str:
        """Generate text completion for a prompt."""
        if not self.is_loaded:
            self.load()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
            padding=False,
        ).to(self._model.device)

        max_tok = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=temp,
                do_sample=do_sample and temp > 0,
                top_p=top_p,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode only the newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def answer_clinical_question(
        self,
        question: str,
        context: str,
        mcq_options: Optional[dict[str, str]] = None,
    ) -> dict:
        """High-level clinical QA with timing and structured output."""
        start = time.perf_counter()

        if mcq_options:
            options_str = "\n".join(f"  {k}. {v}" for k, v in mcq_options.items())
            prompt = CLINICAL_MCQ_TEMPLATE.format(
                context=context[:3000],
                question=question,
                options=options_str,
            )
        else:
            prompt = CLINICAL_PROMPT_TEMPLATE.format(
                context=context[:3000],
                question=question,
            )

        answer = self.generate(prompt)
        elapsed = time.perf_counter() - start

        return {
            "answer": answer,
            "prompt_tokens": len(self._tokenizer.encode(prompt)),
            "response_time_s": round(elapsed, 3),
            "model": self.model_name,
        }


class OllamaLLM:
    """
    Fallback LLM using Ollama local inference server.
    Requires Ollama running at localhost:11434.
    Recommended model: llama3.1:8b or medllama3.
    """

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._client = None

    def load(self) -> None:
        import httpx
        self._client = httpx.Client(base_url=self.base_url, timeout=120.0)
        # Verify Ollama is reachable
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Ollama available. Models: {models}")
        except Exception as e:
            raise RuntimeError(f"Ollama not reachable at {self.base_url}: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        if not self.is_loaded:
            self.load()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_new_tokens},
        }
        resp = self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def answer_clinical_question(
        self,
        question: str,
        context: str,
        mcq_options: Optional[dict[str, str]] = None,
    ) -> dict:
        start = time.perf_counter()
        if mcq_options:
            options_str = "\n".join(f"  {k}. {v}" for k, v in mcq_options.items())
            prompt = CLINICAL_MCQ_TEMPLATE.format(
                context=context[:3000],
                question=question,
                options=options_str,
            )
        else:
            prompt = CLINICAL_PROMPT_TEMPLATE.format(
                context=context[:3000],
                question=question,
            )
        answer = self.generate(prompt)
        elapsed = time.perf_counter() - start
        return {
            "answer": answer,
            "response_time_s": round(elapsed, 3),
            "model": f"ollama/{self.model}",
        }


class BioGPTFallback:
    """
    Fallback LLM: microsoft/BioGPT-Large (CPU/MPS friendly).
    Falls back to BioGPT base if Large doesn't fit in memory.
    """

    def __init__(self):
        self.model_name = settings.llm_model_name if "BioGPT" in settings.llm_model_name else "microsoft/BioGPT-Large"
        self._pipeline = None

    def load(self) -> None:
        from transformers import pipeline as hf_pipeline
        device = _best_device()
        device_idx = 0 if device == "cuda" else (-1 if device == "cpu" else device)

        candidates = [self.model_name, "microsoft/BioGPT-Large", "microsoft/BioGPT"]
        for model_id in candidates:
            try:
                logger.info(f"Loading {model_id} on {device}…")
                self._pipeline = hf_pipeline(
                    "text-generation",
                    model=model_id,
                    device=device_idx,
                )
                self.model_name = model_id
                logger.success(f"{model_id} loaded on {device}.")
                return
            except Exception as e:
                logger.warning(f"Could not load {model_id}: {e}")
        raise RuntimeError("No BioGPT model could be loaded.")

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        if not self.is_loaded:
            self.load()
        max_tok = min(max_new_tokens, settings.llm_max_new_tokens)
        out = self._pipeline(
            prompt,
            max_new_tokens=max_tok,
            do_sample=False,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()

    def answer_clinical_question(
        self,
        question: str,
        context: str,
        mcq_options: Optional[dict[str, str]] = None,
    ) -> dict:
        start = time.perf_counter()
        prompt = f"Context: {context[:1500]}\nQuestion: {question}\nAnswer:"
        answer = self.generate(prompt, max_new_tokens=200)
        return {
            "answer": answer,
            "response_time_s": round(time.perf_counter() - start, 3),
            "model": self.model_name,
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm(prefer: str = "biomistral"):
    """
    Factory that returns the best available LLM.
    Falls back gracefully: BioMistral → Ollama → BioGPT.

    If LLM_MODEL_NAME in settings points to a BioGPT/small model, skip straight
    to BioGPTFallback so we don't waste time trying to 4-bit-quantize it.
    """
    _biogpt_models = ("biogpt", "microsoft/biogpt")
    llm_name = settings.llm_model_name.lower()

    # Short-circuit: if the configured model is a BioGPT variant, skip BioMistral
    skip_biomistral = any(m in llm_name for m in _biogpt_models)

    if (prefer == "biomistral" or prefer == "auto") and not skip_biomistral:
        try:
            llm = BioMistralLLM(model_name="BioMistral/BioMistral-7B-DARE")
            llm.load()
            return llm
        except Exception as e:
            logger.warning(f"BioMistral load failed: {e}. Trying Ollama…")

    if prefer in ("ollama", "auto"):
        try:
            llm = OllamaLLM()
            llm.load()
            return llm
        except Exception as e:
            logger.warning(f"Ollama unavailable: {e}. Falling back to BioGPT…")

    logger.info("Loading BioGPT fallback.")
    llm = BioGPTFallback()
    llm.load()
    return llm
