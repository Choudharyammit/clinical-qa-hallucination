"""
DeBERTa NLI fine-tuning on MedNLI-style data.

Fine-tunes microsoft/deberta-v3-base (or a smaller variant) on medical NLI pairs
to improve entailment classification for clinical claims. Used by HallucinationDetectorAgent.

Data sources (in order of preference):
  1. bigbio/mednli (requires MIMIC PhysioNet credentials)
  2. Local JSONL at data/raw/mednli/train.jsonl
  3. Synthetic bootstrap pairs from PubMed chunks (fallback)

Output:
  models/nli_finetuned/   — HuggingFace checkpoint directory

Usage:
    python -m src.models.nli_trainer --epochs 3 --batch-size 16
    python -m src.models.nli_trainer --dry-run   # validate pipeline only
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import settings

# Label mapping used during training and inference
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_mednli_hf(split: str = "train") -> list[dict]:
    """Load MedNLI from HuggingFace (requires MIMIC credentials)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("bigbio/mednli", split=split, trust_remote_code=True)
        records = []
        for row in ds:
            records.append({
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": row["label"],
            })
        logger.info(f"Loaded {len(records)} MedNLI records from HuggingFace ({split})")
        return records
    except Exception as e:
        logger.warning(f"Could not load bigbio/mednli: {e}")
        return []


def load_mednli_local(split: str = "train") -> list[dict]:
    """Load MedNLI from local JSONL (pre-downloaded)."""
    path = settings.data_raw_dir / "mednli" / f"{split}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                records.append({
                    "premise": d.get("sentence1", d.get("premise", "")),
                    "hypothesis": d.get("sentence2", d.get("hypothesis", "")),
                    "label": d.get("gold_label", d.get("label", "neutral")),
                })
    logger.info(f"Loaded {len(records)} MedNLI records from {path}")
    return records


def generate_bootstrap_nli_pairs(n: int = 500) -> list[dict]:
    """
    Generate synthetic NLI pairs from PubMed chunks for cold-start training.

    Creates:
      - Entailment:    chunk A → first sentence of chunk A (always entails)
      - Neutral:       chunk A → first sentence of chunk B (usually neutral)
      - Contradiction: chunk A → negated first sentence of chunk A
    """
    chunks_path = settings.data_processed_dir / "pubmed_chunks.jsonl"
    if not chunks_path.exists():
        logger.warning("No pubmed_chunks.jsonl found — cannot generate bootstrap pairs.")
        return []

    chunks: list[str] = []
    with open(chunks_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                text = d.get("text", "").strip()
                if len(text.split()) > 20:
                    chunks.append(text)

    if len(chunks) < 10:
        return []

    rng = random.Random(settings.random_seed)
    rng.shuffle(chunks)
    records: list[dict] = []

    def first_sentence(text: str) -> str:
        sents = [s.strip() for s in text.split(".") if len(s.strip()) > 15]
        return sents[0] + "." if sents else text[:150]

    def negate(sent: str) -> str:
        replacements = [
            (" is ", " is not "), (" are ", " are not "),
            (" reduces ", " does not reduce "), (" increases ", " does not increase "),
            (" improves ", " does not improve "), (" was ", " was not "),
        ]
        for src, tgt in replacements:
            if src in sent:
                return sent.replace(src, tgt, 1)
        return "It is not the case that " + sent.lower()

    target_per_class = n // 3
    for i in range(min(target_per_class, len(chunks))):
        premise = chunks[i]
        hyp = first_sentence(premise)
        records.append({"premise": premise, "hypothesis": hyp, "label": "entailment"})

    for i in range(min(target_per_class, len(chunks) - 1)):
        j = (i + len(chunks) // 2) % len(chunks)
        records.append({
            "premise": chunks[i],
            "hypothesis": first_sentence(chunks[j]),
            "label": "neutral",
        })

    for i in range(min(target_per_class, len(chunks))):
        hyp = negate(first_sentence(chunks[i]))
        records.append({"premise": chunks[i], "hypothesis": hyp, "label": "contradiction"})

    rng.shuffle(records)
    logger.info(f"Generated {len(records)} bootstrap NLI pairs from PubMed chunks")
    return records


def get_training_data(use_bootstrap_fallback: bool = True) -> tuple[list[dict], list[dict]]:
    """
    Load training + validation data, trying sources in order.
    Returns (train_records, val_records).
    """
    train = load_mednli_local("train") or load_mednli_hf("train")
    val = load_mednli_local("validation") or load_mednli_hf("validation")

    if not train and use_bootstrap_fallback:
        logger.warning("No MedNLI data found — using bootstrap pairs from PubMed.")
        all_pairs = generate_bootstrap_nli_pairs(600)
        split = int(0.85 * len(all_pairs))
        train, val = all_pairs[:split], all_pairs[split:]

    return train, val


# ── Dataset class ──────────────────────────────────────────────────────────────

class NLIDataset:
    """PyTorch Dataset for NLI training pairs."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        import torch
        r = self.records[idx]
        enc = self.tokenizer(
            r["premise"],
            r["hypothesis"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_str = r["label"].lower()
        label = LABEL2ID.get(label_str, 1)  # default neutral
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ── Trainer ────────────────────────────────────────────────────────────────────

def train_nli_model(
    model_name: str = "microsoft/deberta-v3-small",
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    dry_run: bool = False,
) -> Path:
    """
    Fine-tune DeBERTa for sequence classification (3-class NLI).

    Args:
        model_name: Base HuggingFace model to fine-tune.
        output_dir: Where to save the fine-tuned checkpoint.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: AdamW learning rate.
        max_length: Token sequence max length.
        dry_run: Load model and run one batch only (no save).

    Returns:
        Path to saved checkpoint directory.
    """
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    output_dir = output_dir or (settings.base_dir / "models" / "nli_finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {model_name}")

    # Load tokenizer + model
    logger.info("Loading tokenizer and model…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    # Data
    train_data, val_data = get_training_data()
    if not train_data:
        raise RuntimeError("No training data available. See logs above for details.")

    if dry_run:
        train_data = train_data[:batch_size * 2]
        val_data = val_data[:batch_size] if val_data else train_data[:batch_size]
        epochs = 1
        logger.info("Dry run — using minimal data, 1 epoch, no save.")

    train_ds = NLIDataset(train_data, tokenizer, max_length)
    val_ds = NLIDataset(val_data, tokenizer, max_length) if val_data else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training loop
    best_val_acc = 0.0
    logger.info(f"Training: {len(train_data)} samples × {epochs} epochs")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        elapsed = time.time() - epoch_start

        # Validation
        val_acc = 0.0
        if val_loader:
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.argmax(dim=-1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)
            val_acc = v_correct / v_total if v_total > 0 else 0.0

        logger.info(
            f"Epoch {epoch}/{epochs} — loss={total_loss/max(len(train_loader),1):.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({elapsed:.0f}s)"
        )

        # Save best checkpoint
        if not dry_run and val_acc >= best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.success(f"Checkpoint saved → {output_dir} (val_acc={val_acc:.4f})")

    if dry_run:
        logger.info("Dry run complete — no checkpoint saved.")
    else:
        logger.success(f"Training complete. Best val_acc={best_val_acc:.4f}. Checkpoint: {output_dir}")

    return output_dir


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa on MedNLI")
    parser.add_argument("--model", default="microsoft/deberta-v3-small",
                        help="Base HuggingFace model")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pipeline only — no full training")
    args = parser.parse_args()

    settings.ensure_dirs()
    train_nli_model(
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
