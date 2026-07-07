"""bundle/predict.py — BundlePredictor for HW3_A.

This module is the inference contract for the encoder bundle.
HW3_B imports BundlePredictor from this file at runtime.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BundlePredictor:
    """Loads the frozen model and exposes embed/info methods."""

    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(os.getenv("BUNDLE_DEVICE", "cpu"))
        torch.manual_seed(0)
        self.model, self.tokenizer = load_bundle(self.bundle_dir, self.device)
        self.model.eval()

    def embed(self, texts: List[str]) -> np.ndarray:
        return embed(self.model, self.tokenizer, texts, self.device)

    def info(self) -> dict:
        return info(self.bundle_dir)


def load_bundle(
    model_dir: str | Path, device: torch.device | None = None
) -> Tuple[AutoModel, AutoTokenizer]:
    model_dir = Path(model_dir)
    if device is None:
        device = torch.device(os.getenv("BUNDLE_DEVICE", "cpu"))
    model = AutoModel.from_pretrained(str(model_dir)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return model, tokenizer


def embed(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state

    # Mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    # L2 normalize
    pooled = F.normalize(pooled, p=2, dim=1)

    return pooled.detach().cpu().numpy().astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return float((a_n * b_n).sum())


def info(bundle_dir: Path | None = None) -> dict:
    meta = {}
    if bundle_dir is not None:
        meta_path = Path(bundle_dir).parent / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
    return {
        "model_name": meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        "embedding_dim": meta.get("embedding_dim", 384),
        "max_seq_len": meta.get("max_seq_len", 256),
        "bundle_format": "hw3a/v1",
    }