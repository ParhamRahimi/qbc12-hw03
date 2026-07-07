"""app.predictor — thin wrapper over the bundle's BundlePredictor."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def embed_texts(predictor, texts: Sequence[str]) -> np.ndarray:
    """Return (B, 384) L2-normalized embeddings."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    arr = predictor.embed(list(texts))
    if arr.shape[1] != 384:
        raise ValueError(f"Expected embedding dim 384, got {arr.shape[1]}")
    return arr.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (a.shape[0], b.shape[0]) cosine similarity matrix."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_n @ b_n.T