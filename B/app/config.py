"""app.config — environment variable contract for HW3_B."""
from __future__ import annotations

import os
from typing import List

# --- App identity ---
APP_TITLE = "QBC12 HW03-B Encoder Embedding & Search API"
APP_VERSION = "0.1.0"

# --- Bundle location ---
BUNDLE_DIR = os.getenv("BUNDLE_DIR", "/app/bundle")
BUNDLE_DEVICE = os.getenv("BUNDLE_DEVICE", "cpu")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://qbc12-qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qbc12_corpus")

# --- Postgres ---
DATABASE_HOST = os.getenv("DATABASE_HOST", "185.50.38.163")
DATABASE_PORT = os.getenv("DATABASE_PORT", "32112")
DATABASE_NAME = os.getenv("DATABASE_NAME", "qbc12_hw03_encoder")
DATABASE_API_RO_PASSWORD = os.getenv("DATABASE_API_RO_PASSWORD", "")

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL and DATABASE_API_RO_PASSWORD:
    DATABASE_URL = (
        f"postgresql://qbc12_hw03_api_ro:{DATABASE_API_RO_PASSWORD}"
        f"@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    )

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://185.50.38.163:33014")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
STUDENT_USERNAME = os.getenv("STUDENT_USERNAME", MLFLOW_TRACKING_USERNAME or "student_unknown")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")

# --- Search knobs ---
SEARCH_DEFAULT_TOP_K = 10
SEARCH_MAX_TOP_K = 100
SEARCH_MAX_BATCH_TEXTS = 256

# --- Embedding knobs ---
EMBED_MAX_SEQ_LEN = 256
EMBED_BATCH_HARD_CAP = 256
EMBED_DIM = 384

# --- Model info ---
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
EMBEDDING_MAX_SEQ_LEN = 256


def get_qdrant_client(read: bool = True):
    """Lazily import and create QdrantClient. Cached at module level in client_qdrant."""
    from app.client_qdrant import get_client
    return get_client()