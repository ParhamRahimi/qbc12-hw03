"""app.client_qdrant — Qdrant client (read-only API key)."""
from __future__ import annotations

import time
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from . import config

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=10.0,
        )
    return _client


def ping() -> bool:
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def vector_count(collection: str) -> Optional[int]:
    try:
        info = get_client().get_collection(collection_name=collection)
        return info.vectors_count
    except Exception:
        return None


def search(
    collection: str,
    vector: List[float],
    top_k: int,
    lang: Optional[str] = None,
    primary: Optional[str] = None,
    exclude_neutral: bool = False,
) -> List[models.ScoredPoint]:
    must = []
    must_not = []
    if lang:
        must.append(models.FieldCondition(key="lang", match=models.MatchValue(value=lang)))
    if primary:
        must.append(models.FieldCondition(key="primary_label", match=models.MatchValue(value=primary)))
    if exclude_neutral:
        must_not.append(models.FieldCondition(key="primary_label", match=models.MatchValue(value="neutral")))

    qfilter = None
    if must or must_not:
        qfilter = models.Filter(must=must, must_not=must_not)

    return get_client().search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
        with_vectors=False,
    )