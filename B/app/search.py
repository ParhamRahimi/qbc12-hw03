"""app.search — hybrid (Qdrant ANN + PG audit) search orchestration."""
from __future__ import annotations

import time
from typing import List, Optional

from . import client_pg, client_qdrant, config
from .schemas import SearchHit


def hybrid_search(
    query_vector: List[float],
    top_k: int,
    lang: Optional[str] = None,
    primary: Optional[str] = None,
    exclude_neutral: bool = False,
) -> tuple[List[SearchHit], float]:
    t0 = time.perf_counter()

    qdr_hits = client_qdrant.search(
        collection=config.QDRANT_COLLECTION,
        vector=query_vector,
        top_k=top_k,
        lang=lang,
        primary=primary,
        exclude_neutral=exclude_neutral,
    )

    if not qdr_hits:
        return [], (time.perf_counter() - t0) * 1000.0

    ids = [str(h.id) for h in qdr_hits]
    pg_rows = client_pg.fetch_corpus_hits(ids)
    row_map = {r.get("id", ""): r for r in pg_rows}

    hits = []
    for h in qdr_hits:
        row = row_map.get(str(h.id), {})
        hits.append(SearchHit(
            id=str(h.id),
            score=float(h.score),
            text=row.get("text", ""),
            primary=row.get("primary_label", ""),
            labels=list(row.get("labels", [])),
            lang=row.get("lang", ""),
            source=row.get("source", ""),
        ))

    took_ms = (time.perf_counter() - t0) * 1000.0
    return hits, took_ms