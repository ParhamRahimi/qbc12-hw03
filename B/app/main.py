"""app.main — FastAPI entrypoint for HW3_B.

Endpoints:
  GET  /              — service root
  GET  /health        — bundle + Qdrant + PG reachability
  GET  /model-info    — bundle metadata + Qdrant vector count
  POST /embed         — text(s) → 384-dim vectors
  POST /predict       — single text → predicted emotion label
  POST /search        — query → Qdrant ANN + PG audit
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status

from . import client_pg, client_qdrant, config
from . import predictor as predictor_mod
from .model_loader import ModelService
from .schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RootResponse,
    SearchRequest,
    SearchResponse,
)
from .search import hybrid_search

log = logging.getLogger("hw3_b")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("HW3_B starting. BUNDLE_DIR=%s", config.BUNDLE_DIR)
    model_service.load()
    if model_service.state.loaded:
        log.info("Bundle loaded: %s", model_service.state.bundle_dir)
    else:
        log.error("Bundle load FAILED: %s", model_service.state.error)
    yield
    log.info("HW3_B shutting down.")


app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/", response_model=RootResponse, tags=["service"])
def root():
    return RootResponse(
        message="QBC12 HW3 Encoder API",
        docs="/docs",
        health="/health",
        version=config.APP_VERSION,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    bundle_ok = model_service.state.loaded
    qdrant_ok = client_qdrant.ping()
    pg_ok = client_pg.ping()

    all_ok = bundle_ok and qdrant_ok and pg_ok
    any_ok = bundle_ok or qdrant_ok or pg_ok

    if all_ok:
        st = "ok"
    elif any_ok:
        st = "degraded"
    else:
        st = "error"

    error_msg = None
    if not all_ok:
        parts = []
        if not bundle_ok:
            parts.append(f"bundle: {model_service.state.error or 'not loaded'}")
        if not qdrant_ok:
            parts.append("qdrant unreachable")
        if not pg_ok:
            parts.append("pg unreachable")
        error_msg = "; ".join(parts)

    return HealthResponse(
        status=st,
        bundle_loaded=bundle_ok,
        bundle_dir=str(model_service.state.bundle_dir) if model_service.state.bundle_dir else "",
        qdrant_reachable=qdrant_ok,
        pg_reachable=pg_ok,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info():
    if not model_service.state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    meta = model_service.metadata
    return ModelInfoResponse(
        bundle_version=meta.get("bundle_version", "0.1.0"),
        model_id=meta.get("model_name", config.EMBEDDING_MODEL_ID),
        model_revision=meta.get("model_revision", "unknown"),
        device=config.BUNDLE_DEVICE,
        max_seq_len=meta.get("max_seq_len", config.EMBEDDING_MAX_SEQ_LEN),
        embedding_dim=meta.get("embedding_dim", config.EMBEDDING_DIM),
        bundle_dir=str(model_service.state.bundle_dir) if model_service.state.bundle_dir else "",
        qdrant_collection=config.QDRANT_COLLECTION,
        qdrant_vector_count=client_qdrant.vector_count(config.QDRANT_COLLECTION),
    )


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


@app.post("/embed", response_model=EmbedResponse, tags=["embedding"])
def embed(req: EmbedRequest):
    if not model_service.state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(req.texts) > config.EMBED_BATCH_HARD_CAP:
        raise HTTPException(status_code=413, detail=f"Batch size exceeds hard cap of {config.EMBED_BATCH_HARD_CAP}")

    t0 = time.perf_counter()
    vectors = predictor_mod.embed_texts(model_service.require_predictor(), req.texts)
    elapsed = (time.perf_counter() - t0) * 1000
    log.info("Embedded %d texts in %.1f ms", len(req.texts), elapsed)

    return EmbedResponse(
        count=len(req.texts),
        dim=int(vectors.shape[1]),
        embeddings=vectors.tolist(),
    )


# ---------------------------------------------------------------------------
# /predict — single text → emotion label via nearest neighbor
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictResponse, tags=["embedding"])
def predict(req: PredictRequest):
    if not model_service.state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    vec = predictor_mod.embed_texts(model_service.require_predictor(), [req.text])
    vec_list = vec[0].tolist()

    qc = client_qdrant.get_client()
    hits = qc.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=vec_list,
        limit=1,
    )

    if not hits:
        raise HTTPException(status_code=404, detail="No match found in corpus")

    best = hits[0]
    elapsed = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        text=req.text,
        predicted_label=best.payload.get("primary_label", "unknown") if best.payload else "unknown",
        confidence=float(best.score),
        matched_text=best.payload.get("text", "") if best.payload else "",
        elapsed_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest):
    if not model_service.state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    query_vec = predictor_mod.embed_texts(model_service.require_predictor(), [req.query])
    query_vec_list = query_vec[0].tolist()

    hits, took_ms = hybrid_search(
        query_vec_list,
        req.top_k,
        req.lang,
        req.primary,
        req.exclude_neutral,
    )

    return SearchResponse(
        query=req.query,
        count=len(hits),
        top_k=req.top_k,
        took_ms=took_ms,
        hits=hits,
    )