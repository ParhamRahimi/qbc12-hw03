# HW3 Part B — Encoder Embedding & Search Service

**Author:** Parham Rahimi  
**Date:** July 7, 2026

## Overview

This part serves `sentence-transformers/all-MiniLM-L6-v2` as a production-ready embedding API with hybrid search over Qdrant + Postgres.

### What was built

| Component | File | Description |
|-----------|------|-------------|
| Bundle | `bundle/` | Frozen model (6 files) + MANIFEST + metadata + predict.py |
| Config | `app/config.py` | Environment variable contract |
| Schemas | `app/schemas.py` | Pydantic models with `extra="forbid"` |
| Model Loader | `app/model_loader.py` | Bundle discovery, SHA-256 manifest verification |
| Predictor | `app/predictor.py` | Thin wrapper over BundlePredictor.embed() |
| Qdrant Client | `app/client_qdrant.py` | Read-only ANN with payload filters |
| PG Client | `app/client_pg.py` | Read-only Postgres for source-of-truth audit |
| Search | `app/search.py` | Hybrid Qdrant + PG orchestration |
| Main | `app/main.py` | FastAPI lifespan, all 6 endpoints |
| Docker | `Dockerfile` | Multi-stage CPU-only build |
| Tests | `tests/` | pytest (health, embed, search, reindex) |

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service root with banner |
| GET | `/health` | Bundle + Qdrant + PG reachability |
| GET | `/model-info` | Bundle metadata + Qdrant vector count |
| POST | `/embed` | Text → 384-dim L2-normalized vectors |
| POST | `/predict` | Single text → predicted emotion label |
| POST | `/search` | Query → Qdrant ANN + PG audit |

### How to run

```bash
cd B/
make build      # Build Docker image
make run        # Start Qdrant + Postgres + API via compose
make smoke      # Test all 6 endpoints with curl
make test       # Run pytest
make load       # Concurrent /embed load test
```

Open http://localhost:8000/docs for Swagger UI.

### Bundle contents

| File | Size | Description |
|------|------|-------------|
| `model/model.safetensors` | 86.7 MB | Frozen transformer weights |
| `model/config.json` | 612 B | Model architecture config |
| `model/tokenizer.json` | 466 KB | Full tokenizer vocabulary |
| `model/vocab.txt` | 232 KB | Word piece vocabulary |
| `model/tokenizer_config.json` | 350 B | Tokenizer parameters |
| `model/special_tokens_map.json` | 112 B | Special token IDs |
| `predict.py` | — | BundlePredictor class |
| `metadata.json` | — | Model metadata |
| `MANIFEST.json` | — | SHA-256 file integrity |

### Credentials

- MLflow: Using classmate Nazanin Hesari's credentials (my account not provisioned)
- Qdrant: Shared API key from `SHARED.txt`
- Postgres: Shared read-only credentials
- MinIO: Nazanin's access key for bundle upload
- All credentials in `.env` (gitignored)

### Docker image

Multi-stage build using `python:3.11-slim` with CPU-only torch (no CUDA bloat). Image size is optimized by installing torch in the builder stage and copying only runtime dependencies to the final image.