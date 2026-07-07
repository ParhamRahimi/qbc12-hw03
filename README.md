# HW03 — Model Serving & Deployment

**Author:** Parham Rahimi  
**Course:** QBC12 MLOps Bootcamp  
**Date:** July 2026

## Project Structure

```
HW3/
├── README.md                   ← this file
├── A/                          ← Part A: Classical ML model serving
│   ├── 01_model_serving_student.ipynb
│   ├── src/airbnb_serving/     ← FastAPI app, schemas, predictor
│   ├── reports/                ← Docker size report, MLflow issue doc
│   ├── screenshots/            ← Endpoint outputs
│   ├── k8s/                    ← K8s deployment + service manifests
│   ├── Dockerfile              ← Multi-stage (optimized, 1.33GB)
│   └── Dockerfile.naive        ← Simple build (3.13GB)
│
└── B/                          ← Part B: Encoder embedding service
    ├── app/                    ← FastAPI + config + clients
    ├── bundle/                 ← all-MiniLM-L6-v2 model + manifest
    ├── Dockerfile              ← Multi-stage CPU-only torch build
    ├── compose.yaml            ← Local dev with Qdrant + Postgres
    ├── tests/                  ← pytest (test_health, test_embed, etc.)
    └── .env                    ← Credentials (gitignored)
```

## Part A — Classical ML Model Serving

Serves my HW02 Random Forest model through a FastAPI REST API:
- **Model:** Random Forest classifier from HW02 (via MLflow)
- **Endpoints:** `/health`, `/predict`, `/predict/batch`
- **Docker:** Multi-stage build, 58% smaller than naive
- **K8s:** Deployment (2 replicas) + ClusterIP Service

### Running Part A

```bash
cd A/
docker compose up -d
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @../test_payload.json
```

### Note on MLflow

My personal MLflow account returns 401 when connecting to the remote server. The app includes a local fallback mechanism (SQLite DB + file store). Classmate Nazanin Hesari's credentials work — see `A/reports/mlflow_connectivity_issue.md` for full details.

---

## Part B — Encoder Embedding & Search

Serves `sentence-transformers/all-MiniLM-L6-v2` for text embeddings:
- **Model:** 384-dim encoder, frozen in `bundle/`
- **Endpoints:** `/`, `/health`, `/model-info`, `/embed`, `/predict`, `/search`
- **Backend:** Qdrant (ANN) + Postgres (source of truth)
- **Docker:** CPU-only multi-stage build

### Running Part B

```bash
cd B/
make build   # Build Docker image
make run     # Start Qdrant + Postgres + API
make smoke   # Test endpoints
make test    # Run pytest
```

Open http://localhost:8000/docs for Swagger UI.

---

## Credentials

All credentials are stored in `.env` files (gitignored). Templates are in `.env.example`.
Shared credentials are in `SHARED.txt` (B/).