# HW3 Part A — Classical ML Model Serving

**Author:** Parham Rahimi  
**Date:** July 7, 2026

## Overview

This part wraps my HW02 Random Forest model into a deployable FastAPI service with Docker and Kubernetes manifests.

### What was built

| Component | File | Description |
|-----------|------|-------------|
| Pydantic Schemas | `src/airbnb_serving/schema.py` | `ListingFeatures` (26 fields), `PredictionResponse` |
| Prediction Logic | `src/airbnb_serving/predictor.py` | `predict_single()`, `predict_batch()` |
| FastAPI App | `src/airbnb_serving/app.py` | Lifespan context manager, 3 endpoints |
| Package | `pyproject.toml`, `requirements.txt` | `airbnb-serving` package |
| Docker | `Dockerfile`, `Dockerfile.naive` | Multi-stage (1.33GB) vs naive (3.13GB) |
| Compose | `docker-compose.yml` | Orchestrates optimized image |
| Kubernetes | `k8s/deployment.yaml`, `k8s/service.yaml` | 2 replicas, ClusterIP |

### Endpoints

| Method | Path | Response |
|--------|------|----------|
| GET | `/health` | `{"status":"ok","model_run_id":"..."}` |
| POST | `/predict` | `{"prediction":0,"probability_high_demand":0.083,...}` |
| POST | `/predict/batch` | `[{...}, {...}]` |

### How to run

```bash
cd A/
pip install -e .
docker compose up -d
curl http://localhost:8000/health
```

### MLflow credentials

My personal MLflow account (`student_parham_rahimi`) is not provisioned on the shared server (returns 401). The app tries remote MLflow first, then falls back to a local SQLite database copied from HW2. The full investigation is documented in `reports/mlflow_connectivity_issue.md`.

For testing, the service successfully loaded models using classmate Nazanin Hesari's credentials (with admin permission).

### Docker image sizes

| Tag | Size | Reduction |
|-----|------|-----------|
| naïve (full Python 3.11) | 3.13 GB | — |
| optimized (multi-stage, slim) | 1.33 GB | 58% smaller |

### Screenshots

Available in `screenshots/` — health endpoint, predict, batch, Swagger docs, Docker image sizes.

### Notebook

`01_model_serving_student.ipynb` — executed with all outputs. Includes reproducibility check, batch vs single benchmark, and K8s conceptual answers.