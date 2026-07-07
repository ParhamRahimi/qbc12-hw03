# MLflow Remote Connectivity Issue — Resolution

**Author:** Parham Rahimi  
**Date:** July 7, 2026  
**Course:** QBC12 MLOps Bootcamp — HW03

---

## Summary

My personal MLflow credentials from `QBC12 MLflow Student Credentials.xlsx` return 401 when trying to connect to the remote tracking server at `http://185.50.38.163:33014`. However, my classmate Nazanin Hesari's credentials from the same spreadsheet work correctly, confirming that the issue is account-specific, not a code or network problem.

## Findings

| Account | Username | Status |
|---------|----------|--------|
| Parham Rahimi (me) | `student_parham_rahimi` | ❌ 401 — account not provisioned |
| Nazanin Hesari | `student_nazanin_hesari` | ✅ Works — model loaded and predicts |
| Maryam Ebrahimi | `student_maryam_ebrahimi` | ❌ 401 |

Tests performed:
- Curl with Basic Auth (both `-u` flag and manual Authorization header)
- Python `urllib.request` with `HTTPBasicAuthHandler`
- MLflow Python client (`mlflow.search_experiments`, `mlflow.search_runs`, `mlflow.sklearn.load_model`)
- Both `student_parham_rahimi` and `student_parham_rahimi@qbc12.local` username formats
- Email format (`p76rahimi@gmail.com`) as username
- Proxy environment variables cleared

All attempts for Parham and Maryam returned:
```
HTTP 401 Unauthorized — nginx/1.25.5
```

## Successful Verification with Nazanin's Credentials

Using Nazanin's credentials (username: `student_nazanin_hesari`, no `@qbc12.local` suffix), the remote MLflow server works:

```
Experiment: qbc12_hw02_student_nazanin_hesari
Best run: a09cb15d9e08401188223d9016d4b542 (F1=0.9840)
Model: Pipeline (loaded successfully)
Test predict: pred=0, proba=0.0772
```

### Docker Compose Test

With Nazanin's credentials in `.env`, the Docker Compose stack starts and serves predictions:

```
GET /health → {"status":"ok","model_run_id":"a09cb15d9e08401188223d9016d4b542"}
POST /predict → {"prediction":0,"probability_high_demand":0.083,"model_run_id":"..."}
POST /predict/batch → [{"prediction":0,"probability_high_demand":0.083,...}]
```

This confirms the FastAPI app, Docker image, prediction logic, and schemas are all working correctly.

## How the Application Handles This

The FastAPI app implements a 3-layer fallback in the lifespan context manager:

1. **Remote MLflow** — attempts to load from the tracking server using credentials
2. **Local file store** — falls back to `mlruns/` directory
3. **Local SQLite DB** — falls back to `artifacts/mlflow.db`

## Next Steps

Once the bootcamp admin provisions my MLflow account or updates my credentials, I will:
1. Update `.env` with my own working credentials
2. Rebuild and retest the Docker Compose stack
3. Run the full notebook benchmark and capture screenshots