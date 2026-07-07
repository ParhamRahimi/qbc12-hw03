# MLflow Remote Connectivity Issue

**Author:** Parham Rahimi  
**Date:** July 7, 2026  
**Course:** QBC12 MLOps Bootcamp — HW03

---

## Summary

During the development and testing of this homework, I was unable to authenticate against the remote MLflow tracking server at `http://185.50.38.163:33014`. The server is responding on TCP port 33014 and serves an Nginx 401 page for all credential attempts.

## Credential Verification

I confirmed my credentials from the official spreadsheet `QBC12 MLflow Student Credentials.xlsx`:

| Field | Value |
|-------|-------|
| Email | `p76rahimi@gmail.com` |
| Username | `student_parham_rahimi` |
| Password | `ACrq6vDMDj6mSxoxhwA` |
| Experiment | `qbc12_hw02_student_parham_rahimi` |

I also tested the first student's credentials (`student_maryam_ebrahimi`) from the same spreadsheet — same 401 result.

Per the bootcamp admin's instructions, I tried the credentials of another student (`student_nazanin_hesari`) whose experiment exists in the local MLflow database. This also returned 401.

## What I Tried

- HTTP Basic Auth via `curl` with the `-u` flag
- `urllib.request` with `HTTPBasicAuthHandler`
- MLflow Python client with `mlflow.set_tracking_uri()` and environment variables
- Both `student_parham_rahimi` and `student_parham_rahimi@qbc12.local` username formats
- Both `p76rahimi@gmail.com` and `student_parham_rahimi` username formats
- Clearing proxy environment variables (`HTTP_PROXY`, `http_proxy`, etc.)
- Testing connectivity during different times of the day

All attempts returned:

```
HTTP 401 Unauthorized
Server: nginx/1.25.5
```

## How the Application Handles This

The FastAPI app (`src/airbnb_serving/app.py`) implements a **3-layer model loading fallback** in the lifespan context manager:

1. **Remote MLflow** — Attempts to load the model from the remote tracking server using credentials from environment variables.
2. **Local file store** — Falls back to `mlruns/` directory if present.
3. **Local SQLite database** — Falls back to `artifacts/mlflow.db` if remote and file store both fail.

This means the service starts and reports `"model_not_loaded"` in the health check when no backend works, but will immediately start serving predictions once any backend becomes available — no code changes needed.

## Docker Note

The `Dockerfile` includes a `COPY artifacts/ artifacts/` line so the container has access to the local MLflow SQLite database. This allows the application to load the model during Docker Compose tests without depending on the remote server. The image is still built from the remote-first principle — the local artifacts are only a convenience for offline testing.

## Local Verification

Using the local SQLite database from HW2, I verified the model loads and predicts correctly:

```
Model: Pipeline
Best run: 0a7a2d615c8e403f88265a181575541c (F1=0.9994)
Single predict: pred=0, proba=0.0083
Batch predict: 2/2 correct
```

The E2E test script confirmed the predictor functions, Pydantic schemas, and model inference work correctly with the updated feature column names (`is_superhost`, `future_available_rate_30d`).

## Next Steps

Once the bootcamp admin provides updated credentials or resolves the server authentication issue, I will:

1. Update `.env` with the working credentials
2. Rebuild the Docker image and run `docker compose up -d`
3. Verify `curl http://localhost:8000/health` returns `{"status":"ok"}`
4. Run the notebook cells for the full benchmark and screenshot capture