# Stage 1 — Builder
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2 — Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin /usr/local/bin/
COPY src/ src/
COPY artifacts/ artifacts/
COPY pyproject.toml ./
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "airbnb_serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
