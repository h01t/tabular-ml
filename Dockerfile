# ── Stage 1: Build dependencies ──────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build deps in a virtual env for clean copy
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only the runtime dependencies (not dev/notebook deps)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install the project package
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .


# ── Stage 2: Runtime image ──────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy model artifacts (preprocessing pipeline + trained model)
COPY artifacts/preprocessing_pipeline.joblib artifacts/
COPY artifacts/xgboost_model.joblib artifacts/

# Copy config (needed if app references it, but API uses hardcoded paths)
COPY configs/ configs/

# Own everything by appuser
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "tabular_ml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
