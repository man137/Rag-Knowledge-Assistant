# Multi-stage build: builder installs pinned requirements into a venv, the
# final image copies the venv and the app and runs as a non-root user.
# No secrets are baked in: everything credential-shaped arrives at runtime.

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
LABEL org.opencontainers-image.title="rag-enterprise-agent"
LABEL org.opencontainers-image.description="Hand-built enterprise RAG + agent demo (FastAPI)"
RUN useradd --create-home --shell /bin/bash appuser
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
# .dockerignore keeps .env, caches, indexes and logs out of the image
COPY . .
# Create writable runtime directories and grant ownership to appuser
RUN mkdir -p logs indexes cache && chown -R appuser:appuser /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys, os; \
p = os.environ.get('PORT', '8000'); \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + p + '/health', timeout=4) else 1)" || exit 1
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
