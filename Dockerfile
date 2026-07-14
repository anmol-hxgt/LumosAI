# LumosAI — FastAPI app container
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer so Docker can cache
# this step — rebuilds are much faster when only app code changes, not deps)
# Install torch's CPU-only build first, in its own layer. The default
# PyPI torch wheel bundles CUDA support and is ~500MB+; the CPU-only wheel
# from PyTorch's own index is much smaller and far less likely to drop
# mid-download on a slow/unstable connection. Splitting this into its own
# RUN step also means Docker caches it separately — if a LATER step fails,
# this successful layer won't need to be re-downloaded on retry.
RUN pip install --no-cache-dir --default-timeout=180 --retries=15 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=180 --retries=15 -r requirements.txt

# Copy the actual application code
COPY ingestion/ ./ingestion/
COPY retrieval/ ./retrieval/
COPY agent/ ./agent/
COPY api/ ./api/
COPY repo_indexer.py .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]