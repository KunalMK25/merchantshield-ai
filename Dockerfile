# MerchantShield AI — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder-node): builds the React frontend into static assets.
#   Stage 2 (runtime):      Python 3.13 runtime; generates the synthetic dataset,
#                           builds feature CSV, installs Python deps, and runs the
#                           FastAPI backend which serves both the API and the built
#                           frontend at the same origin on port 8000.
#
# IMPORTANT — what this image contains and what it doesn't:
#   ✓  Frozen LightGBM model artifact (ml/models/candidate_lgbm_v1.pkl, committed)
#   ✓  Built React frontend (produced in Stage 1)
#   ✓  Generated synthetic dataset and features (produced during image build)
#   ✗  No real transaction data — synthetic data only (see README, Dataset section)
#   ✗  No login — optional shared token for analyst actions only (MS_ANALYST_TOKEN);
#      prototype, not for public exposure as-is
#   ✗  No production WSGI/ASGI tuning. Keep ONE uvicorn worker: the in-app retraining
#      scheduler (MS_RETRAIN_INTERVAL_HOURS) and the file-based model registry assume it.
#
# State that must survive restarts (mounted as volumes by docker-compose.yml):
#   backend/data         audit database
#   ml/registry_store    model registry (contest runs, champion, history)
#   ml/data_store        generated daily batches, retraining log, drift log, alert queue
#
# VERIFIED BY CI: the maintainer's machine has no Docker, so the GitHub Actions workflow builds this image
# and smoke-tests it on every push (see .github/workflows/ci.yml).
#
# Build:   docker build -t merchantshield-ai .
# Run:     docker run -p 8000:8000 merchantshield-ai
# Open:    http://localhost:8000          (dashboard)
#          http://localhost:8000/docs     (API docs)

# ---------------------------------------------------------------------------
# Stage 1: build the frontend
# ---------------------------------------------------------------------------
FROM node:22.12.0-slim AS builder-node

WORKDIR /app/frontend

# Install dependencies first (layer-cached until package-lock.json changes)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

# Copy source and build
COPY frontend/ ./
RUN npm run build
# Result: /app/frontend/dist/


# ---------------------------------------------------------------------------
# Stage 2: Python runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# Keep Python output unbuffered so logs appear immediately in `docker logs`
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system runtime dependencies required by LightGBM.
# libgomp1 provides the GNU OpenMP runtime (libgomp.so.1) which LightGBM's
# compiled C++ extension links against. python:3.13-slim (Debian Bookworm slim)
# does not include it by default, causing an OSError at import/load time.
# Placed before pip install so this layer is cached independently of Python deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer-cached until requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project source (model artifacts, ml/ pipeline, backend/)
# .dockerignore excludes: node_modules, frontend/dist, __pycache__, *.csv, *.db
COPY . .

# Copy the built frontend from Stage 1 into the expected location
# (FastAPI's StaticFiles mount in backend/main.py looks for frontend/dist/)
COPY --from=builder-node /app/frontend/dist ./frontend/dist

# Generate the synthetic dataset and feature CSV.
# These are excluded from Git (large, reproducible) so they must be produced
# here. The generator uses a fixed RNG seed (RNG_SEED=42 in generate_synthetic.py)
# so the output is identical to what CI and the README's local-setup steps produce.
RUN python ml/data/generate_synthetic.py && \
    python ml/features/build_features.py

# Seed a small demo state so the registry, retraining, drift and alert pages are not empty on first view:
# two ordinary simulated days, then one with an injected spending shift (about a minute at build time).
# Disable with: docker build --build-arg SEED_DEMO=0 .
ARG SEED_DEMO=1
RUN if [ "$SEED_DEMO" = "1" ]; then \
        python -m ml.retrain.job --models lightgbm xgboost && \
        python -m ml.retrain.job --models lightgbm xgboost && \
        python -m ml.retrain.job --models lightgbm xgboost --drift amount_scale=2.5 ; \
    fi

# 8000 by default. Hosts such as Render and Hugging Face Spaces choose the port and pass it as $PORT.
EXPOSE 8000

# Report unhealthy if the API stops answering (it answers /health even when degraded)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

# Run the FastAPI backend.
# --host 0.0.0.0 is required inside Docker so the port is reachable from the
# host via the -p 8000:8000 mapping; 127.0.0.1 would only be accessible inside
# the container.
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
