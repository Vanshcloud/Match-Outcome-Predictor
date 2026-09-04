# The serving image. Two stages: one that resolves wheels, one that runs.
#
# The split is not cosmetic. Building scipy and psycopg pulls a compiler
# toolchain, and a toolchain in a running container is a toolchain an attacker
# can use — so the wheels are built in a stage that is thrown away and only the
# installed packages are copied forward.
#
# What is NOT in this image is as deliberate as what is. No data, no artefact,
# no training stack: `requirements-api.txt` says which libraries were left out
# and why, and `data/` and `models/` are mounted at run time. An image with a
# model baked in is an image that has to be rebuilt to retrain, and one with
# 300k matches inside is 400 MB of bytes that change every match week.

# --- builder -----------------------------------------------------------------
# Pinned to a digest-stable tag rather than `latest`: an image whose base moves
# under it is an image whose contents nobody can reproduce.
FROM python:3.13-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential for anything without a wheel for this platform; libpq-dev is
# not needed because psycopg[binary] ships its own libpq.
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-api.txt ./
# Into a prefix rather than the system tree, so the runtime stage copies one
# directory and inherits nothing else from the builder.
RUN python -m pip install --upgrade pip \
 && python -m pip install --prefix=/install -r requirements-api.txt

# --- runtime -----------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Match Outcome Predictor API" \
      org.opencontainers.image.description="Calibrated home/draw/away probabilities for 39 professional football competitions" \
      org.opencontainers.image.source="https://github.com/Vanshcloud/Match-Outcome-Predictor" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# curl is here for the HEALTHCHECK below and nothing else. A container that
# declares a health check it has no way to run reports healthy forever.
RUN apt-get update \
 && apt-get install --no-install-recommends -y curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app

COPY --from=builder /install /usr/local

WORKDIR /app
# Only what the service reads. `scripts/`, `tests/` and `docs/` are not copied:
# they are not on any request path, and every file in an image is a file
# somebody has to account for.
COPY --chown=app:app api/ ./api/
COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/ ./configs/

# Mount points. Created here so they exist and are writable by `app` even when
# nothing is mounted over them — the service starts degraded rather than
# failing on a directory it cannot read.
RUN mkdir -p /app/data /app/models && chown -R app:app /app/data /app/models

USER app
EXPOSE 8000

# The same question the load balancer asks. `/health` is 200 while the process
# is alive and reports `degraded` when it has no model, which is what makes it
# a liveness probe; readiness is the `status` field, and the compose file gates
# on it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${API_PORT}/health" || exit 1

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly. Through a shell
# it would not, and every stop would take the full grace period and then a
# SIGKILL.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
