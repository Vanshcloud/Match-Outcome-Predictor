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
  CMD curl -fsS "http://127.0.0.1:${API_PORT:-8000}/health" || exit 1

# `sh -c` with an explicit `exec`, so `$API_PORT` is expanded *and* uvicorn
# still replaces the shell as PID 1 and receives SIGTERM directly. Plain exec
# form does not expand variables, which is how the port came to be hardcoded
# here while the HEALTHCHECK above read the variable — set API_PORT and the app
# served on 8000 while the probe asked the new port and failed forever. Without
# the `exec`, the shell stays PID 1, every stop takes the full grace period and
# ends in a SIGKILL.
#
# The host is fixed at 0.0.0.0 and is deliberately *not* a variable. Binding a
# container to anything narrower is a mistake — the published port is how
# exposure is controlled — and a configurable bind address is one a health
# probe on 127.0.0.1 can be configured out of, which is the failure this change
# exists to remove rather than reintroduce.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"${API_PORT:-8000}\""]

# --- dashboard ---------------------------------------------------------------
# Its own stage rather than its own Dockerfile: it shares the base, the
# non-root user and the mount points with the runtime stage above, and two
# files describing one build is how they drift.
#
# It builds from `requirements-dashboard.txt`, which leaves out the entire
# modelling stack — this process reads report tables and asks the service for a
# probability, and never unpickles an estimator. CI enforces the same rule in
# the source: `dashboard` may not import `api`.
FROM python:3.13-slim-bookworm AS dashboard-builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-dashboard.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install --prefix=/install -r requirements-dashboard.txt

FROM python:3.13-slim-bookworm AS dashboard

LABEL org.opencontainers.image.title="Match Outcome Predictor dashboard" \
      org.opencontainers.image.description="Reliability, per-competition breakdowns and a live fixture price" \
      org.opencontainers.image.source="https://github.com/Vanshcloud/Match-Outcome-Predictor" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models \
    DASHBOARD_PORT=8501 \
    DASHBOARD_API_URL=http://api:8000

RUN apt-get update \
 && apt-get install --no-install-recommends -y curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app

COPY --from=dashboard-builder /install /usr/local

WORKDIR /app
# `api/` is deliberately not copied. The dashboard is a client of that service
# over HTTP, and an image that contained the handlers would be one where the
# rule CI enforces in source could be broken by an import at run time.
COPY --chown=app:app dashboard/ ./dashboard/
COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/ ./configs/

RUN mkdir -p /app/data /app/models && chown -R app:app /app/data /app/models

USER app
EXPOSE 8501

# Streamlit's own readiness endpoint. It answers once the server is accepting
# connections, which is the same question the compose file asks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${DASHBOARD_PORT:-8501}/_stcore/health" || exit 1

# `sh -c` with an explicit `exec`, for the reason the runtime stage above
# records: exec form does not expand variables, and without the `exec` the
# shell stays PID 1 and every stop takes the full grace period.
#
# `--server.address 0.0.0.0` is a literal for the same reason the API's bind
# address is: the published port is how exposure is controlled. Usage
# statistics are off — this is somebody's own machine reading their own data.
CMD ["sh", "-c", "exec streamlit run dashboard/app.py \
--server.port \"${DASHBOARD_PORT:-8501}\" \
--server.address 0.0.0.0 \
--server.headless true \
--browser.gatherUsageStats false"]
