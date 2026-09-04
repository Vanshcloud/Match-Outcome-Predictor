# Security

## What this project handles

No personal data, no credentials, no user accounts. The only external input is
static CSV from a public provider, fetched over HTTPS with no key. The one
place a secret can appear is `PREDICTION_LOG_DSN`, which carries a database
password and is read from the environment only — never from
`configs/config.yaml`, which is committed.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/Vanshcloud/Match-Outcome-Predictor/security/advisories/new).
Please do not open a public issue for anything exploitable. A first response
should arrive within a week.

## What is in scope

- The API in `api/`: the request schemas, the prediction path, the served
  artefact loader.
- SQL reaching the database in `src/storage/`.
- The Dockerfile and `docker-compose.yml` as published.

## What is not

- **Model quality.** A badly calibrated probability is a bug, not a
  vulnerability; [docs/MODEL_CARD.md](docs/MODEL_CARD.md) says where the model
  is honest and where it is not, and `docs/EVALUATION.md` says what beats it.
- **Running the compose stack on a public interface.** It ships with a
  development password and no TLS, and says so in the file. It is a local
  reproduction of the deployment, not the deployment.
- **The provider's data.** Anything served by football-data.co.uk is outside
  this repository's control; see [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Practices this repository already enforces

| Property | Enforced by |
|---|---|
| No secret in the tree | `.gitignore` for `.env`; only `.env.example` is committed |
| No arbitrary object construction from config | `yaml.safe_load`, never `yaml.load` |
| No SQL built from user input | Every value is a bound parameter; identifiers are matched against `^[a-z_][a-z0-9_]*$` |
| No outbound HTTP outside one module | CI invariant: `requests` is reachable only from `src/utils/http.py` |
| No unpinned lint or runtime dependency | `requirements*.txt` plus a committed `uv.lock` |
| Least-privilege CI | `permissions: contents: read` on the workflow |
| Non-root container | `USER app` in the Dockerfile |
