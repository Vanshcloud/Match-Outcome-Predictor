# Security

## What this project handles

The data pipelines read public static CSV from football-data.co.uk over HTTPS
with no key. The optional services add four secrets. Each is read from the
environment, or from Streamlit's own secrets file, and never from
`configs/config.yaml`, which is committed:

| Secret | Used for | Supplied by |
|---|---|---|
| `PREDICTION_LOG_DSN` | The API's PostgreSQL prediction log; carries a database password | Environment |
| `FOOTBALL_DATA_API_KEY` | The dashboard's live fixture and squad feed (football-data.org), sent as a request header | Environment |
| `DASHBOARD_WEBHOOK_URL` | Match-event notifications. The URL itself is the credential | Environment |
| OIDC client secret and cookie secret | Optional Streamlit sign-in | `.streamlit/secrets.toml` (gitignored) |

**Personal data.** The dashboard keeps the leagues and clubs each reader
follows in a local JSON file (`data/dashboard/profiles.json`, or
`DASHBOARD_PROFILE_STORE`). With OIDC configured, a signed-in reader's verified
email is the key their favourites are stored under. Named profiles have no
password: anyone who can open the dashboard can choose any profile.

**The dashboard is unauthenticated by default and is not meant for public
hosting.** `make dashboard` binds it to 127.0.0.1.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/Vanshcloud/Match-Outcome-Predictor/security/advisories/new).
Please do not open a public issue for anything exploitable. A first response
should arrive within a week.

## What is in scope

- The API in `api/`: the request schemas, the prediction path, the served
  artefact loader.
- Secret handling in the dashboard's providers (`dashboard/providers/`) and the
  profile store (`dashboard/domain/store.py`).
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
| No secret in the tree or an image | `.gitignore` and `.dockerignore` both exclude `.env`, `.env.*` and `.streamlit/secrets.toml`; only `.env.example` is committed |
| No credential in logs | Webhook errors keep only the host; urllib3's DEBUG request lines are switched off (`src/utils/logging.py`) |
| No arbitrary object construction from config | `yaml.safe_load`, never `yaml.load` |
| No SQL built from user input | Every value is a bound parameter; identifiers are matched against `^[a-z_][a-z0-9_]*$` |
| No outbound HTTP outside one module | CI invariant: `requests` is reachable only from `src/utils/http.py` |
| No unpinned lint or runtime dependency | `requirements*.txt` plus a committed `uv.lock` |
| Least-privilege CI | `permissions: contents: read` on the workflow |
| Non-root container | `USER app` in the Dockerfile |
