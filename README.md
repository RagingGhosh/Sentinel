# Sentinel

Sentinel is a domain-agnostic complaint intake and triage platform. Complaints
come in, a human agent triages and works them, and (from Phase 2 on) three ML
models assist that human: category suggestion, duplicate detection, and SLA
breach risk scoring. It is a from-scratch rebuild of an earlier project
("CGMS") whose complaint data lived entirely in browser `localStorage` with no
backend, no models, and a broken shared JavaScript bundle — Sentinel keeps
none of that code.

This is a portfolio piece. It is optimized for correctness, tests, a public
deployment, and a README that explains its decisions — not for feature count.

## Architectural principles

Everything in this codebase serves four invariants:

1. **The system understands the concept of a domain, never the meaning of a
   particular domain.** There is no `if domain == "cfpb"` anywhere under
   `complaints/`. Domain identity lives in the database (`domains.Domain`,
   `domains.Category`); domain behavior lives in `domains/packs.py`.
2. **The model suggests; the human decides.** ML output is immutable evidence
   (`complaints.Prediction`) recorded alongside a complaint, and it is never
   written into a human-owned field (`Complaint.category`,
   `Complaint.priority`). Every state change goes through
   `complaints/services.py` and writes a `ComplaintEvent`, so the full
   history — including whether a human accepted or overrode a suggestion —
   is always reconstructable.
3. **ML degrades to absent, never to broken.** Every model failure —
   including "no artifacts installed at all" — leaves the application fully
   functional. `/healthz` reports which models are loaded, at which
   versions, and which are running as null implementations, so a deployment
   that lost its artifacts says so instead of silently getting worse.
4. **Measure before adding infrastructure.** No vector database, no task
   queue, no microservice until a measurement demands it.

## Phase status

Phase 1 (this codebase) ships the complete application with **zero ML
models**: the full complaint lifecycle, the permission model, the
server-rendered UI, the REST API, demo seed data, and deployment
configuration. `ML_ARTIFACT_VERSIONS` is empty and every model call
resolves to a null implementation on purpose — the platform has to stand on
its own before any model exists, so that if the ML work in later phases
overruns, what's already here still works.

Phase 2 adds dataset ingest and trains the three models. Phase 3 adds
request-ID log correlation, a risk-ranked queue, and revisits deployment
parameters (worker count, the pgvector migration point) against measured
load.

## Model performance

Not applicable yet. Phase 1 ships no trained models — every prediction path
resolves to an explicit null implementation, and `/healthz` reports `null`
for triage, dedup, and risk on every domain. This section will carry the
live accuracy/precision/recall figures, evaluated against a stated baseline,
once Phase 2 trains real models. No metric is written here until it has
actually been measured.

## Local setup

Requires Python 3.13.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt         # macOS/Linux

cp .env.example .env
# edit .env: at minimum set SECRET_KEY to any non-empty string for local dev

.venv/Scripts/python manage.py migrate
.venv/Scripts/python manage.py bootstrap_groups
.venv/Scripts/python manage.py seed_demo
.venv/Scripts/python manage.py runserver
```

Visit `http://localhost:8000/`. The seed command creates a `demo-agent` user
(Agent group) and a `demo-user` user (Submitter group) along with six
complaints moving through different states — see
`complaints/management/commands/seed_demo.py`. Both `bootstrap_groups` and
`seed_demo` are safe to run repeatedly.

`manage.py` and the test suite both require `SECRET_KEY` in the environment
and deliberately have **no default** — see "Environment variables" below.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | Yes, no default | Django's cryptographic signing key. The app refuses to start without it — a missing secret fails closed rather than falling back to an insecure default. |
| `DEBUG` | No (default `False`) | Django debug mode. Defaults off so a misconfigured deploy fails closed rather than leaking stack traces. |
| `ALLOWED_HOSTS` | Production only | Comma-separated list of hosts Django will serve. |
| `DATABASE_URL` | No (default: local SQLite file) | A `django-environ` database URL, e.g. `postgres://user:pass@host:5432/dbname`. |
| `GOOGLE_CLIENT_ID` | For Google sign-in | OAuth client ID, read via `SOCIALACCOUNT_PROVIDERS` — never stored as a database `SocialApp` row. |
| `GOOGLE_CLIENT_SECRET` | For Google sign-in | OAuth client secret, same as above. |
| `DJANGO_SETTINGS_MODULE` | Production only | Set to `config.settings.prod` in production. Local commands default to `config.settings.dev`; tests default to it via `conftest.py`. |

`.env.example` holds placeholders only. Copy it to `.env` for local
development; `.env` is gitignored and must never be committed. No real
credential appears anywhere in this repository.

## Running tests

```bash
.venv/Scripts/python -m pytest --cov=complaints --cov=domains --cov=ml --cov-report=term-missing
```

CI (`.github/workflows/ci.yml`) additionally runs `ruff check .`,
`ruff format --check .`, `mypy complaints domains ml`, and
`manage.py makemigrations --check --dry-run` on every push and pull request.
The migration check exists specifically so a model change can never reach
production without its migration.

## Deployment

The app deploys to [Render](https://render.com) using `render.yaml` (a
Render Blueprint), `build.sh` as the build command, and gunicorn as the
application server. These steps are manual — they require the Render
account and Google Cloud console credentials of whoever owns the
deployment:

1. **Create the Render service.** In the Render dashboard, choose
   "New +" -> "Blueprint", connect this GitHub repository, and let Render
   read `render.yaml`. It provisions a free-tier managed Postgres database
   (`sentinel-db`) and a web service running
   `gunicorn config.wsgi:application --preload --workers 2`.
2. **Set the environment variables Render prompts for** (declared as
   `sync: false` in `render.yaml`, so nothing sensitive is ever committed):
   - `SECRET_KEY` — generate a fresh random value; do not reuse a
     development key.
   - `ALLOWED_HOSTS` — the Render-assigned domain (and any custom domain),
     comma-separated.
   - `DATABASE_URL` — filled in automatically from the linked `sentinel-db`
     database; no action needed.
   - `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — from an existing Google
     OAuth client (see step 3).
   - `DJANGO_SETTINGS_MODULE` — already set to `config.settings.prod` in
     `render.yaml`; only change it if you know why.
3. **Add the production callback URL to the existing Google OAuth client.**
   In the Google Cloud Console, under the OAuth 2.0 client used for
   `GOOGLE_CLIENT_ID`, add an authorized redirect URI:
   `https://<your-render-domain>/accounts/google/login/callback/`.
4. **Deploy.** Render runs `build.sh` (install dependencies, `collectstatic`,
   `migrate`, `bootstrap_groups`, `seed_demo`) and then starts gunicorn.
   `seed_demo` runs on every boot and is idempotent, so redeploys never
   multiply the demo data.
5. **Verify by hand**, once deployed:
   - `GET /healthz` returns `{"status": "ok", "models": {...: "null"}}` for
     every domain.
   - Google sign-in completes end to end.
   - A normal user can submit a complaint.
   - A user elevated to the Agent group in the Django admin can triage that
     complaint, and `due_at` appears once triaged.

Worker count (2, in `render.yaml`) is provisional. Phase 3 measures RSS
under load and fixes it; if two workers do not fit in the instance's memory,
the answer is one.
