# Sentinel — Design

**Date:** 2026-09-03
**Status:** Approved, ready for implementation planning

## 1. Overview

Sentinel is a domain-agnostic complaint intake and triage platform with three
machine learning models assisting human agents: automatic category suggestion,
duplicate detection, and SLA breach risk scoring.

It is a rebuild of an earlier project ("CGMS") whose complaint data lived
entirely in browser `localStorage` with no backend persistence, no models, and
a broken shared JavaScript bundle. Sentinel keeps none of that code.

**Purpose:** a portfolio piece. Optimize for correctness, tests, a public
deployment, and a README that explains its decisions — not for feature count.

## 2. Goals and non-goals

**Goals**

- A complaint lifecycle that multiple users share, backed by a real database.
- Three ML models, each evaluated against a stated baseline with published metrics.
- Two domain packs (CFPB, NYC 311) demonstrating that the platform is not
  coupled to either.
- Deployed publicly within a 512MB memory budget.
- Reproducible training: scripts, pinned artifacts, versioned metadata.

**Non-goals**

- Real-time collaboration, notifications, or messaging.
- File attachments on complaints.
- A separate SPA frontend.
- Multi-tenancy beyond the domain-pack abstraction.
- Anything asynchronous (Celery/Redis) in v1.

## 3. Core principles

These are the invariants. Everything below serves them.

1. **The system understands the concept of a domain, never the meaning of a
   particular domain.** No `if domain == "cfpb"` anywhere in `complaints/`.
2. **The model suggests; the human decides.** ML output is immutable evidence
   recorded alongside the complaint, never written into human-owned fields.
3. **ML degrades to absent, never to broken.** Every model failure — including
   "no artifacts installed" — leaves the application fully functional.
4. **Measure before adding infrastructure.** No vector database, no task queue,
   no microservice until a measurement demands it.

## 4. Architecture

### 4.1 Domain packs

A domain is data in the database; its behavior is code resolved through a registry.

- **Database:** `Domain` and `Category` rows hold taxonomy and configuration.
- **Python:** `PACKS: dict[str, type[DomainPack]]` maps a domain slug to a pack
  class supplying a dataset adapter and a model bundle.

```python
pack = PACKS[complaint.domain.slug]
```

Adding a third domain means adding a pack and rows. It touches no complaint logic.

### 4.2 Inference interfaces

All models sit behind protocols so implementations are swappable:

```python
class TriageModel(Protocol):
    def predict(self, text: str) -> TriagePrediction: ...

class DedupIndex(Protocol):
    def query(self, text: str, k: int) -> list[Match]: ...

class RiskModel(Protocol):
    def predict(self, features: RiskFeatures) -> RiskScore: ...
```

Result objects are frozen dataclasses, domain-independent, and each carries the
`model_version` that produced it.

Three consequences:

- Tests inject deterministic fakes instead of loading ONNX.
- A fresh clone with no artifacts runs on null implementations.
- Moving to a separate ML service later is a new implementation class.

### 4.3 Module layout

```
sentinel/
├── accounts/     auth (allauth + Google), roles, permissions
├── complaints/   Complaint, lifecycle, views, DRF API
├── domains/      Domain/Category models, pack registry
├── ml/
│   ├── base.py       protocols and result dataclasses
│   ├── triage.py     TF-IDF + calibrated linear classifier
│   ├── dedup.py      ONNX MiniLM + cosine search
│   ├── risk.py       histogram gradient boosting
│   ├── registry.py   lazy loading, per-domain bundles, version pinning
│   ├── training/     reproducible training scripts
│   └── artifacts/    versioned model files
├── ingest/       management commands: fetch, normalize, load
└── config/       settings: base / dev / prod
```

### 4.4 Dedup storage and search

Embeddings are stored as raw `float32` bytes in a `BinaryField`, not JSON — a
384-dim vector is 1,536 bytes binary against roughly 9KB as serialized text, and
loads into a numpy matrix without parsing.

Search is brute-force cosine similarity over an in-memory matrix. This is a
deliberately bounded implementation, not a placeholder. `pgvector` slots in
behind the same `DedupIndex` interface when measured latency or memory justifies
it — not at an arbitrary row count.

## 5. Data model

### 5.1 Complaint

```python
class Complaint(models.Model):
    domain       = FK(Domain)
    category     = FK(Category, null=True)     # human-owned ground truth
    priority     = CharField(choices=...)       # human-owned
    status       = CharField(choices=...)
    title, body  = ...
    submitted_by = FK(User)
    assignee     = FK(User, null=True)
    duplicate_of = FK("self", null=True)
    embedding    = BinaryField(null=True)       # float32 bytes
    created_at, triaged_at, due_at, resolved_at
```

`category` and `priority` are never written by a model.

### 5.2 Prediction — immutable evidence

```python
class Prediction(models.Model):
    complaint     = FK(Complaint, related_name="predictions")
    kind          = CharField(choices=["triage", "risk", "dedup"])
    payload       = JSONField()
    model_name    = CharField()
    model_version = CharField()
    created_at    = DateTimeField(auto_now_add=True)
```

`payload` is JSON because the three prediction shapes genuinely differ, but each
shape is defined by an explicit Python dataclass. The database stores JSON; the
application owns the schema. Rows are never updated.

### 5.3 ComplaintEvent — audit trail and acceptance tracking

```python
class ComplaintEvent(models.Model):
    complaint  = FK(Complaint, related_name="events")
    kind       = CharField(choices=["status", "category", "priority",
                                    "assignment", "duplicate"])
    from_value = CharField(null=True)
    to_value   = CharField(null=True)
    actor      = FK(User, null=True)
    prediction = FK(Prediction, null=True)
    note       = TextField(blank=True)
    created_at = DateTimeField(auto_now_add=True)
```

One table answers who changed what, when, from what, to what, and why. When a
decision responds to a suggestion, `prediction` is set: a matching `to_value` is
an acceptance, a differing one an override. The retraining signal is a query
against this table.

### 5.4 Lifecycle

```
SUBMITTED → IN_REVIEW → IN_PROGRESS → RESOLVED → CLOSED
                     ↘ DUPLICATE (terminal, points at canonical)
```

Every transition writes a `ComplaintEvent`.

**Database `CheckConstraint`s:**

1. `status == DUPLICATE` requires `duplicate_of IS NOT NULL`.
2. `duplicate_of != self`.
3. `duplicate_of` may not reference a complaint that is itself a duplicate
   (prevents chains and cycles; enforced at application level with a
   constraint-backed test).

### 5.5 SLA

`due_at = triaged_at + category.sla_hours`.

**`triaged_at` is the moment an agent confirms category and priority** — never
the moment a model produced a prediction. A model predicts instantly; a human
may confirm hours later, and conflating them corrupts every SLA measurement.

Once resolved, `resolved_at > due_at` is a real breach label, which becomes
training data for a Sentinel-specific risk model replacing the 311-trained one.

### 5.6 Roles and permissions

Three roles for v1: **Submitter**, **Agent**, **Admin**.

Authorization uses granular Django permissions (`complaints.triage_complaint`,
`complaints.reassign_complaint`, `complaints.resolve_complaint`,
`domains.manage`, `ml.view_metrics`), with groups as role bundles. No
`if user.role == ...` checks. A Supervisor tier can later be added as a new
grouping of existing permissions without redesigning authorization.

## 6. ML pipeline

### 6.1 The pack contract

Every domain adapter produces one shape. This is the entire coupling surface
between a data source and the platform:

```python
@dataclass(frozen=True)
class RawComplaint:
    external_id: str
    text: str
    category_label: str
    submitted_at: datetime
    resolved_at: datetime | None
    sla_met: bool | None
```

| Source | text | category_label | sla_met |
|---|---|---|---|
| CFPB | consumer narrative | `Product` | `Timely response?` |
| NYC 311 | `Descriptor` | `Complaint Type` | computed from `Closed - Created` |

CFPB supplies the text models (real prose, real human labels). NYC 311 supplies
the SLA model (real resolution times). Both are free, public, and require no key.

### 6.2 Triage

TF-IDF (word 1–2 grams + char 3–5 grams) into a calibrated logistic regression.

Linear on purpose: calibrated probabilities (confidence is shown to agents and
must mean something), single-digit millisecond inference, a few MB, and
interpretable — top contributing terms are surfaced in the UI as the reason for
a suggestion. Below a confidence threshold the model abstains rather than
guessing.

**Evaluation:** macro-F1 (CFPB classes are severely imbalanced), per-class
precision/recall, confusion matrix, and top-3 accuracy since the UI presents a
shortlist. Reported against majority-class and stratified-random baselines.

### 6.3 Dedup

`all-MiniLM-L6-v2` exported to ONNX, 384-dim normalized embeddings, cosine
similarity, top-k above a threshold. ONNX Runtime rather than PyTorch: roughly
90MB against 300–500MB resident.

**This model must earn its place.** It is benchmarked against TF-IDF cosine
similarity, and if it does not win it is cut. CFPB carries no duplicate labels,
so the evaluation set is built by perturbation — paraphrase, truncate, and inject
typos into held-out complaints, then measure recall@k for retrieving the
original. The benchmark result is published in the README either way.

**This benchmark must be labelled honestly.** It measures synthetic
duplicate-retrieval — whether a perturbed copy retrieves its original. It is not
real-world duplicate-detection accuracy, because no labelled duplicate corpus
exists here. The README states that distinction explicitly rather than letting a
recall@k number imply more than it earned.

### 6.4 Risk

Histogram gradient boosting on structured features. Label is SLA breach.

**The features must be domain-independent, which rules out category identity.**
The model trains on NYC 311 categories but serves CFPB complaints, so a
one-hot or target-encoded `category_id` would be meaningless at inference time.
Categories therefore enter only through domain-neutral properties:

| Feature | Why it transfers |
|---|---|
| `category.sla_hours` | a deadline is a deadline in any domain |
| historical mean resolution time for the category | a number, not an identity |
| historical breach rate for the category | a number, not an identity |
| priority rank (normalized 0–1) | ordinal in every domain |
| age at prediction time | domain-free |
| submission hour, weekday | domain-free |
| text length | domain-free |
| queue depth at submission | domain-free |
| assignee open count | domain-free |

This is the §3.1 principle applied to feature engineering: the model sees how a
category *behaves*, never which category it *is*.

**Cross-domain transfer is measured, not assumed.** Designing features to
transfer is a hypothesis; the model trains on NYC 311 and serves CFPB, and that
gap has to be quantified. It can be: CFPB carries its own `Timely response?`
field — an independent SLA label from an unrelated domain. So the protocol is:

1. Train on NYC 311, evaluate in-domain on held-out 311 (the ceiling).
2. Evaluate the same model on CFPB's `Timely response?` labels (the transfer).
3. Train a CFPB-native model as the in-domain reference (the other ceiling).

The gap between (1)/(3) and (2) is the real cost of transfer, and it is published
whatever it shows. A poor result is a legitimate finding about domain shift, not
a failure to hide — and it is the number that determines whether a domain ships
with the shared risk model or needs its own.

**Evaluation:** ROC-AUC, precision@k (it ranks a queue, so the top of the list
matters more than the global curve), and a calibration curve — an uncalibrated
risk score displayed to a human is actively misleading.

### 6.5 Artifacts and versioning

```
ml/artifacts/<domain>/<model>/<version>/
    model.joblib | model.onnx
    metadata.json    # trained_at, git_sha, dataset_hash, metrics, feature_spec
```

The current version per model is pinned in settings; rollback is a config
change. Because `metadata.json` carries measured metrics, the admin model-card
page renders live from the artifact — the deployed model reports its own
performance.

Small sklearn artifacts are committed to git. The MiniLM ONNX file is downloaded
at build time and verified against a SHA256.

### 6.6 Serving

On submit (synchronous, ~25ms total):

```
save complaint (SUBMITTED)
  → embed text (ONNX)
  → triage.predict  → Prediction
  → dedup.query     → Prediction
  → render suggestions and possible duplicates
```

**Risk does not run at submit.** `due_at` depends on the human-confirmed
category, so the risk model fires on the triage-confirm transition. This follows
directly from the `triaged_at` rule in §5.5.

Every model call is wrapped: failure logs, records no prediction, and the
complaint still saves — the same code path as a clone with no artifacts.

## 7. Testing

| Layer | What it covers | Speed |
|---|---|---|
| Unit | lifecycle transitions, check constraints, SLA arithmetic | fast |
| ML consumer | application logic against fake implementations | fast |
| API + permissions | DRF viewsets; permission matrix parametrized over every (role, action) pair **including negatives** | fast |
| End-to-end | submit through resolve | moderate |
| **Model regression** | `pytest -m ml` loads real artifacts, re-runs evaluation, fails if a metric drops below the floor in `metadata.json` | slow, skipped without artifacts |

The model regression suite is what stops a quietly worse retrained model from
shipping.

## 8. Error handling and configuration

- ML failures degrade to absent (§6.6).
- Ingest is idempotent, resumable, and checksummed.
- Structured JSON logging with request IDs; ML log lines carry complaint ID and
  model version.
- `/healthz` reports registry state — which models loaded, at which versions,
  which are running null — so a deployment that lost its artifacts says so
  instead of silently getting worse.

**Configuration**, addressing the previous project's actual failures:

- Split settings: `base` / `dev` / `prod`.
- `SECRET_KEY` from environment, no default.
- `DEBUG` defaults **off**, so a misconfigured deploy fails closed.
- `ALLOWED_HOSTS` from environment.
- **Google OAuth credentials from environment via `SOCIALACCOUNT_PROVIDERS`,
  never from a `SocialApp` database row.** The previous project stored a live
  client secret in a committed-adjacent SQLite file.
- `.env.example` committed; `.env` and `*.sqlite3` gitignored from the first
  commit.

## 9. Deployment

Render free tier (primary target), managed Postgres, gunicorn, whitenoise for
static files. Build downloads and verifies the ONNX artifact, migrates, runs
collectstatic, and seeds demo data so the public URL is not an empty table.

GitHub Actions runs ruff, mypy over `ml/` and `complaints/`, and pytest on every
push.

**Memory is a hypothesis to be measured, not a calculation.** Total RSS includes
Python, Django, numpy, ONNX Runtime, model allocations, and request buffers —
not just the 90MB model. `gunicorn --preload` should let the large C-level
allocations (the ONNX weight arena, numpy data buffers) stay shared
copy-on-write across workers, since those pages are not touched by refcounting.
That expectation gets verified by measuring RSS/PSS under representative load.
**If two workers do not fit comfortably, the answer is one worker.** Worker
count is a measured deployment parameter, not an architectural commitment.

## 10. Deliberately unlocked parameters

These are implementation parameters, adjustable once measured:

1. **Worker count** — benchmark first.
2. **pgvector migration point** — triggered by a latency or memory threshold,
   not a row count.
3. **Confidence and similarity thresholds** — tuned against the evaluation sets.
4. **Whether MiniLM ships at all** — decided by the §6.3 benchmark.

## 11. Out of scope for v1

- LLM-powered summarization and drafted replies. Deferred deliberately: it
  overlaps the triage model and adds an API key plus per-request cost to a
  public deployment. If added later it is scoped narrowly to summarizing long
  complaints and drafting agent responses, degrading gracefully with no key set.
- Supervisor role.
- Asynchronous inference.
- Attachments, notifications, real-time updates.

## 12. README as deliverable

The README is a shipped artifact, not an afterthought — it is what a reviewer
actually reads. It carries the architecture decisions with their reasoning
(numpy before pgvector, linear before transformers for triage, the
MiniLM-vs-TF-IDF benchmark and its result), the live metrics table, and
screenshots.

## 13. Implementation phasing

This design is too large for a single implementation plan. It decomposes into
three phases, each independently demonstrable and each leaving the application
in a working, deployable state.

**Phase 1 — Platform.** Django project, settings split, domain packs, the data
model of §5 with its constraints, the permission matrix, the complaint
lifecycle UI, DRF API, and the full non-ML test suite. ML runs entirely on null
implementations. **Exit criteria:** a human can submit, triage, assign, resolve,
and mark duplicates; deployed publicly; CI green.

**Phase 2 — Data and models.** Ingest commands for both sources, the pack
adapters, training scripts, the three models with their evaluations and
baselines, artifact packaging, and the model regression suite. **Exit criteria:**
published metrics for all three models; the §6.3 MiniLM-vs-TF-IDF benchmark
decided either way.

**Phase 3 — Integration.** Registry wiring, synchronous serving, suggestion and
duplicate UI, the risk-ranked agent queue, the live model card page, `/healthz`
registry reporting, and the memory measurement that fixes worker count.
**Exit criteria:** models visibly assisting a human in the deployed app.

Phase 1 is the subject of the first implementation plan. Phases 2 and 3 get
their own plans, written after the phase before them lands.
