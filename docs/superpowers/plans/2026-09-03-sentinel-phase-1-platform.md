# Sentinel Phase 1 — Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully working, deployed complaint management platform where humans can submit, triage, assign, resolve and de-duplicate complaints — with the ML layer present as interfaces backed by null implementations, so no model is required for the application to function.

**Architecture:** A Django project split into four apps. `domains/` holds the domain-as-data taxonomy plus a code-side pack registry. `complaints/` holds the complaint lifecycle, where human-owned fields are the ground truth and model output is separate immutable evidence. `ml/` defines protocols and frozen result dataclasses with null implementations. `accounts/` holds auth and a permission-based authorization model. All state changes flow through a service layer that writes an audit event for every transition.

**Tech Stack:** Python 3.13, Django 5.2 LTS, Django REST Framework, django-allauth (Google), PostgreSQL (prod) / SQLite (dev), pytest + pytest-django + factory_boy, ruff, mypy, gunicorn + whitenoise, GitHub Actions, Render.

**Spec:** `docs/superpowers/specs/2026-09-03-sentinel-design.md`

## Global Constraints

These apply to every task. Each task's requirements implicitly include this section.

- **No `if domain == "<slug>"` logic anywhere in `complaints/`.** Domain-specific behavior enters only through the pack registry.
- **Models never write `Complaint.category` or `Complaint.priority`.** Those are human-owned. Model output goes to `Prediction`.
- **`Prediction` rows are never updated or deleted.** Immutable evidence.
- **ML failure degrades to absent, never to broken.** Any model call that raises must log and leave the complaint saved.
- **`DEBUG` defaults to `False`.** A missing env var fails closed, never open.
- **`SECRET_KEY` has no default.** Missing it raises at startup.
- **No secret is ever written to the database.** OAuth credentials come from environment variables via `SOCIALACCOUNT_PROVIDERS`.
- **`triaged_at` is set only by human confirmation of category and priority** — never by a prediction.
- **Every state change writes a `ComplaintEvent`.** No exceptions.
- **TDD:** every task writes a failing test first, watches it fail, then implements.
- Python 3.13. Django 5.2 LTS. Line length 100. `ruff` and `mypy` clean on `complaints/`, `domains/`, `ml/`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `config/settings/base.py` | Shared settings; reads env |
| `config/settings/dev.py` | Local overrides, SQLite |
| `config/settings/prod.py` | Postgres, security headers, whitenoise |
| `config/urls.py` | Root URL conf |
| `domains/models.py` | `Domain`, `Category` |
| `domains/packs.py` | `DomainPack` protocol, `PACKS`, `get_pack()` |
| `ml/base.py` | Protocols + frozen result dataclasses |
| `ml/null.py` | Null implementations |
| `ml/registry.py` | Resolution, version pinning, `registry_status()` |
| `complaints/models.py` | `Complaint`, `Prediction`, `ComplaintEvent`, choices |
| `complaints/services.py` | Lifecycle transitions, triage, assignment, duplicates |
| `complaints/api.py` | DRF serializers + viewsets |
| `complaints/views.py` | Server-rendered views |
| `complaints/permissions.py` | Group definitions and bootstrap |
| `accounts/` | allauth config, signup role assignment |
| `tests/` | Mirrors app structure |

---

### Task 1: Project scaffolding and settings split

**Files:**
- Create: `pyproject.toml`, `.env.example`, `manage.py`, `config/__init__.py`, `config/settings/__init__.py`, `config/settings/base.py`, `config/settings/dev.py`, `config/settings/prod.py`, `config/urls.py`, `config/wsgi.py`, `conftest.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: nothing
- Produces: a Django project importable as `config.settings.dev` / `config.settings.prod`; env var contract `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, `DATABASE_URL`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`

- [ ] **Step 1: Create the virtualenv and install dependencies**

```bash
cd /c/sentinel
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/pip install "django>=5.2,<6.0" djangorestframework django-allauth django-environ \
    psycopg2-binary gunicorn whitenoise \
    pytest pytest-django pytest-cov factory_boy ruff mypy django-stubs
.venv/Scripts/pip freeze > requirements.txt
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_settings.py`:

```python
import importlib
import pytest
from django.core.exceptions import ImproperlyConfigured


def test_secret_key_has_no_default(monkeypatch):
    """A missing SECRET_KEY must fail loudly, not fall back to a shipped value."""
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(ImproperlyConfigured):
        importlib.reload(importlib.import_module("config.settings.base"))


def test_debug_defaults_off(monkeypatch):
    """A misconfigured deploy must fail closed."""
    monkeypatch.setenv("SECRET_KEY", "test-key")
    monkeypatch.delenv("DEBUG", raising=False)
    module = importlib.reload(importlib.import_module("config.settings.base"))
    assert module.DEBUG is False


def test_oauth_credentials_come_from_environment(monkeypatch):
    """Credentials must never be sourced from the database."""
    monkeypatch.setenv("SECRET_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "id-from-env")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret-from-env")
    module = importlib.reload(importlib.import_module("config.settings.base"))
    app = module.SOCIALACCOUNT_PROVIDERS["google"]["APP"]
    assert app["client_id"] == "id-from-env"
    assert app["secret"] == "secret-from-env"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 4: Write `config/settings/base.py`**

```python
from pathlib import Path

import environ
from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent.parent

env = environ.Env()
env_file = BASE_DIR / ".env"
if env_file.exists():
    env.read_env(str(env_file))

try:
    SECRET_KEY = env("SECRET_KEY")
except environ.ImproperlyConfigured as exc:
    raise ImproperlyConfigured("SECRET_KEY environment variable is required") from exc

DEBUG = env.bool("DEBUG", default=False)
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=[])

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
    "rest_framework",
    "domains",
    "complaints",
    "accounts",
]
SITE_ID = 1

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]

# Credentials come from the environment. Never from a SocialApp database row.
SOCIALACCOUNT_PROVIDERS = {
    "google": {
        "APP": {
            "client_id": env("GOOGLE_CLIENT_ID", default=""),
            "secret": env("GOOGLE_CLIENT_SECRET", default=""),
            "key": "",
        },
        "SCOPE": ["profile", "email"],
        "AUTH_PARAMS": {"access_type": "online", "prompt": "select_account"},
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_REDIRECT_URL = "/complaints/"
LOGOUT_REDIRECT_URL = "/"

# Model artifact versions pinned per domain/model. Phase 1 ships none.
ML_ARTIFACT_VERSIONS: dict[str, dict[str, str]] = {}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {"format": '{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'}
    },
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "json"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
```

- [ ] **Step 5: Write `config/settings/dev.py` and `config/settings/prod.py`**

`config/settings/dev.py`:

```python
from .base import *  # noqa: F403
from .base import BASE_DIR, env

DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

DATABASES = {"default": env.db_url("DATABASE_URL", default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}")}
```

`config/settings/prod.py`:

```python
from .base import *  # noqa: F403
from .base import env

DATABASES = {"default": env.db_url("DATABASE_URL")}

SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}
```

- [ ] **Step 6: Write `manage.py`, `config/urls.py`, `config/wsgi.py`, `.env.example`, `pyproject.toml`, `conftest.py`**

`manage.py` — standard Django `manage.py`, with `DJANGO_SETTINGS_MODULE` defaulting to `config.settings.dev`.

`config/urls.py`:

```python
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("allauth.urls")),
    path("", include("complaints.urls")),
]
```

`config/wsgi.py` — standard, defaulting to `config.settings.prod`.

`.env.example`:

```
SECRET_KEY=change-me
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
```

`pyproject.toml` — configure `ruff` (line-length 100), `mypy` (django-stubs plugin, `config.settings.dev`), and `pytest` (`DJANGO_SETTINGS_MODULE = "config.settings.dev"`, `testpaths = ["tests"]`, markers `ml`).

`conftest.py`:

```python
import os

os.environ.setdefault("SECRET_KEY", "test-only-key")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_settings.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml requirements.txt .env.example manage.py config/ conftest.py tests/
git commit -m "feat: Django project scaffolding with split settings

SECRET_KEY has no default and DEBUG defaults off, so a misconfigured
deploy fails closed. OAuth credentials are read from the environment
rather than a SocialApp database row."
```

---

### Task 2: Domain and Category models

**Files:**
- Create: `domains/__init__.py`, `domains/apps.py`, `domains/models.py`, `domains/admin.py`, `domains/migrations/`
- Test: `tests/domains/test_models.py`, `tests/factories.py`

**Interfaces:**
- Consumes: Task 1 settings
- Produces:
  - `domains.models.Domain(slug: str, name: str, is_active: bool)`
  - `domains.models.Category(domain: FK[Domain], slug: str, name: str, sla_hours: int)`
  - `Category` unique together on `(domain, slug)`
  - `tests.factories.DomainFactory`, `tests.factories.CategoryFactory`

- [ ] **Step 1: Write the failing test**

Create `tests/domains/test_models.py`:

```python
import pytest
from django.db import IntegrityError

from domains.models import Category, Domain


@pytest.mark.django_db
def test_domain_slug_is_unique():
    Domain.objects.create(slug="cfpb", name="Consumer Financial")
    with pytest.raises(IntegrityError):
        Domain.objects.create(slug="cfpb", name="Duplicate")


@pytest.mark.django_db
def test_category_slug_unique_per_domain_not_globally():
    """The same category slug in two domains is legitimate."""
    cfpb = Domain.objects.create(slug="cfpb", name="Consumer Financial")
    nyc = Domain.objects.create(slug="nyc311", name="Civic Services")
    Category.objects.create(domain=cfpb, slug="other", name="Other", sla_hours=72)
    Category.objects.create(domain=nyc, slug="other", name="Other", sla_hours=48)
    assert Category.objects.count() == 2


@pytest.mark.django_db
def test_category_slug_collides_within_one_domain():
    cfpb = Domain.objects.create(slug="cfpb", name="Consumer Financial")
    Category.objects.create(domain=cfpb, slug="mortgage", name="Mortgage", sla_hours=72)
    with pytest.raises(IntegrityError):
        Category.objects.create(domain=cfpb, slug="mortgage", name="Dup", sla_hours=24)


@pytest.mark.django_db
def test_sla_hours_must_be_positive():
    cfpb = Domain.objects.create(slug="cfpb", name="Consumer Financial")
    with pytest.raises(IntegrityError):
        Category.objects.create(domain=cfpb, slug="bad", name="Bad", sla_hours=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/domains/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'domains'`

- [ ] **Step 3: Write `domains/models.py`**

```python
from django.db import models


class Domain(models.Model):
    slug = models.SlugField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        permissions = [("manage_domain", "Can manage domains and categories")]

    def __str__(self) -> str:
        return self.name


class Category(models.Model):
    domain = models.ForeignKey(Domain, on_delete=models.CASCADE, related_name="categories")
    slug = models.SlugField(max_length=100)
    name = models.CharField(max_length=200)
    sla_hours = models.PositiveIntegerField(
        help_text="Hours from human triage confirmation until the complaint is due."
    )

    class Meta:
        ordering = ["domain", "name"]
        constraints = [
            models.UniqueConstraint(
                fields=["domain", "slug"], name="unique_category_slug_per_domain"
            ),
            models.CheckConstraint(condition=models.Q(sla_hours__gt=0), name="sla_hours_positive"),
        ]

    def __str__(self) -> str:
        return f"{self.domain.slug}/{self.slug}"
```

Note: on Django 5.2 the `CheckConstraint` keyword is `condition`. If a `TypeError` names `check`, the installed version is older than expected — stop and report rather than silently switching.

- [ ] **Step 4: Create `tests/factories.py`**

```python
import factory
from django.contrib.auth.models import User

from domains.models import Category, Domain


class UserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda o: f"{o.username}@example.com")


class DomainFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Domain

    slug = factory.Sequence(lambda n: f"domain{n}")
    name = factory.Sequence(lambda n: f"Domain {n}")


class CategoryFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Category

    domain = factory.SubFactory(DomainFactory)
    slug = factory.Sequence(lambda n: f"category{n}")
    name = factory.Sequence(lambda n: f"Category {n}")
    sla_hours = 72
```

- [ ] **Step 5: Make migrations and run the tests**

```bash
.venv/Scripts/python manage.py makemigrations domains
.venv/Scripts/pytest tests/domains/test_models.py -v
```

Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add domains/ tests/
git commit -m "feat: Domain and Category models

Category slugs are unique per domain rather than globally, so two
domains may both have an 'other' category."
```

---

### Task 3: Domain pack registry

**Files:**
- Create: `domains/packs.py`
- Test: `tests/domains/test_packs.py`

**Interfaces:**
- Consumes: `domains.models.Domain`
- Produces:
  - `domains.packs.DomainPack` — Protocol with `slug: str`, `display_name: str`
  - `domains.packs.PACKS: dict[str, type[DomainPack]]`
  - `domains.packs.get_pack(slug: str) -> type[DomainPack]` — raises `UnknownDomainError`
  - `domains.packs.UnknownDomainError(KeyError)`

- [ ] **Step 1: Write the failing test**

Create `tests/domains/test_packs.py`:

```python
import pytest

from domains.packs import PACKS, DomainPack, UnknownDomainError, get_pack


def test_registered_pack_resolves_by_slug():
    pack = get_pack("cfpb")
    assert pack.slug == "cfpb"


def test_unknown_slug_raises_a_named_error():
    with pytest.raises(UnknownDomainError) as exc:
        get_pack("does-not-exist")
    assert "does-not-exist" in str(exc.value)


def test_every_pack_key_matches_its_slug():
    """A mismatch here would make get_pack return the wrong pack silently."""
    for key, pack in PACKS.items():
        assert key == pack.slug


def test_packs_satisfy_the_protocol():
    for pack in PACKS.values():
        assert isinstance(pack.slug, str)
        assert isinstance(pack.display_name, str)
        assert issubclass(pack, DomainPack)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/domains/test_packs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'domains.packs'`

- [ ] **Step 3: Write `domains/packs.py`**

```python
"""Domain packs.

A domain's *identity* lives in the database as a Domain row. A domain's
*behavior* lives here in code. Nothing outside this module may branch on a
domain slug.

Phase 1 packs carry only identity. Phase 2 adds dataset adapters; Phase 3
adds model bundles.
"""

from typing import ClassVar, Protocol, runtime_checkable


class UnknownDomainError(KeyError):
    """Raised when a domain slug has no registered pack."""


@runtime_checkable
class DomainPack(Protocol):
    slug: ClassVar[str]
    display_name: ClassVar[str]


class CFPBPack:
    slug: ClassVar[str] = "cfpb"
    display_name: ClassVar[str] = "Consumer Financial"


class NYC311Pack:
    slug: ClassVar[str] = "nyc311"
    display_name: ClassVar[str] = "Civic Services"


PACKS: dict[str, type[DomainPack]] = {
    CFPBPack.slug: CFPBPack,
    NYC311Pack.slug: NYC311Pack,
}


def get_pack(slug: str) -> type[DomainPack]:
    try:
        return PACKS[slug]
    except KeyError as exc:
        raise UnknownDomainError(f"No domain pack registered for slug {slug!r}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/domains/test_packs.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add domains/packs.py tests/domains/test_packs.py
git commit -m "feat: domain pack registry

Database rows identify a domain; the registry resolves a slug to code.
This is the only place a domain slug may be branched on."
```

---

### Task 4: ML protocols and null implementations

**Files:**
- Create: `ml/__init__.py`, `ml/apps.py`, `ml/base.py`, `ml/null.py`, `ml/registry.py`
- Test: `tests/ml/test_null.py`, `tests/ml/test_registry.py`

**Interfaces:**
- Consumes: `django.conf.settings.ML_ARTIFACT_VERSIONS`
- Produces:
  - `ml.base.TriagePrediction(category_slug: str | None, confidence: float, model_version: str)`
  - `ml.base.Match(complaint_id: int, similarity: float)`
  - `ml.base.RiskScore(score: float, band: str, model_version: str)`
  - `ml.base.RiskFeatures(sla_hours: int, category_mean_resolution_hours: float, category_breach_rate: float, priority_rank: float, age_hours: float, submitted_hour: int, submitted_weekday: int, text_length: int, queue_depth: int, assignee_open_count: int)`
  - `ml.base.TriageModel`, `ml.base.DedupIndex`, `ml.base.RiskModel` — Protocols
  - `ml.null.NullTriageModel`, `ml.null.NullDedupIndex`, `ml.null.NullRiskModel`
  - `ml.registry.get_triage_model(domain_slug: str) -> TriageModel`
  - `ml.registry.get_dedup_index(domain_slug: str) -> DedupIndex`
  - `ml.registry.get_risk_model(domain_slug: str) -> RiskModel`
  - `ml.registry.registry_status() -> dict[str, dict[str, str]]`

- [ ] **Step 1: Write the failing test**

Create `tests/ml/test_null.py`:

```python
from ml.base import RiskFeatures
from ml.null import NullDedupIndex, NullRiskModel, NullTriageModel

FEATURES = RiskFeatures(
    sla_hours=72,
    category_mean_resolution_hours=40.0,
    category_breach_rate=0.1,
    priority_rank=0.5,
    age_hours=2.0,
    submitted_hour=9,
    submitted_weekday=2,
    text_length=400,
    queue_depth=12,
    assignee_open_count=3,
)


def test_null_triage_abstains_rather_than_guessing():
    prediction = NullTriageModel().predict("my mortgage servicer lost my payment")
    assert prediction.category_slug is None
    assert prediction.confidence == 0.0
    assert prediction.model_version == "null"


def test_null_dedup_returns_no_matches():
    assert NullDedupIndex().query("anything", k=5) == []


def test_null_risk_returns_an_unknown_band():
    score = NullRiskModel().predict(FEATURES)
    assert score.band == "unknown"
    assert score.model_version == "null"


def test_results_are_immutable():
    """Prediction results are evidence. Nothing downstream may mutate them."""
    import dataclasses
    import pytest

    prediction = NullTriageModel().predict("text")
    with pytest.raises(dataclasses.FrozenInstanceError):
        prediction.confidence = 0.99  # type: ignore[misc]
```

Create `tests/ml/test_registry.py`:

```python
from ml.null import NullDedupIndex, NullRiskModel, NullTriageModel
from ml.registry import get_dedup_index, get_risk_model, get_triage_model, registry_status


def test_missing_artifacts_yield_null_implementations():
    """A fresh clone with no artifacts must still run."""
    assert isinstance(get_triage_model("cfpb"), NullTriageModel)
    assert isinstance(get_dedup_index("cfpb"), NullDedupIndex)
    assert isinstance(get_risk_model("cfpb"), NullRiskModel)


def test_registry_status_reports_null_state_per_domain():
    status = registry_status()
    assert status["cfpb"]["triage"] == "null"
    assert status["nyc311"]["dedup"] == "null"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/ml -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.base'`

- [ ] **Step 3: Write `ml/base.py`**

```python
"""Inference interfaces.

Every result object is frozen and carries the model_version that produced it,
so a prediction can always be traced to a specific artifact.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TriagePrediction:
    category_slug: str | None
    confidence: float
    model_version: str


@dataclass(frozen=True)
class Match:
    complaint_id: int
    similarity: float


@dataclass(frozen=True)
class RiskScore:
    score: float
    band: str
    model_version: str


@dataclass(frozen=True)
class RiskFeatures:
    """Domain-independent by construction.

    Categories enter through how they *behave* (sla_hours, mean resolution,
    breach rate), never through which category they *are*. A category identity
    feature would be meaningless when a model trained on one domain serves
    another.
    """

    sla_hours: int
    category_mean_resolution_hours: float
    category_breach_rate: float
    priority_rank: float
    age_hours: float
    submitted_hour: int
    submitted_weekday: int
    text_length: int
    queue_depth: int
    assignee_open_count: int


class TriageModel(Protocol):
    def predict(self, text: str) -> TriagePrediction: ...


class DedupIndex(Protocol):
    def query(self, text: str, k: int) -> list[Match]: ...


class RiskModel(Protocol):
    def predict(self, features: RiskFeatures) -> RiskScore: ...
```

- [ ] **Step 4: Write `ml/null.py`**

```python
"""Null implementations.

These are the behavior of the system when no artifacts are installed, and the
fallback when a real model raises. Abstaining is always preferred to guessing.
"""

from ml.base import Match, RiskFeatures, RiskScore, TriagePrediction

NULL_VERSION = "null"


class NullTriageModel:
    def predict(self, text: str) -> TriagePrediction:
        return TriagePrediction(category_slug=None, confidence=0.0, model_version=NULL_VERSION)


class NullDedupIndex:
    def query(self, text: str, k: int) -> list[Match]:
        return []


class NullRiskModel:
    def predict(self, features: RiskFeatures) -> RiskScore:
        return RiskScore(score=0.0, band="unknown", model_version=NULL_VERSION)
```

- [ ] **Step 5: Write `ml/registry.py`**

```python
"""Model resolution.

Phase 1 always resolves to null implementations. Phase 3 replaces the bodies
of the loader functions; no consumer changes.
"""

from django.conf import settings

from domains.packs import PACKS
from ml.base import DedupIndex, RiskModel, TriageModel
from ml.null import NULL_VERSION, NullDedupIndex, NullRiskModel, NullTriageModel

MODEL_KINDS = ("triage", "dedup", "risk")


def _pinned_version(domain_slug: str, kind: str) -> str | None:
    return settings.ML_ARTIFACT_VERSIONS.get(domain_slug, {}).get(kind)


def get_triage_model(domain_slug: str) -> TriageModel:
    return NullTriageModel()


def get_dedup_index(domain_slug: str) -> DedupIndex:
    return NullDedupIndex()


def get_risk_model(domain_slug: str) -> RiskModel:
    return NullRiskModel()


def registry_status() -> dict[str, dict[str, str]]:
    """Which model version is serving each domain. Surfaced by /healthz."""
    return {
        slug: {kind: _pinned_version(slug, kind) or NULL_VERSION for kind in MODEL_KINDS}
        for slug in PACKS
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/ml -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add ml/ tests/ml/
git commit -m "feat: ML protocols with null implementations

The application is fully functional with no artifacts installed. Result
objects are frozen and carry their model_version."
```

---

### Task 5: Complaint model and its constraints

**Files:**
- Create: `complaints/__init__.py`, `complaints/apps.py`, `complaints/models.py`, `complaints/migrations/`
- Modify: `tests/factories.py`
- Test: `tests/complaints/test_models.py`

**Interfaces:**
- Consumes: `domains.models.Domain`, `domains.models.Category`
- Produces:
  - `complaints.models.Status` — TextChoices: `SUBMITTED`, `IN_REVIEW`, `IN_PROGRESS`, `RESOLVED`, `CLOSED`, `DUPLICATE`
  - `complaints.models.Priority` — TextChoices: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
  - `complaints.models.Complaint`
  - `tests.factories.ComplaintFactory`

- [ ] **Step 1: Write the failing test**

Create `tests/complaints/test_models.py`:

```python
import pytest
from django.db import IntegrityError

from complaints.models import Complaint, Status
from tests.factories import ComplaintFactory


@pytest.mark.django_db
def test_duplicate_status_requires_a_canonical_complaint():
    complaint = ComplaintFactory()
    complaint.status = Status.DUPLICATE
    with pytest.raises(IntegrityError):
        complaint.save()


@pytest.mark.django_db
def test_a_complaint_cannot_duplicate_itself():
    complaint = ComplaintFactory()
    complaint.status = Status.DUPLICATE
    complaint.duplicate_of = complaint
    with pytest.raises(IntegrityError):
        complaint.save()


@pytest.mark.django_db
def test_duplicate_with_a_canonical_is_valid():
    canonical = ComplaintFactory()
    duplicate = ComplaintFactory()
    duplicate.status = Status.DUPLICATE
    duplicate.duplicate_of = canonical
    duplicate.save()
    assert Complaint.objects.filter(status=Status.DUPLICATE).count() == 1


@pytest.mark.django_db
def test_new_complaints_start_submitted_and_untriaged():
    complaint = ComplaintFactory()
    assert complaint.status == Status.SUBMITTED
    assert complaint.triaged_at is None
    assert complaint.due_at is None
    assert complaint.category is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/complaints/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'complaints.models'`

- [ ] **Step 3: Write `complaints/models.py`**

```python
from django.conf import settings
from django.db import models

from domains.models import Category, Domain


class Status(models.TextChoices):
    SUBMITTED = "submitted", "Submitted"
    IN_REVIEW = "in_review", "In review"
    IN_PROGRESS = "in_progress", "In progress"
    RESOLVED = "resolved", "Resolved"
    CLOSED = "closed", "Closed"
    DUPLICATE = "duplicate", "Duplicate"


class Priority(models.TextChoices):
    LOW = "low", "Low"
    MEDIUM = "medium", "Medium"
    HIGH = "high", "High"
    CRITICAL = "critical", "Critical"


PRIORITY_RANK = {
    Priority.LOW: 0.25,
    Priority.MEDIUM: 0.5,
    Priority.HIGH: 0.75,
    Priority.CRITICAL: 1.0,
}


class Complaint(models.Model):
    domain = models.ForeignKey(Domain, on_delete=models.PROTECT, related_name="complaints")

    # Human-owned ground truth. Never written by a model.
    category = models.ForeignKey(
        Category, on_delete=models.PROTECT, null=True, blank=True, related_name="complaints"
    )
    priority = models.CharField(max_length=20, choices=Priority.choices, null=True, blank=True)

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.SUBMITTED)
    title = models.CharField(max_length=300)
    body = models.TextField()

    submitted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="submitted_complaints"
    )
    assignee = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assigned_complaints",
    )
    duplicate_of = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True, related_name="duplicates"
    )

    # float32 bytes, not JSON: 1,536 bytes against roughly 9KB serialized.
    embedding = models.BinaryField(null=True, blank=True, editable=False)

    created_at = models.DateTimeField(auto_now_add=True)
    triaged_at = models.DateTimeField(null=True, blank=True)
    due_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        permissions = [
            ("view_queue", "Can view the agent work queue"),
            ("triage_complaint", "Can confirm category and priority"),
            ("assign_complaint", "Can assign complaints to agents"),
            ("resolve_complaint", "Can resolve and close complaints"),
            ("mark_duplicate", "Can mark a complaint as a duplicate"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(status=Status.DUPLICATE) | models.Q(duplicate_of__isnull=False),
                name="duplicate_requires_canonical",
            ),
            models.CheckConstraint(
                condition=~models.Q(duplicate_of=models.F("id")),
                name="duplicate_of_is_not_self",
            ),
        ]

    def __str__(self) -> str:
        return f"#{self.pk} {self.title[:60]}"

    @property
    def is_overdue(self) -> bool:
        from django.utils import timezone

        return self.due_at is not None and self.resolved_at is None and self.due_at < timezone.now()
```

- [ ] **Step 4: Add `ComplaintFactory` to `tests/factories.py`**

```python
from complaints.models import Complaint


class ComplaintFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Complaint

    domain = factory.SubFactory(DomainFactory)
    submitted_by = factory.SubFactory(UserFactory)
    title = factory.Sequence(lambda n: f"Complaint {n}")
    body = "The servicer applied my payment to the wrong account and will not correct it."
```

- [ ] **Step 5: Make migrations and run the tests**

```bash
.venv/Scripts/python manage.py makemigrations complaints
.venv/Scripts/pytest tests/complaints/test_models.py -v
```

Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add complaints/ tests/
git commit -m "feat: Complaint model with duplicate integrity constraints

A DUPLICATE complaint must name a canonical, and may not name itself.
Both are enforced by database check constraints, not just application code."
```

---

### Task 6: Prediction model — immutable evidence

**Files:**
- Modify: `complaints/models.py`
- Test: `tests/complaints/test_prediction.py`

**Interfaces:**
- Consumes: `complaints.models.Complaint`
- Produces:
  - `complaints.models.PredictionKind` — TextChoices: `TRIAGE`, `DEDUP`, `RISK`
  - `complaints.models.Prediction(complaint, kind, payload, model_name, model_version, created_at)`
  - `complaints.models.ImmutableRecordError(Exception)`
  - `tests.factories.PredictionFactory`

- [ ] **Step 1: Write the failing test**

Create `tests/complaints/test_prediction.py`:

```python
import pytest

from complaints.models import ImmutableRecordError, Prediction, PredictionKind
from tests.factories import ComplaintFactory


@pytest.mark.django_db
def test_prediction_records_the_model_version_that_produced_it():
    complaint = ComplaintFactory()
    prediction = Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.TRIAGE,
        payload={"category_slug": "mortgage", "confidence": 0.91},
        model_name="triage",
        model_version="v1",
    )
    assert prediction.model_version == "v1"


@pytest.mark.django_db
def test_predictions_cannot_be_updated():
    """Predictions are evidence. Rewriting one would falsify the audit trail."""
    complaint = ComplaintFactory()
    prediction = Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.TRIAGE,
        payload={"category_slug": "mortgage", "confidence": 0.91},
        model_name="triage",
        model_version="v1",
    )
    prediction.model_version = "v2"
    with pytest.raises(ImmutableRecordError):
        prediction.save()


@pytest.mark.django_db
def test_predictions_cannot_be_deleted():
    complaint = ComplaintFactory()
    prediction = Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.DEDUP,
        payload={"matches": []},
        model_name="dedup",
        model_version="v1",
    )
    with pytest.raises(ImmutableRecordError):
        prediction.delete()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/complaints/test_prediction.py -v`
Expected: FAIL — `ImportError: cannot import name 'Prediction'`

- [ ] **Step 3: Add to `complaints/models.py`**

```python
class ImmutableRecordError(Exception):
    """Raised on any attempt to modify an append-only record."""


class PredictionKind(models.TextChoices):
    TRIAGE = "triage", "Triage"
    DEDUP = "dedup", "Duplicate detection"
    RISK = "risk", "SLA risk"


class Prediction(models.Model):
    """Append-only model output.

    A Prediction records what a model said and which artifact said it. It is
    never written into Complaint.category or Complaint.priority — those belong
    to a human. Keeping the two apart is what makes live evaluation possible:
    joining predictions to the human decisions in ComplaintEvent gives a real
    accuracy figure rather than a test-set one.
    """

    complaint = models.ForeignKey(Complaint, on_delete=models.CASCADE, related_name="predictions")
    kind = models.CharField(max_length=20, choices=PredictionKind.choices)
    payload = models.JSONField()
    model_name = models.CharField(max_length=100)
    model_version = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["complaint", "kind", "-created_at"])]
        permissions = [("view_ml_metrics", "Can view model metrics")]

    def save(self, *args, **kwargs):
        if self.pk is not None:
            raise ImmutableRecordError("Prediction rows are append-only and cannot be updated")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ImmutableRecordError("Prediction rows are append-only and cannot be deleted")

    def __str__(self) -> str:
        return f"{self.kind}@{self.model_version} for #{self.complaint_id}"
```

- [ ] **Step 4: Add `PredictionFactory` to `tests/factories.py`**

```python
from complaints.models import Prediction, PredictionKind


class PredictionFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Prediction

    complaint = factory.SubFactory(ComplaintFactory)
    kind = PredictionKind.TRIAGE
    payload = factory.LazyFunction(lambda: {"category_slug": "mortgage", "confidence": 0.9})
    model_name = "triage"
    model_version = "v1"
```

- [ ] **Step 5: Make migrations and run the tests**

```bash
.venv/Scripts/python manage.py makemigrations complaints
.venv/Scripts/pytest tests/complaints/ -v
```

Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add complaints/ tests/
git commit -m "feat: append-only Prediction model

Predictions are evidence, not state. Update and delete both raise, so
the audit trail cannot be rewritten after the fact."
```

---

### Task 7: ComplaintEvent — audit trail and acceptance tracking

**Files:**
- Modify: `complaints/models.py`
- Test: `tests/complaints/test_events.py`

**Interfaces:**
- Consumes: `complaints.models.Complaint`, `complaints.models.Prediction`
- Produces:
  - `complaints.models.EventKind` — TextChoices: `STATUS`, `CATEGORY`, `PRIORITY`, `ASSIGNMENT`, `DUPLICATE`
  - `complaints.models.ComplaintEvent(complaint, kind, from_value, to_value, actor, prediction, note, created_at)`
  - `ComplaintEvent.was_prediction_accepted -> bool | None`

- [ ] **Step 1: Write the failing test**

Create `tests/complaints/test_events.py`:

```python
import pytest

from complaints.models import ComplaintEvent, EventKind
from tests.factories import ComplaintFactory, PredictionFactory, UserFactory


@pytest.mark.django_db
def test_event_without_a_prediction_has_no_acceptance_verdict():
    event = ComplaintEvent.objects.create(
        complaint=ComplaintFactory(),
        kind=EventKind.CATEGORY,
        from_value=None,
        to_value="mortgage",
        actor=UserFactory(),
    )
    assert event.was_prediction_accepted is None


@pytest.mark.django_db
def test_matching_decision_counts_as_acceptance():
    complaint = ComplaintFactory()
    prediction = PredictionFactory(
        complaint=complaint, payload={"category_slug": "mortgage", "confidence": 0.9}
    )
    event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.CATEGORY,
        to_value="mortgage",
        actor=UserFactory(),
        prediction=prediction,
    )
    assert event.was_prediction_accepted is True


@pytest.mark.django_db
def test_differing_decision_counts_as_override():
    """Overrides are the retraining signal, so they must be identifiable."""
    complaint = ComplaintFactory()
    prediction = PredictionFactory(
        complaint=complaint, payload={"category_slug": "mortgage", "confidence": 0.9}
    )
    event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.CATEGORY,
        to_value="credit_card",
        actor=UserFactory(),
        prediction=prediction,
    )
    assert event.was_prediction_accepted is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/complaints/test_events.py -v`
Expected: FAIL — `ImportError: cannot import name 'ComplaintEvent'`

- [ ] **Step 3: Add to `complaints/models.py`**

```python
class EventKind(models.TextChoices):
    STATUS = "status", "Status change"
    CATEGORY = "category", "Category change"
    PRIORITY = "priority", "Priority change"
    ASSIGNMENT = "assignment", "Assignment change"
    DUPLICATE = "duplicate", "Duplicate decision"


class ComplaintEvent(models.Model):
    """Who changed what, when, from what, to what, and why.

    When a decision responds to a model suggestion, `prediction` is set. A
    to_value matching the prediction is an acceptance; a differing one is an
    override. Querying overrides gives the retraining corpus.
    """

    complaint = models.ForeignKey(Complaint, on_delete=models.CASCADE, related_name="events")
    kind = models.CharField(max_length=20, choices=EventKind.choices)
    from_value = models.CharField(max_length=200, null=True, blank=True)
    to_value = models.CharField(max_length=200, null=True, blank=True)
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="complaint_events",
    )
    prediction = models.ForeignKey(
        "Prediction", on_delete=models.SET_NULL, null=True, blank=True, related_name="decisions"
    )
    note = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [models.Index(fields=["complaint", "created_at"])]

    @property
    def was_prediction_accepted(self) -> bool | None:
        """None when the decision responded to no suggestion."""
        if self.prediction is None:
            return None
        suggested = self.prediction.payload.get("category_slug")
        return suggested == self.to_value

    def __str__(self) -> str:
        return f"{self.kind}: {self.from_value} -> {self.to_value}"
```

- [ ] **Step 4: Make migrations and run the tests**

```bash
.venv/Scripts/python manage.py makemigrations complaints
.venv/Scripts/pytest tests/complaints/ -v
```

Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add complaints/ tests/
git commit -m "feat: ComplaintEvent audit trail

One table answers who/what/when/from/to/why, and links decisions back to
the prediction they accepted or overrode."
```

---

### Task 8: Lifecycle service — transitions, triage, SLA

**Files:**
- Create: `complaints/services.py`
- Test: `tests/complaints/test_services.py`

**Interfaces:**
- Consumes: all of `complaints.models`
- Produces:
  - `complaints.services.InvalidTransition(Exception)`
  - `complaints.services.ALLOWED_TRANSITIONS: dict[str, set[str]]`
  - `complaints.services.transition(complaint, to_status, actor, note="") -> ComplaintEvent`
  - `complaints.services.triage(complaint, category, priority, actor, prediction=None) -> ComplaintEvent`
  - `complaints.services.assign(complaint, assignee, actor) -> ComplaintEvent`
  - `complaints.services.resolve(complaint, actor, note="") -> ComplaintEvent`

- [ ] **Step 1: Write the failing test**

Create `tests/complaints/test_services.py`:

```python
from datetime import timedelta

import pytest
from django.utils import timezone

from complaints import services
from complaints.models import ComplaintEvent, EventKind, Priority, Status
from tests.factories import CategoryFactory, ComplaintFactory, PredictionFactory, UserFactory


@pytest.mark.django_db
def test_legal_transition_writes_an_event():
    complaint = ComplaintFactory()
    actor = UserFactory()
    services.transition(complaint, Status.IN_REVIEW, actor)
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_REVIEW
    event = ComplaintEvent.objects.get(complaint=complaint, kind=EventKind.STATUS)
    assert (event.from_value, event.to_value) == (Status.SUBMITTED, Status.IN_REVIEW)
    assert event.actor == actor


@pytest.mark.django_db
def test_illegal_transition_is_refused():
    complaint = ComplaintFactory()
    with pytest.raises(services.InvalidTransition):
        services.transition(complaint, Status.CLOSED, UserFactory())


@pytest.mark.django_db
def test_illegal_transition_leaves_no_event_behind():
    """A refused transition must not half-apply."""
    complaint = ComplaintFactory()
    with pytest.raises(services.InvalidTransition):
        services.transition(complaint, Status.CLOSED, UserFactory())
    complaint.refresh_from_db()
    assert complaint.status == Status.SUBMITTED
    assert ComplaintEvent.objects.filter(complaint=complaint).count() == 0


@pytest.mark.django_db
def test_triage_sets_the_sla_clock_from_human_confirmation():
    """due_at derives from triaged_at, which is the human decision moment."""
    category = CategoryFactory(sla_hours=48)
    complaint = ComplaintFactory(domain=category.domain)
    before = timezone.now()

    services.triage(complaint, category, Priority.HIGH, UserFactory())

    complaint.refresh_from_db()
    assert complaint.category == category
    assert complaint.priority == Priority.HIGH
    assert complaint.status == Status.IN_REVIEW
    assert complaint.triaged_at >= before
    assert complaint.due_at == complaint.triaged_at + timedelta(hours=48)


@pytest.mark.django_db
def test_triage_links_the_prediction_it_accepted():
    category = CategoryFactory(slug="mortgage", sla_hours=72)
    complaint = ComplaintFactory(domain=category.domain)
    prediction = PredictionFactory(
        complaint=complaint, payload={"category_slug": "mortgage", "confidence": 0.9}
    )

    services.triage(complaint, category, Priority.LOW, UserFactory(), prediction=prediction)

    event = ComplaintEvent.objects.get(complaint=complaint, kind=EventKind.CATEGORY)
    assert event.was_prediction_accepted is True


@pytest.mark.django_db
def test_triage_records_an_override_when_the_human_disagrees():
    category = CategoryFactory(slug="credit_card", sla_hours=72)
    complaint = ComplaintFactory(domain=category.domain)
    prediction = PredictionFactory(
        complaint=complaint, payload={"category_slug": "mortgage", "confidence": 0.9}
    )

    services.triage(complaint, category, Priority.LOW, UserFactory(), prediction=prediction)

    event = ComplaintEvent.objects.get(complaint=complaint, kind=EventKind.CATEGORY)
    assert event.was_prediction_accepted is False


@pytest.mark.django_db
def test_triage_rejects_a_category_from_another_domain():
    """Cross-domain categories would corrupt every per-domain metric."""
    complaint = ComplaintFactory()
    foreign_category = CategoryFactory()
    with pytest.raises(services.InvalidTransition):
        services.triage(complaint, foreign_category, Priority.LOW, UserFactory())


@pytest.mark.django_db
def test_resolve_stamps_resolved_at():
    category = CategoryFactory(sla_hours=24)
    complaint = ComplaintFactory(domain=category.domain)
    actor = UserFactory()
    services.triage(complaint, category, Priority.LOW, actor)
    services.transition(complaint, Status.IN_PROGRESS, actor)
    services.resolve(complaint, actor)
    complaint.refresh_from_db()
    assert complaint.status == Status.RESOLVED
    assert complaint.resolved_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/complaints/test_services.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'complaints.services'`

- [ ] **Step 3: Write `complaints/services.py`**

```python
"""The only place complaint state changes.

Every mutation writes a ComplaintEvent, and every mutation is atomic, so a
refused change leaves neither state nor audit trail behind.
"""

from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from complaints.models import (
    Complaint,
    ComplaintEvent,
    EventKind,
    Prediction,
    Priority,
    Status,
)
from domains.models import Category


class InvalidTransition(Exception):
    """Raised when a requested change is not legal from the current state."""


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    Status.SUBMITTED: {Status.IN_REVIEW, Status.DUPLICATE},
    Status.IN_REVIEW: {Status.IN_PROGRESS, Status.DUPLICATE},
    Status.IN_PROGRESS: {Status.RESOLVED, Status.DUPLICATE},
    Status.RESOLVED: {Status.CLOSED, Status.IN_PROGRESS},
    Status.CLOSED: set(),
    Status.DUPLICATE: set(),
}


@transaction.atomic
def transition(complaint: Complaint, to_status: str, actor, note: str = "") -> ComplaintEvent:
    from_status = complaint.status
    if to_status not in ALLOWED_TRANSITIONS[from_status]:
        raise InvalidTransition(f"Cannot move a complaint from {from_status} to {to_status}")

    complaint.status = to_status
    if to_status == Status.RESOLVED and complaint.resolved_at is None:
        complaint.resolved_at = timezone.now()
    complaint.save()

    return ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.STATUS,
        from_value=from_status,
        to_value=to_status,
        actor=actor,
        note=note,
    )


@transaction.atomic
def triage(
    complaint: Complaint,
    category: Category,
    priority: str,
    actor,
    prediction: Prediction | None = None,
) -> ComplaintEvent:
    """Human confirmation of category and priority. Starts the SLA clock.

    triaged_at is set here and nowhere else. A model predicting instantly does
    not start the clock; a human confirming does.
    """
    if category.domain_id != complaint.domain_id:
        raise InvalidTransition(
            f"Category {category} belongs to another domain than complaint #{complaint.pk}"
        )
    if priority not in Priority.values:
        raise InvalidTransition(f"Unknown priority {priority!r}")

    previous_category = complaint.category.slug if complaint.category else None
    previous_priority = complaint.priority

    now = timezone.now()
    complaint.category = category
    complaint.priority = priority
    complaint.triaged_at = now
    complaint.due_at = now + timedelta(hours=category.sla_hours)
    complaint.save()

    category_event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.CATEGORY,
        from_value=previous_category,
        to_value=category.slug,
        actor=actor,
        prediction=prediction,
    )
    ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.PRIORITY,
        from_value=previous_priority,
        to_value=priority,
        actor=actor,
    )
    if complaint.status == Status.SUBMITTED:
        transition(complaint, Status.IN_REVIEW, actor, note="Triaged")

    return category_event


@transaction.atomic
def assign(complaint: Complaint, assignee, actor) -> ComplaintEvent:
    previous = complaint.assignee.username if complaint.assignee else None
    complaint.assignee = assignee
    complaint.save()
    return ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.ASSIGNMENT,
        from_value=previous,
        to_value=assignee.username if assignee else None,
        actor=actor,
    )


def resolve(complaint: Complaint, actor, note: str = "") -> ComplaintEvent:
    return transition(complaint, Status.RESOLVED, actor, note=note)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/complaints/test_services.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add complaints/services.py tests/complaints/test_services.py
git commit -m "feat: complaint lifecycle service

All state changes flow through here, atomically, each writing an audit
event. triaged_at is set by human confirmation only, and due_at derives
from it."
```

---

### Task 9: Duplicate marking with chain prevention

**Files:**
- Modify: `complaints/services.py`
- Test: `tests/complaints/test_duplicates.py`

**Interfaces:**
- Consumes: Task 8 service layer
- Produces: `complaints.services.mark_duplicate(complaint, canonical, actor, prediction=None) -> ComplaintEvent`

- [ ] **Step 1: Write the failing test**

Create `tests/complaints/test_duplicates.py`:

```python
import pytest

from complaints import services
from complaints.models import ComplaintEvent, EventKind, Status
from tests.factories import ComplaintFactory, UserFactory


@pytest.mark.django_db
def test_marking_a_duplicate_points_at_the_canonical():
    canonical = ComplaintFactory()
    duplicate = ComplaintFactory()
    services.mark_duplicate(duplicate, canonical, UserFactory())
    duplicate.refresh_from_db()
    assert duplicate.status == Status.DUPLICATE
    assert duplicate.duplicate_of == canonical


@pytest.mark.django_db
def test_marking_a_duplicate_writes_both_a_duplicate_and_a_status_event():
    canonical = ComplaintFactory()
    duplicate = ComplaintFactory()
    services.mark_duplicate(duplicate, canonical, UserFactory())
    kinds = set(ComplaintEvent.objects.filter(complaint=duplicate).values_list("kind", flat=True))
    assert kinds == {EventKind.DUPLICATE, EventKind.STATUS}


@pytest.mark.django_db
def test_a_complaint_cannot_be_its_own_duplicate():
    complaint = ComplaintFactory()
    with pytest.raises(services.InvalidTransition):
        services.mark_duplicate(complaint, complaint, UserFactory())


@pytest.mark.django_db
def test_cannot_point_at_a_complaint_that_is_itself_a_duplicate():
    """Prevents chains: C -> B -> A. Every duplicate names a real canonical."""
    canonical = ComplaintFactory()
    first = ComplaintFactory()
    second = ComplaintFactory()
    services.mark_duplicate(first, canonical, UserFactory())
    with pytest.raises(services.InvalidTransition):
        services.mark_duplicate(second, first, UserFactory())


@pytest.mark.django_db
def test_cannot_mark_a_duplicate_across_domains():
    canonical = ComplaintFactory()
    other_domain_complaint = ComplaintFactory()
    with pytest.raises(services.InvalidTransition):
        services.mark_duplicate(other_domain_complaint, canonical, UserFactory())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/complaints/test_duplicates.py -v`
Expected: FAIL — `AttributeError: module 'complaints.services' has no attribute 'mark_duplicate'`

- [ ] **Step 3: Add `mark_duplicate` to `complaints/services.py`**

```python
@transaction.atomic
def mark_duplicate(
    complaint: Complaint,
    canonical: Complaint,
    actor,
    prediction: Prediction | None = None,
) -> ComplaintEvent:
    """Mark `complaint` as a duplicate of `canonical`.

    The database enforces "not itself" and "must have a canonical". The chain
    rule needs a query, so it lives here: a canonical may not itself be a
    duplicate, which keeps every duplicate exactly one hop from a real
    complaint and makes cycles impossible.
    """
    if complaint.pk == canonical.pk:
        raise InvalidTransition("A complaint cannot be a duplicate of itself")
    if canonical.duplicate_of_id is not None or canonical.status == Status.DUPLICATE:
        raise InvalidTransition(
            f"Complaint #{canonical.pk} is itself a duplicate; point at its canonical instead"
        )
    if complaint.domain_id != canonical.domain_id:
        raise InvalidTransition("Complaints in different domains cannot be duplicates")

    from_status = complaint.status
    if Status.DUPLICATE not in ALLOWED_TRANSITIONS[from_status]:
        raise InvalidTransition(f"Cannot mark a {from_status} complaint as a duplicate")

    complaint.duplicate_of = canonical
    complaint.status = Status.DUPLICATE
    complaint.save()

    ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.STATUS,
        from_value=from_status,
        to_value=Status.DUPLICATE,
        actor=actor,
    )
    return ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.DUPLICATE,
        from_value=None,
        to_value=str(canonical.pk),
        actor=actor,
        prediction=prediction,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/complaints/ -v`
Expected: 23 passed

- [ ] **Step 5: Commit**

```bash
git add complaints/services.py tests/complaints/test_duplicates.py
git commit -m "feat: duplicate marking with chain prevention

A canonical may not itself be a duplicate, so every duplicate is exactly
one hop from a real complaint and cycles are impossible."
```

---

### Task 10: Permission matrix

**Files:**
- Create: `complaints/permissions.py`, `complaints/management/commands/bootstrap_groups.py`
- Test: `tests/test_permissions.py`

**Interfaces:**
- Consumes: `Complaint.Meta.permissions`, `Prediction.Meta.permissions`, `Domain.Meta.permissions`
- Produces:
  - `complaints.permissions.SUBMITTER`, `AGENT`, `ADMIN` — group name constants
  - `complaints.permissions.GROUP_PERMISSIONS: dict[str, set[str]]`
  - `complaints.permissions.bootstrap_groups() -> None`
  - Management command `bootstrap_groups`

- [ ] **Step 1: Write the failing test**

Create `tests/test_permissions.py`:

```python
import pytest
from django.contrib.auth.models import Group

from complaints.permissions import ADMIN, AGENT, SUBMITTER, bootstrap_groups
from tests.factories import UserFactory

# (group, permission, is_granted) for every meaningful pair, negatives included.
MATRIX = [
    (SUBMITTER, "complaints.add_complaint", True),
    (SUBMITTER, "complaints.view_complaint", True),
    (SUBMITTER, "complaints.view_queue", False),
    (SUBMITTER, "complaints.triage_complaint", False),
    (SUBMITTER, "complaints.assign_complaint", False),
    (SUBMITTER, "complaints.resolve_complaint", False),
    (SUBMITTER, "complaints.mark_duplicate", False),
    (SUBMITTER, "domains.manage_domain", False),
    (SUBMITTER, "complaints.view_ml_metrics", False),
    (AGENT, "complaints.add_complaint", True),
    (AGENT, "complaints.view_complaint", True),
    (AGENT, "complaints.view_queue", True),
    (AGENT, "complaints.triage_complaint", True),
    (AGENT, "complaints.assign_complaint", True),
    (AGENT, "complaints.resolve_complaint", True),
    (AGENT, "complaints.mark_duplicate", True),
    (AGENT, "domains.manage_domain", False),
    (AGENT, "complaints.view_ml_metrics", False),
    (ADMIN, "complaints.view_queue", True),
    (ADMIN, "complaints.triage_complaint", True),
    (ADMIN, "complaints.assign_complaint", True),
    (ADMIN, "complaints.resolve_complaint", True),
    (ADMIN, "complaints.mark_duplicate", True),
    (ADMIN, "domains.manage_domain", True),
    (ADMIN, "complaints.view_ml_metrics", True),
]


@pytest.mark.django_db
@pytest.mark.parametrize("group_name,permission,granted", MATRIX)
def test_permission_matrix(group_name, permission, granted):
    bootstrap_groups()
    user = UserFactory()
    user.groups.add(Group.objects.get(name=group_name))
    user = type(user).objects.get(pk=user.pk)  # drop the permission cache
    assert user.has_perm(permission) is granted


@pytest.mark.django_db
def test_bootstrap_is_idempotent():
    bootstrap_groups()
    bootstrap_groups()
    assert Group.objects.filter(name__in=[SUBMITTER, AGENT, ADMIN]).count() == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_permissions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'complaints.permissions'`

- [ ] **Step 3: Write `complaints/permissions.py`**

```python
"""Roles are bundles of permissions, never hardcoded checks.

Authorization asks `user.has_perm("complaints.triage_complaint")`, never
`user.role == "agent"`. Adding a Supervisor tier later is a new bundle here,
with no change to any call site.
"""

from django.contrib.auth.models import Group, Permission

SUBMITTER = "Submitter"
AGENT = "Agent"
ADMIN = "Admin"

_SUBMITTER_PERMS = {
    "complaints.add_complaint",
    "complaints.view_complaint",
}

_AGENT_PERMS = _SUBMITTER_PERMS | {
    "complaints.view_queue",
    "complaints.change_complaint",
    "complaints.triage_complaint",
    "complaints.assign_complaint",
    "complaints.resolve_complaint",
    "complaints.mark_duplicate",
}

_ADMIN_PERMS = _AGENT_PERMS | {
    "domains.manage_domain",
    "complaints.view_ml_metrics",
}

GROUP_PERMISSIONS: dict[str, set[str]] = {
    SUBMITTER: _SUBMITTER_PERMS,
    AGENT: _AGENT_PERMS,
    ADMIN: _ADMIN_PERMS,
}


def bootstrap_groups() -> None:
    """Create the three groups and set their permissions. Idempotent."""
    for group_name, codenames in GROUP_PERMISSIONS.items():
        group, _ = Group.objects.get_or_create(name=group_name)
        permissions = []
        for dotted in codenames:
            app_label, codename = dotted.split(".")
            permissions.append(
                Permission.objects.get(content_type__app_label=app_label, codename=codename)
            )
        group.permissions.set(permissions)
```

- [ ] **Step 4: Write the management command**

Create `complaints/management/__init__.py`, `complaints/management/commands/__init__.py`, and `complaints/management/commands/bootstrap_groups.py`:

```python
from django.core.management.base import BaseCommand

from complaints.permissions import bootstrap_groups


class Command(BaseCommand):
    help = "Create the Submitter, Agent and Admin groups with their permissions."

    def handle(self, *args, **options):
        bootstrap_groups()
        self.stdout.write(self.style.SUCCESS("Groups bootstrapped."))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_permissions.py -v`
Expected: 26 passed

- [ ] **Step 6: Commit**

```bash
git add complaints/permissions.py complaints/management/ tests/test_permissions.py
git commit -m "feat: permission-based authorization with three role bundles

Every (role, permission) pair is asserted including the negatives, which
is where authorization bugs actually live."
```

---

### Task 11: Accounts — auth and automatic role assignment

**Files:**
- Create: `accounts/__init__.py`, `accounts/apps.py`, `accounts/signals.py`
- Modify: `config/settings/base.py`
- Test: `tests/accounts/test_signals.py`

**Interfaces:**
- Consumes: `complaints.permissions.SUBMITTER`
- Produces: new users automatically join the Submitter group

- [ ] **Step 1: Write the failing test**

Create `tests/accounts/test_signals.py`:

```python
import pytest
from django.contrib.auth.models import User

from complaints.permissions import SUBMITTER, bootstrap_groups


@pytest.mark.django_db
def test_new_users_become_submitters():
    bootstrap_groups()
    user = User.objects.create_user(username="newcomer", email="new@example.com")
    assert user.groups.filter(name=SUBMITTER).exists()
    assert user.has_perm("complaints.add_complaint")


@pytest.mark.django_db
def test_new_users_are_not_agents():
    bootstrap_groups()
    user = User.objects.create_user(username="newcomer2", email="new2@example.com")
    assert not user.has_perm("complaints.triage_complaint")


@pytest.mark.django_db
def test_superusers_are_not_downgraded():
    bootstrap_groups()
    admin = User.objects.create_superuser(username="root", email="root@example.com", password="x")
    assert admin.has_perm("complaints.triage_complaint")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/accounts -v`
Expected: FAIL — new user has no groups

- [ ] **Step 3: Write `accounts/signals.py` and wire it in `accounts/apps.py`**

```python
# accounts/signals.py
import logging

from django.contrib.auth.models import Group, User
from django.db.models.signals import post_save
from django.dispatch import receiver

from complaints.permissions import SUBMITTER

logger = logging.getLogger(__name__)


@receiver(post_save, sender=User)
def assign_default_group(sender, instance: User, created: bool, **kwargs) -> None:
    """Every new account starts as a Submitter. Elevation is deliberate."""
    if not created:
        return
    try:
        instance.groups.add(Group.objects.get(name=SUBMITTER))
    except Group.DoesNotExist:
        logger.warning("Submitter group missing; run `manage.py bootstrap_groups`")
```

```python
# accounts/apps.py
from django.apps import AppConfig


class AccountsConfig(AppConfig):
    name = "accounts"

    def ready(self) -> None:
        from accounts import signals  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/accounts -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add accounts/ tests/accounts/
git commit -m "feat: new accounts default to the Submitter role

Elevation to Agent or Admin is always a deliberate act."
```

---

### Task 12: DRF API

**Files:**
- Create: `complaints/api.py`, `complaints/urls.py`
- Modify: `config/settings/base.py` (add `REST_FRAMEWORK`)
- Test: `tests/complaints/test_api.py`

**Interfaces:**
- Consumes: `complaints.services`, `complaints.permissions`
- Produces:
  - `GET/POST /api/complaints/`
  - `GET /api/complaints/{id}/`
  - `POST /api/complaints/{id}/triage/` — body `{"category": <id>, "priority": "high"}`
  - `POST /api/complaints/{id}/resolve/`
  - `GET /healthz`

- [ ] **Step 1: Write the failing test**

Create `tests/complaints/test_api.py`:

```python
import pytest
from django.contrib.auth.models import Group
from rest_framework.test import APIClient

from complaints.models import Priority, Status
from complaints.permissions import AGENT, bootstrap_groups
from tests.factories import CategoryFactory, ComplaintFactory, DomainFactory, UserFactory


@pytest.fixture
def agent(db):
    bootstrap_groups()
    user = UserFactory()
    user.groups.add(Group.objects.get(name=AGENT))
    return type(user).objects.get(pk=user.pk)


@pytest.fixture
def api():
    return APIClient()


@pytest.mark.django_db
def test_anonymous_users_cannot_list_complaints(api):
    assert api.get("/api/complaints/").status_code in (401, 403)


@pytest.mark.django_db
def test_submitters_see_only_their_own_complaints(api):
    bootstrap_groups()
    mine = UserFactory()
    ComplaintFactory(submitted_by=mine)
    ComplaintFactory()  # someone else's
    api.force_authenticate(mine)
    response = api.get("/api/complaints/")
    assert response.status_code == 200
    assert len(response.data["results"]) == 1


@pytest.mark.django_db
def test_agents_see_every_complaint(api, agent):
    ComplaintFactory()
    ComplaintFactory()
    api.force_authenticate(agent)
    response = api.get("/api/complaints/")
    assert len(response.data["results"]) == 2


@pytest.mark.django_db
def test_submitter_cannot_triage(api):
    bootstrap_groups()
    submitter = UserFactory()
    category = CategoryFactory()
    complaint = ComplaintFactory(domain=category.domain)
    api.force_authenticate(submitter)
    response = api.post(
        f"/api/complaints/{complaint.pk}/triage/",
        {"category": category.pk, "priority": Priority.HIGH},
        format="json",
    )
    assert response.status_code == 403


@pytest.mark.django_db
def test_agent_can_triage_and_the_sla_clock_starts(api, agent):
    category = CategoryFactory(sla_hours=24)
    complaint = ComplaintFactory(domain=category.domain)
    api.force_authenticate(agent)
    response = api.post(
        f"/api/complaints/{complaint.pk}/triage/",
        {"category": category.pk, "priority": Priority.HIGH},
        format="json",
    )
    assert response.status_code == 200
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_REVIEW
    assert complaint.due_at is not None


@pytest.mark.django_db
def test_triage_with_a_foreign_domain_category_is_a_400_not_a_500(api, agent):
    complaint = ComplaintFactory()
    foreign = CategoryFactory()
    api.force_authenticate(agent)
    response = api.post(
        f"/api/complaints/{complaint.pk}/triage/",
        {"category": foreign.pk, "priority": Priority.LOW},
        format="json",
    )
    assert response.status_code == 400


@pytest.mark.django_db
def test_creating_a_complaint_records_the_submitter(api):
    bootstrap_groups()
    user = UserFactory()
    domain = DomainFactory()
    api.force_authenticate(user)
    response = api.post(
        "/api/complaints/",
        {"domain": domain.pk, "title": "Wrong charge", "body": "I was billed twice."},
        format="json",
    )
    assert response.status_code == 201
    assert response.data["submitted_by"] == user.username


@pytest.mark.django_db
def test_healthz_reports_model_registry_state(api):
    response = api.get("/healthz")
    assert response.status_code == 200
    assert response.data["models"]["cfpb"]["triage"] == "null"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/complaints/test_api.py -v`
Expected: FAIL — 404 on every route

- [ ] **Step 3: Add `REST_FRAMEWORK` to `config/settings/base.py`**

```python
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework.authentication.SessionAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
}
```

- [ ] **Step 4: Write `complaints/api.py`**

```python
from django.db.models import QuerySet
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from complaints import services
from complaints.models import Complaint, Priority
from domains.models import Category
from ml.registry import registry_status


class ComplaintSerializer(serializers.ModelSerializer):
    submitted_by = serializers.SlugRelatedField(slug_field="username", read_only=True)
    is_overdue = serializers.BooleanField(read_only=True)

    class Meta:
        model = Complaint
        fields = [
            "id",
            "domain",
            "category",
            "priority",
            "status",
            "title",
            "body",
            "submitted_by",
            "assignee",
            "duplicate_of",
            "created_at",
            "triaged_at",
            "due_at",
            "resolved_at",
            "is_overdue",
        ]
        read_only_fields = [
            "category",
            "priority",
            "status",
            "submitted_by",
            "assignee",
            "duplicate_of",
            "created_at",
            "triaged_at",
            "due_at",
            "resolved_at",
        ]


class TriageSerializer(serializers.Serializer):
    category = serializers.PrimaryKeyRelatedField(queryset=Category.objects.all())
    priority = serializers.ChoiceField(choices=Priority.choices)


class ComplaintViewSet(viewsets.ModelViewSet):
    serializer_class = ComplaintSerializer
    http_method_names = ["get", "post", "head", "options"]

    def get_queryset(self) -> QuerySet[Complaint]:
        queryset = Complaint.objects.select_related("domain", "category", "submitted_by")
        if self.request.user.has_perm("complaints.view_queue"):
            return queryset
        return queryset.filter(submitted_by=self.request.user)

    def perform_create(self, serializer) -> None:
        serializer.save(submitted_by=self.request.user)

    @action(detail=True, methods=["post"])
    def triage(self, request, pk=None):
        if not request.user.has_perm("complaints.triage_complaint"):
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        complaint = self.get_object()
        payload = TriageSerializer(data=request.data)
        payload.is_valid(raise_exception=True)
        try:
            services.triage(
                complaint,
                payload.validated_data["category"],
                payload.validated_data["priority"],
                request.user,
            )
        except services.InvalidTransition as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        complaint.refresh_from_db()
        return Response(ComplaintSerializer(complaint).data)

    @action(detail=True, methods=["post"])
    def resolve(self, request, pk=None):
        if not request.user.has_perm("complaints.resolve_complaint"):
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        complaint = self.get_object()
        try:
            services.resolve(complaint, request.user, note=request.data.get("note", ""))
        except services.InvalidTransition as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        complaint.refresh_from_db()
        return Response(ComplaintSerializer(complaint).data)


@api_view(["GET"])
@permission_classes([AllowAny])
def healthz(request):
    """Reports which model version serves each domain.

    A deployment that lost its artifacts says so here rather than silently
    getting worse.
    """
    return Response({"status": "ok", "models": registry_status()})
```

- [ ] **Step 5: Write `complaints/urls.py` and register it**

```python
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from complaints.api import ComplaintViewSet, healthz

router = DefaultRouter()
router.register("complaints", ComplaintViewSet, basename="complaint")

urlpatterns = [
    path("api/", include(router.urls)),
    path("healthz", healthz, name="healthz"),
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/complaints/test_api.py -v`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add complaints/api.py complaints/urls.py config/settings/base.py tests/complaints/test_api.py
git commit -m "feat: DRF API with queryset-scoped visibility

Submitters see only their own complaints; agents see the queue. Invalid
transitions surface as 400, not 500."
```

---

### Task 13: Server-rendered UI

**Files:**
- Create: `templates/base.html`, `templates/complaints/list.html`, `templates/complaints/detail.html`, `templates/complaints/submit.html`, `templates/complaints/queue.html`, `static/css/sentinel.css`
- Modify: `complaints/views.py`, `complaints/urls.py`
- Test: `tests/complaints/test_views.py`

**Interfaces:**
- Consumes: `complaints.services`, `complaints.models`
- Produces: named URLs `complaint-list`, `complaint-detail`, `complaint-submit`, `complaint-queue`

- [ ] **Step 1: Write the failing test**

Create `tests/complaints/test_views.py`:

```python
import pytest
from django.contrib.auth.models import Group
from django.urls import reverse

from complaints.models import Complaint, Status
from complaints.permissions import AGENT, bootstrap_groups
from tests.factories import CategoryFactory, ComplaintFactory, DomainFactory, UserFactory


@pytest.mark.django_db
def test_submit_page_requires_login(client):
    response = client.get(reverse("complaint-submit"))
    assert response.status_code == 302
    assert "/accounts/" in response.url


@pytest.mark.django_db
def test_submitting_creates_a_complaint_in_submitted_state(client):
    bootstrap_groups()
    user = UserFactory()
    domain = DomainFactory()
    client.force_login(user)
    response = client.post(
        reverse("complaint-submit"),
        {"domain": domain.pk, "title": "Double charged", "body": "Billed twice in March."},
    )
    assert response.status_code == 302
    complaint = Complaint.objects.get()
    assert complaint.status == Status.SUBMITTED
    assert complaint.submitted_by == user
    assert complaint.category is None


@pytest.mark.django_db
def test_submitter_cannot_open_another_users_complaint(client):
    bootstrap_groups()
    intruder = UserFactory()
    someone_elses = ComplaintFactory()
    client.force_login(intruder)
    response = client.get(reverse("complaint-detail", args=[someone_elses.pk]))
    assert response.status_code == 404


@pytest.mark.django_db
def test_queue_is_closed_to_submitters(client):
    bootstrap_groups()
    client.force_login(UserFactory())
    assert client.get(reverse("complaint-queue")).status_code == 403


@pytest.mark.django_db
def test_queue_lists_open_complaints_for_agents(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    ComplaintFactory(title="Open one")
    client.force_login(agent)
    response = client.get(reverse("complaint-queue"))
    assert response.status_code == 200
    assert b"Open one" in response.content


@pytest.mark.django_db
def test_agent_triage_form_moves_the_complaint_to_in_review(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    category = CategoryFactory(sla_hours=12)
    complaint = ComplaintFactory(domain=category.domain)
    client.force_login(agent)
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "triage", "category": category.pk, "priority": "high"},
    )
    assert response.status_code == 302
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_REVIEW
    assert complaint.due_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/complaints/test_views.py -v`
Expected: FAIL — `NoReverseMatch: 'complaint-submit' is not a valid view function or pattern name`

- [ ] **Step 3: Write `complaints/views.py`**

Implement four views. Use `LoginRequiredMixin` / `login_required`, and `PermissionRequiredMixin` with `raise_exception = True` on the queue so an unauthorized user gets 403 rather than a redirect loop.

```python
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin
from django.shortcuts import get_object_or_404, redirect, render
from django.views.generic import ListView

from complaints import services
from complaints.models import Complaint, Priority, Status
from domains.models import Category, Domain


class ComplaintListView(LoginRequiredMixin, ListView):
    """A submitter's own complaints."""

    template_name = "complaints/list.html"
    context_object_name = "complaints"
    paginate_by = 25

    def get_queryset(self):
        return Complaint.objects.filter(submitted_by=self.request.user).select_related(
            "domain", "category"
        )


class QueueView(LoginRequiredMixin, PermissionRequiredMixin, ListView):
    """The agent work queue: everything not yet finished."""

    permission_required = "complaints.view_queue"
    raise_exception = True
    template_name = "complaints/queue.html"
    context_object_name = "complaints"
    paginate_by = 25

    def get_queryset(self):
        return (
            Complaint.objects.exclude(status__in=[Status.CLOSED, Status.DUPLICATE])
            .select_related("domain", "category", "assignee")
            .order_by("due_at", "-created_at")
        )


@login_required
def submit(request):
    if request.method == "POST":
        complaint = Complaint.objects.create(
            domain=get_object_or_404(Domain, pk=request.POST["domain"]),
            title=request.POST["title"],
            body=request.POST["body"],
            submitted_by=request.user,
        )
        return redirect("complaint-detail", pk=complaint.pk)
    return render(
        request, "complaints/submit.html", {"domains": Domain.objects.filter(is_active=True)}
    )


@login_required
def detail(request, pk):
    queryset = Complaint.objects.select_related("domain", "category", "submitted_by")
    if not request.user.has_perm("complaints.view_queue"):
        queryset = queryset.filter(submitted_by=request.user)
    complaint = get_object_or_404(queryset, pk=pk)

    if request.method == "POST":
        action = request.POST.get("action")
        try:
            if action == "triage" and request.user.has_perm("complaints.triage_complaint"):
                category = get_object_or_404(Category, pk=request.POST["category"])
                services.triage(complaint, category, request.POST["priority"], request.user)
            elif action == "resolve" and request.user.has_perm("complaints.resolve_complaint"):
                services.resolve(complaint, request.user)
        except services.InvalidTransition as exc:
            from django.contrib import messages

            messages.error(request, str(exc))
        return redirect("complaint-detail", pk=complaint.pk)

    return render(
        request,
        "complaints/detail.html",
        {
            "complaint": complaint,
            "events": complaint.events.select_related("actor", "prediction"),
            "categories": Category.objects.filter(domain=complaint.domain),
            "priorities": Priority.choices,
        },
    )
```

- [ ] **Step 4: Add the routes to `complaints/urls.py`**

```python
from complaints.views import ComplaintListView, QueueView, detail, submit

urlpatterns += [
    path("", ComplaintListView.as_view(), name="complaint-list"),
    path("complaints/submit/", submit, name="complaint-submit"),
    path("complaints/queue/", QueueView.as_view(), name="complaint-queue"),
    path("complaints/<int:pk>/", detail, name="complaint-detail"),
]
```

- [ ] **Step 5: Write the templates**

`templates/base.html` — a single `<nav>`, a `{% block content %}`, messages rendering, and one `<link>` to `sentinel.css`. Show the queue link only when `perms.complaints.view_queue` is true.

`templates/complaints/list.html`, `queue.html` — tables of complaints. The queue shows `due_at` and highlights rows where `complaint.is_overdue`.

`templates/complaints/submit.html` — domain select, title, body, `{% csrf_token %}`.

`templates/complaints/detail.html` — complaint fields, the event timeline, and an agent-only triage form guarded by `{% if perms.complaints.triage_complaint %}`. Leave a clearly marked empty block where Phase 3 inserts suggestions and duplicate candidates.

All templates escape user content by Django default. Do not add `|safe` anywhere.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/complaints/test_views.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add complaints/views.py complaints/urls.py templates/ static/ tests/complaints/test_views.py
git commit -m "feat: server-rendered complaint UI

Submitters see their own complaints; agents get a due-date-ordered
queue. A submitter requesting another user's complaint gets a 404, not a
403, so the API does not confirm the record exists."
```

---

### Task 14: Demo seed data

**Files:**
- Create: `complaints/management/commands/seed_demo.py`
- Test: `tests/test_seed.py`

**Interfaces:**
- Consumes: `domains.packs.PACKS`, `complaints.permissions.bootstrap_groups`
- Produces: management command `seed_demo`

- [ ] **Step 1: Write the failing test**

Create `tests/test_seed.py`:

```python
import pytest
from django.core.management import call_command

from complaints.models import Complaint
from domains.models import Domain
from domains.packs import PACKS


@pytest.mark.django_db
def test_seed_creates_a_domain_row_for_every_registered_pack():
    call_command("seed_demo")
    assert set(Domain.objects.values_list("slug", flat=True)) == set(PACKS)


@pytest.mark.django_db
def test_seed_creates_complaints():
    call_command("seed_demo")
    assert Complaint.objects.exists()


@pytest.mark.django_db
def test_seed_is_idempotent():
    """The deploy runs this on every boot; it must not multiply rows."""
    call_command("seed_demo")
    first = Complaint.objects.count()
    call_command("seed_demo")
    assert Complaint.objects.count() == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_seed.py -v`
Expected: FAIL — `CommandError: Unknown command: 'seed_demo'`

- [ ] **Step 3: Write `complaints/management/commands/seed_demo.py`**

The command must:
1. Call `bootstrap_groups()`.
2. For every slug in `PACKS`, `get_or_create` a `Domain` using `pack.display_name`.
3. `get_or_create` categories per domain — for `cfpb`: `mortgage` (72h), `credit_card` (48h), `debt_collection` (72h), `other` (96h); for `nyc311`: `noise` (24h), `street_condition` (120h), `sanitation` (48h), `other` (96h).
4. `get_or_create` a demo agent (`demo-agent`) in the Agent group and a demo submitter (`demo-user`).
5. `get_or_create` six complaints keyed on `title` so re-running does not duplicate them — two left `SUBMITTED`, two triaged via `services.triage`, one resolved via `services.resolve`, one marked duplicate via `services.mark_duplicate`.

Use the service layer for every state change so seeded complaints carry a real event history rather than being written straight to the database.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_seed.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add complaints/management/commands/seed_demo.py tests/test_seed.py
git commit -m "feat: idempotent demo seed

Seeded complaints move through the service layer, so they carry real
event histories rather than fabricated state."
```

---

### Task 15: CI and deployment

**Files:**
- Create: `.github/workflows/ci.yml`, `render.yaml`, `build.sh`, `README.md`
- Test: the workflow itself

**Interfaces:**
- Consumes: everything
- Produces: a green CI run and a deployed URL

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      SECRET_KEY: ci-only-key
      DEBUG: "False"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -r requirements.txt
      - name: Lint
        run: ruff check .
      - name: Format check
        run: ruff format --check .
      - name: Type check
        run: mypy complaints domains ml
      - name: Check for missing migrations
        run: python manage.py makemigrations --check --dry-run
      - name: Test
        run: pytest --cov=complaints --cov=domains --cov=ml --cov-report=term-missing
```

The `makemigrations --check` step is what stops a model change from reaching production without its migration.

- [ ] **Step 2: Run the full suite locally and fix anything failing**

```bash
.venv/Scripts/ruff check .
.venv/Scripts/ruff format --check .
.venv/Scripts/mypy complaints domains ml
.venv/Scripts/python manage.py makemigrations --check --dry-run
.venv/Scripts/pytest --cov=complaints --cov=domains --cov=ml --cov-report=term-missing
```

Expected: all green, 60+ tests passing.

- [ ] **Step 3: Write `build.sh` and `render.yaml`**

`build.sh`:

```bash
#!/usr/bin/env bash
set -o errexit
pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate
python manage.py bootstrap_groups
python manage.py seed_demo
```

`render.yaml` declares a web service running
`gunicorn config.wsgi:application --preload --workers 2`
with `DJANGO_SETTINGS_MODULE=config.settings.prod`, a managed Postgres, and
`SECRET_KEY`, `ALLOWED_HOSTS`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` set in
the dashboard — never committed.

Worker count is provisional. Phase 3 measures RSS under load and fixes it; if
two workers do not fit, the answer is one.

- [ ] **Step 4: Write `README.md`**

Cover: what Sentinel is, the four architectural principles from the spec, local
setup, the env var table, how to run tests, the current phase status, and a
placeholder section titled "Model performance" that Phase 2 fills with real
numbers. Do not write any metric that has not been measured.

- [ ] **Step 5: Deploy and verify**

Push, connect the repo to Render, set the env vars, deploy. Then verify by hand:

1. `GET /healthz` returns `{"status": "ok", "models": {...: "null"}}`.
2. Google sign-in completes (after adding the production callback URL to the existing OAuth client).
3. Submit a complaint as a normal user.
4. Elevate a user to Agent in the Django admin, then triage that complaint and confirm `due_at` appears.

- [ ] **Step 6: Commit**

```bash
git add .github/ render.yaml build.sh README.md
git commit -m "ci: lint, type check, migration check and tests; Render deploy config"
```

---

## Phase 1 Exit Criteria

Phase 1 is complete when all of these hold:

- [ ] A human can submit, triage, assign, resolve, and mark duplicates through the UI.
- [ ] A submitter cannot see, open, or act on another user's complaint.
- [ ] The full permission matrix passes, negatives included.
- [ ] Every state change has written a `ComplaintEvent`; no complaint has state without history.
- [ ] `Prediction` update and delete both raise.
- [ ] The three duplicate invariants hold: canonical required, not self, no chains.
- [ ] `due_at` derives from `triaged_at`, and `triaged_at` is only ever set by human confirmation.
- [ ] The application runs correctly with zero model artifacts installed.
- [ ] `/healthz` reports `null` for every model.
- [ ] CI is green: ruff, mypy, no missing migrations, all tests passing.
- [ ] The app is deployed at a public URL with seeded demo data and working Google sign-in.
- [ ] No secret appears in the repository or in any database row.

## What Phase 1 deliberately does not do

No dataset ingest, no training, no real models, no embeddings written, no
suggestion UI, no risk-ranked queue, no model card page. Those are Phases 2 and
3. Phase 1's value is that the application is genuinely finished and deployed
before any model exists — so if the ML work overruns, what already exists still
stands on its own.

**One spec item is deliberately deferred rather than dropped.** Spec §8 requires
request IDs on log lines. Phase 1 ships structured JSON logging but no
request-ID middleware, because the thing request IDs exist to correlate — ML
failures carrying complaint ID and model version — does not exist until Phase 3.
The middleware lands in Phase 3 alongside it. Recorded here so it is tracked,
not forgotten.

**Two steps are specified in prose rather than code** and are the ones to
scrutinise during review: Task 13 Step 5 (templates) and Task 14 Step 3 (seed
command). Both name their exact required behavior — permission guards, category
slugs, SLA hours, idempotency keys — but an implementer has real latitude in
how they write them. Everything else in this plan is literal.
