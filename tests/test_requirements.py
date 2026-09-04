"""The dependency tiers must account for every Phase 1 package, and production
must not carry test or training tooling.

Phase 2 introduces scikit-learn, onnxruntime, pandas and pyarrow. Left in one
flat file installed by build.sh, they would ship to a 512MB instance that never
uses them, and the Phase 3 memory measurement would include a dataframe library
the web process never imports.
"""

import re
from pathlib import Path

REQUIREMENTS_DIR = Path(__file__).resolve().parent.parent / "requirements"

# The Phase 1 freeze, recorded so the split cannot silently drop a package.
# Normalised per PEP 503.
PHASE_1_PACKAGES = frozenset(
    {
        "asgiref",
        "ast-serialize",
        "certifi",
        "charset-normalizer",
        "colorama",
        "coverage",
        "django",
        "django-allauth",
        "django-environ",
        "django-stubs",
        "django-stubs-ext",
        "djangorestframework",
        "factory-boy",
        "faker",
        "gunicorn",
        "idna",
        "iniconfig",
        "librt",
        "mypy",
        "mypy-extensions",
        "packaging",
        "pathspec",
        "pluggy",
        "psycopg2-binary",
        "pygments",
        "pytest",
        "pytest-cov",
        "pytest-django",
        "requests",
        "ruff",
        "sqlparse",
        "types-pyyaml",
        "typing-extensions",
        "tzdata",
        "urllib3",
        "whitenoise",
    }
)

# Nothing in this list may appear in base.txt: production installs base.txt only.
FORBIDDEN_IN_BASE = frozenset(
    {
        "pytest",
        "pytest-django",
        "pytest-cov",
        "coverage",
        "factory-boy",
        "faker",
        "ruff",
        "mypy",
        "django-stubs",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "onnxruntime",
        "numpy",
    }
)

TIERS = ("base", "ml", "train", "dev")


def normalize(name: str) -> str:
    """PEP 503 canonical form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_tier(tier: str) -> list[str]:
    return (REQUIREMENTS_DIR / f"{tier}.txt").read_text(encoding="utf-8").splitlines()


def packages_in(tier: str) -> set[str]:
    """Package names pinned directly in a tier, ignoring -r includes and comments."""
    found = set()
    for line in read_tier(tier):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        found.add(normalize(re.split(r"[=<>!~\[]", line, maxsplit=1)[0]))
    return found


def test_every_tier_file_exists():
    for tier in TIERS:
        assert (REQUIREMENTS_DIR / f"{tier}.txt").is_file(), f"missing {tier}.txt"


def test_every_phase_1_package_appears_in_exactly_one_tier():
    seen: dict[str, list[str]] = {}
    for tier in TIERS:
        for pkg in packages_in(tier):
            seen.setdefault(pkg, []).append(tier)

    missing = PHASE_1_PACKAGES - seen.keys()
    assert not missing, f"Phase 1 packages dropped by the split: {sorted(missing)}"

    duplicated = {p: t for p, t in seen.items() if p in PHASE_1_PACKAGES and len(t) > 1}
    assert not duplicated, f"packages pinned in more than one tier: {duplicated}"


def test_base_carries_no_test_or_training_package():
    offenders = packages_in("base") & FORBIDDEN_IN_BASE
    assert not offenders, (
        f"base.txt is installed in production by build.sh; it must not carry {sorted(offenders)}"
    )


def test_layered_tiers_include_their_parent():
    """ml, train and dev build on base rather than restating it."""
    for tier in ("ml", "train", "dev"):
        first = next(line for line in read_tier(tier) if line.strip() and not line.startswith("#"))
        assert first.startswith("-r "), f"{tier}.txt must start with a -r include, got {first!r}"


def test_train_includes_ml_and_ml_includes_base():
    ml_first = next(line for line in read_tier("ml") if line.strip() and not line.startswith("#"))
    train_first = next(
        line for line in read_tier("train") if line.strip() and not line.startswith("#")
    )
    assert "base.txt" in ml_first, "ml.txt must include base.txt"
    assert "ml.txt" in train_first, "train.txt must include ml.txt, not base.txt directly"


def test_requirements_txt_is_a_shim_for_base():
    """External tooling referring to requirements.txt must keep working."""
    root = REQUIREMENTS_DIR.parent / "requirements.txt"
    body = [line.strip() for line in root.read_text(encoding="utf-8").splitlines()]
    directives = [line for line in body if line and not line.startswith("#")]
    assert directives == ["-r requirements/base.txt"], (
        f"requirements.txt should be a one-line shim, got {directives}"
    )
