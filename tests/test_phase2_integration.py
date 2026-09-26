"""Task 22: the Phase 2 boundary, verified across the whole branch.

Plan §U Task 22 asserts what no single task can prove on its own: that Phase 2,
taken as a whole, stayed offline. The plan's coverage matrix and its finding I3
assign Task 22 four guarantees beyond the five its entry lists, and it owns all
nine:

    G1  no project migration beyond Phase 1's
    G2  ml/registry.py still resolves every model to its null implementation
    G3  no Phase 2 code path can write a Prediction row
    G4  complaints/ contains no source-slug literal
    G5  the application suite passes without the training tier installed
    G6  corpus records never enter Complaint, and no Complaint.pk is synthesised
    G7  Match and DedupIndex are unchanged
    G8  the Phase 3 serving interface is not mutated
    G9  git tracks no Parquet file, no model binary and nothing under data/

Each guarantee is one test named ``test_gN_...``. A checker that passes because it
found nothing to look at proves nothing, so every checker is also fed planted input
by self-tests (``test_gN_checker_...``, ``test_gN_harness_...``,
``test_g5_child_...``), which assert that a violation is caught and that legitimate
input is not -- the pattern tests/test_docs.py established.
Planted violations live in pytest's temporary directory or in memory: nothing
here writes to the repository.

G1, G7 and G8 compare against Phase 1 as it stood at PHASE_1_BOUNDARY. The pinned
values are checked against that commit by ``test_gN_pins_...`` tests, which need
the commit object and therefore skip in a shallow clone such as CI's default
checkout. The guarantees themselves never skip.
"""

import ast
import dataclasses
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import types
import typing
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pytest
from django.conf import settings
from django.core.management import call_command
from django.db import models
from django.db.migrations.loader import MigrationLoader

from complaints.models import Complaint
from domains.packs import PACKS
from ingest.identity import RecordRef, make_ref, parse_ref
from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome
from ml import base, registry
from ml.null import NULL_VERSION, NullDedupIndex, NullRiskModel, NullTriageModel

ROOT = Path(__file__).resolve().parent.parent

PHASE_1_BOUNDARY = "6657ad72fd806589a86834c1c4f6871c17649463"
"""Phase 1's last commit ("docs: record Phase 1 carried-forward items"): the tip of
master, and the parent of the first Phase 2 commit, which added the Phase 2 addendum."""


def git(
    *args: str, cwd: Path = ROOT, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run git. `env` of None inherits the caller's environment, which is right for the
    read-only queries this file makes of the repository itself."""
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def without_git_environment() -> dict[str, str]:
    """The caller's environment minus every GIT_* variable.

    git exports GIT_INDEX_FILE to a pre-commit hook, and a tool or shell may export
    GIT_DIR, GIT_WORK_TREE or GIT_OBJECT_DIRECTORY. Any of them would steer a temporary
    repository's commands at another repository's index, so a temporary repository is
    always driven with this environment instead.
    """
    return {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}


def require_phase_1_boundary() -> None:
    """Skip, saying why, when this clone does not hold the Phase 1 boundary commit."""
    if git("cat-file", "-e", f"{PHASE_1_BOUNDARY}^{{commit}}").returncode != 0:
        pytest.skip("the Phase 1 boundary commit is not in this clone (a shallow checkout)")
    assert git("merge-base", "--is-ancestor", PHASE_1_BOUNDARY, "HEAD").returncode == 0, (
        "PHASE_1_BOUNDARY is not an ancestor of HEAD"
    )


def write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def load_source_as_module(
    name: str, source: str, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """Execute `source` as a registered module, so its annotations resolve as a real one's."""
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


# --- G1. no project migration beyond Phase 1's ------------------------------------------

PHASE_1_MIGRATIONS: dict[tuple[str, str], str] = {
    ("complaints", "0001_initial"): "dc3767d7078827f65c524a3b2210b37bcfc65dec",
    ("complaints", "0002_prediction"): "59296a2159b4945cb98e9fe045a2c68e241acd66",
    ("complaints", "0003_complaintevent"): "960e3cfd2440750e6f73c79a05389137c08254aa",
    ("domains", "0001_initial"): "551f5982c36b2b5564877dd5048e6256356421ff",
}
"""Every project migration at PHASE_1_BOUNDARY, with the git blob id of its content.

Pinning content as well as names means an edited Phase 1 migration fails exactly as an
added one does. `accounts` had no migrations and must still have none."""

PHASE_1_LEAVES = {("complaints", "0003_complaintevent"), ("domains", "0001_initial")}
"""Where each project app's migration graph ended at PHASE_1_BOUNDARY."""


def git_blob_id(data: bytes) -> str:
    """The id git gives a file's content: SHA-1 over a ``blob <size>\\0`` header.

    Line endings are normalised to LF first, as git stores them under core.autocrlf, so
    a Windows checkout (CRLF on disk) hashes to the same committed blob as CI's.
    """
    data = data.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def project_app_labels() -> list[str]:
    """Installed apps whose package sits directly in the repository root."""
    from django.apps import apps

    return sorted(
        config.label
        for config in apps.get_app_configs()
        if Path(config.path).resolve().parent == ROOT
    )


def migration_files(root: Path, app_labels: Iterable[str]) -> dict[tuple[str, str], str]:
    found = {}
    for label in app_labels:
        directory = root / label / "migrations"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name != "__init__.py":
                found[(label, path.stem)] = git_blob_id(path.read_bytes())
    return found


def migration_differences(
    actual: dict[tuple[str, str], str], expected: dict[tuple[str, str], str]
) -> list[str]:
    differences = [f"{app}.{name}: added after Phase 1" for app, name in actual.keys() - expected]
    differences += [
        f"{app}.{name}: removed since Phase 1" for app, name in expected.keys() - actual
    ]
    differences += [
        f"{app}.{name}: edited since Phase 1"
        for app, name in actual.keys() & expected.keys()
        if actual[(app, name)] != expected[(app, name)]
    ]
    return sorted(differences)


@pytest.mark.django_db
def test_g1_no_project_migration_beyond_phase_1():
    """Addendum §8 and §10: Phase 2 produces no schema migration."""
    labels = project_app_labels()
    assert {"accounts", "complaints", "domains"} <= set(labels), labels

    on_disk = migration_files(ROOT, labels)
    assert not migration_differences(on_disk, PHASE_1_MIGRATIONS), migration_differences(
        on_disk, PHASE_1_MIGRATIONS
    )

    # The files scanned are the migrations Django loads, and the graph still ends where
    # Phase 1 left it.
    loader = MigrationLoader(None, ignore_no_migrations=True)
    assert {key for key in loader.disk_migrations if key[0] in labels} == set(PHASE_1_MIGRATIONS)
    assert {node for node in loader.graph.leaf_nodes() if node[0] in labels} == PHASE_1_LEAVES

    # Plan §U Task 22's acceptance: no model change is waiting for a migration either.
    call_command("makemigrations", "--check", "--dry-run", verbosity=0)


def test_g1_checker_catches_an_added_edited_or_removed_migration(tmp_path):
    for label in ("complaints", "domains"):
        shutil.copytree(
            ROOT / label / "migrations",
            tmp_path / label / "migrations",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    labels = ("accounts", "complaints", "domains")
    assert migration_files(tmp_path, labels) == PHASE_1_MIGRATIONS

    dummy = tmp_path / "complaints" / "migrations" / "0004_phase_2_dummy.py"
    dummy.write_text("from django.db import migrations\n", encoding="utf-8")
    assert migration_differences(migration_files(tmp_path, labels), PHASE_1_MIGRATIONS) == [
        "complaints.0004_phase_2_dummy: added after Phase 1"
    ]
    dummy.unlink()

    edited = tmp_path / "domains" / "migrations" / "0001_initial.py"
    lf = edited.read_bytes().replace(b"\r\n", b"\n")
    edited.write_bytes(lf + b"# edited\n")
    assert migration_differences(migration_files(tmp_path, labels), PHASE_1_MIGRATIONS) == [
        "domains.0001_initial: edited since Phase 1"
    ]
    # Whatever line endings the checkout used, the same content is the same blob.
    for checkout in (lf, lf.replace(b"\n", b"\r\n")):
        edited.write_bytes(checkout)
        assert migration_files(tmp_path, labels) == PHASE_1_MIGRATIONS

    (tmp_path / "complaints" / "migrations" / "0003_complaintevent.py").unlink()
    assert migration_differences(migration_files(tmp_path, labels), PHASE_1_MIGRATIONS) == [
        "complaints.0003_complaintevent: removed since Phase 1"
    ]


def test_g1_pins_are_the_phase_1_boundary():
    require_phase_1_boundary()
    listing = git("ls-tree", "-r", PHASE_1_BOUNDARY).stdout.splitlines()
    pinned = {}
    for line in listing:
        meta, path = line.split("\t", 1)
        match = re.fullmatch(r"([^/]+)/migrations/([^/]+)\.py", path)
        if match and match.group(2) != "__init__":
            pinned[(match.group(1), match.group(2))] = meta.split()[2]
    assert pinned == PHASE_1_MIGRATIONS


# --- G2. the registry still resolves every model to its null implementation -------------

MODEL_KINDS = ("triage", "dedup", "risk")

NULL_MODELS = {
    "triage": ("get_triage_model", NullTriageModel),
    "dedup": ("get_dedup_index", NullDedupIndex),
    "risk": ("get_risk_model", NullRiskModel),
}


def registry_violations() -> list[str]:
    found = []
    if settings.ML_ARTIFACT_VERSIONS != {}:
        found.append(f"ML_ARTIFACT_VERSIONS pins artifacts: {settings.ML_ARTIFACT_VERSIONS!r}")
    if tuple(registry.MODEL_KINDS) != MODEL_KINDS:
        found.append(f"the registry's model kinds changed: {registry.MODEL_KINDS!r}")
    expected_status = {slug: dict.fromkeys(MODEL_KINDS, NULL_VERSION) for slug in PACKS}
    status = registry.registry_status()
    if status != expected_status:
        found.append(f"registry_status() is not null everywhere: {status!r}")
    for slug in PACKS:
        for getter, null_class in NULL_MODELS.values():
            model = getattr(registry, getter)(slug)
            if type(model) is not null_class:
                found.append(f"{getter}({slug!r}) returned {type(model).__name__}")
    return found


def test_g2_registry_still_resolves_every_model_to_null():
    """Addendum §8: the registry resolves to null implementations until Phase 3."""
    assert NULL_VERSION == "null"
    assert len(PACKS) >= 2, "every pack must be checked, so there must be packs to check"
    assert not registry_violations(), registry_violations()


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_g2_checker_catches_a_non_null_model(monkeypatch, kind):
    getter, _ = NULL_MODELS[kind]
    monkeypatch.setattr(registry, getter, lambda slug: object())
    violations = registry_violations()
    assert len(violations) == len(PACKS), violations
    assert all(f"{getter}(" in violation for violation in violations)


def test_g2_checker_catches_a_pinned_artifact_version(settings):
    slug = next(iter(PACKS))
    settings.ML_ARTIFACT_VERSIONS = {slug: {"triage": "v1"}}
    violations = registry_violations()
    assert any(v.startswith("ML_ARTIFACT_VERSIONS pins") for v in violations), violations
    assert any(v.startswith("registry_status()") for v in violations), violations


# --- G3. no Phase 2 code path can write a Prediction row --------------------------------

PHASE_2_PACKAGES = ("ingest", "ml/training", "ml/embedders")
"""Plan §D's two new packages, and ml/embedders, which D38 puts under the same rule."""

FORBIDDEN_ROOTS = frozenset(
    {"django", "rest_framework", "allauth", "complaints", "sqlite3", "psycopg", "psycopg2"}
)
"""Everything a Prediction row can be written through: Django and its ecosystem, the
app that owns Prediction, and the database drivers beneath them."""

DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})


def dotted(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def project_file(name: str, root: Path) -> Path | None:
    """The file defining module `name` inside `root`, or None when it lives elsewhere."""
    parts = name.split(".")
    module = root.joinpath(*parts[:-1], f"{parts[-1]}.py")
    if module.is_file():
        return module
    package = root.joinpath(*parts, "__init__.py")
    return package if package.is_file() else None


def package_of(path: Path, root: Path) -> str:
    """The package a module's relative imports resolve against."""
    name = dotted(path, root)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def resolve_import_from(node: ast.ImportFrom, package: str) -> str:
    """The absolute module a ``from ... import`` names, relative forms included."""
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    kept = parts[: len(parts) - node.level + 1] if node.level > 1 else parts
    return ".".join([*kept, node.module] if node.module else kept)


def module_imports(path: Path, root: Path) -> tuple[set[str], list[str]]:
    """Every module `path` imports, and every import it makes that cannot be read statically.

    As tests/test_import_boundaries.py does, relative imports are resolved and a dynamic
    import is followed when its target is a string literal. Unlike that test, an import
    this reader cannot follow is reported rather than ignored: this guarantee claims every
    code path, so an import it cannot read is a failure, not a blind spot. That covers a
    dynamic import with a computed target, ``import_module`` or ``__import__`` bound
    under another name, and either one referenced other than by being called -- stored,
    passed or returned -- since the call that eventually uses it cannot be read here.
    Calling them by their own names, and every ordinary import statement, stay readable.
    ``from package import name`` also yields ``package.name``, which is how a submodule
    imported by name is reached.
    """
    package = package_of(path, root)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    unreadable: list[str] = []
    callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = resolve_import_from(node, package)
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")
            for alias in node.names:
                if alias.name in DYNAMIC_IMPORTERS and alias.asname not in (None, alias.name):
                    unreadable.append(
                        f"{alias.name} imported as {alias.asname} at line {node.lineno}"
                    )
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called in DYNAMIC_IMPORTERS:
                target = node.args[0] if node.args else None
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    found.add(target.value)
                else:
                    unreadable.append(f"{called}() at line {node.lineno} has a computed target")
        elif isinstance(node, ast.Name | ast.Attribute) and id(node) not in callees:
            referenced = node.id if isinstance(node, ast.Name) else node.attr
            if referenced in DYNAMIC_IMPORTERS:
                unreadable.append(
                    f"{referenced} referenced without being called at line {node.lineno}"
                )
    return found, unreadable


def import_closure(
    root: Path, packages: Sequence[str]
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    """Every forbidden import reachable from `packages`, and every module reached.

    Reachability follows imports into any module defined inside `root`, including the
    parent packages whose ``__init__`` an import executes, so a Phase 2 module cannot
    reach Django through a project helper that imports it. A module outside `root` is
    a leaf, judged only by FORBIDDEN_ROOTS. Each module reached maps to the import
    chain that reached it, which is what a violation reports.
    """
    reached: dict[str, tuple[str, ...]] = {}
    queue: deque[Path] = deque()
    for package in packages:
        for path in sorted((root / package).rglob("*.py")):
            reached[dotted(path, root)] = (dotted(path, root),)
            queue.append(path)
    violations: list[str] = []
    while queue:
        path = queue.popleft()
        chain = reached[dotted(path, root)]
        imported, unreadable = module_imports(path, root)
        violations += [f"{' -> '.join(chain)}: {item}" for item in unreadable]
        for name in sorted(imported):
            if name.split(".")[0] in FORBIDDEN_ROOTS:
                violations.append(f"{' -> '.join(chain)} imports {name}")
                continue
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                target = ".".join(parts[:depth])
                file = project_file(target, root)
                if file is not None and target not in reached:
                    reached[target] = (*chain, target)
                    queue.append(file)
    return violations, reached


def test_g3_no_phase_2_code_path_can_write_a_prediction():
    """Plan §U Task 22, addendum §8 and §10: Phase 2 writes no Prediction row.

    Proven statically, over every Phase 2 module and everything they import from the
    project, rather than by running a finite set of paths: nothing Phase 2 can import
    reaches Django, the app that owns Prediction, or a database driver. A write through
    some other channel is outside what an import analysis can see, and is not claimed:
    a subprocess running manage.py, source text run through exec or eval, a callable
    fetched by a string (getattr), or a module loaded through importlib.util's loaders.
    """
    violations, reached = import_closure(ROOT, PHASE_2_PACKAGES)
    assert not violations, violations

    for package in PHASE_2_PACKAGES:
        prefix = package.replace("/", ".")
        assert sum(name.startswith(f"{prefix}.") for name in reached) > 1, package
    # Real edges are followed, including a submodule imported by name
    # (`from ml.embedders import minilm`).
    assert {"ingest.schema", "ml.embedders.minilm", "ml.training.index", "ml"} <= reached.keys()


@pytest.mark.parametrize(
    ("planted", "scanned", "expected"),
    [
        (
            {"phase2/a.py": "from complaints.models import Prediction\n"},
            "phase2",
            "phase2.a imports complaints.models",
        ),
        ({"phase2/a.py": "import django.db\n"}, "phase2", "phase2.a imports django.db"),
        ({"phase2/a.py": "import sqlite3\n"}, "phase2", "phase2.a imports sqlite3"),
        (
            {
                "phase2/a.py": "from helpers import tools\n",
                "helpers/__init__.py": "",
                "helpers/tools.py": "from django.db import connection\n",
            },
            "phase2",
            "phase2.a -> helpers.tools imports django.db",
        ),
        (
            {
                "phase2/sub/__init__.py": "",
                "phase2/sub/a.py": "from .. import outside\n",
                "phase2/outside.py": "import psycopg2\n",
            },
            "phase2/sub",
            "phase2.sub.a -> phase2.outside imports psycopg2",
        ),
        (
            {"phase2/a.py": "import importlib\n\nimportlib.import_module('django.db')\n"},
            "phase2",
            "phase2.a imports django.db",
        ),
        (
            {"phase2/a.py": "import importlib\n\nname = 'x'\nimportlib.import_module(name)\n"},
            "phase2",
            "phase2.a: import_module() at line 4 has a computed target",
        ),
        (
            {"phase2/a.py": "from importlib import import_module as load\n\nload('django.db')\n"},
            "phase2",
            "phase2.a: import_module imported as load at line 1",
        ),
        (
            {"phase2/a.py": "from builtins import __import__ as fetch\n\nfetch('json')\n"},
            "phase2",
            "phase2.a: __import__ imported as fetch at line 1",
        ),
        (
            {"phase2/a.py": "import importlib\n\nload = importlib.import_module\nload('json')\n"},
            "phase2",
            "phase2.a: import_module referenced without being called at line 3",
        ),
        (
            {"phase2/a.py": "import importlib\n\nlist(map(importlib.import_module, ['json']))\n"},
            "phase2",
            "phase2.a: import_module referenced without being called at line 3",
        ),
    ],
    ids=[
        "prediction-model",
        "django",
        "database-driver",
        "transitive-through-a-project-helper",
        "relative-import",
        "literal-dynamic-import",
        "computed-dynamic-import",
        "renamed-import-module",
        "renamed-dunder-import",
        "import-module-stored",
        "import-module-passed",
    ],
)
def test_g3_checker_catches_a_forbidden_import(tmp_path, planted, scanned, expected):
    write_tree(tmp_path, {"phase2/__init__.py": "", **planted})
    violations, _ = import_closure(tmp_path, [scanned])
    assert expected in violations, violations


def test_g3_checker_passes_a_package_with_no_forbidden_reach(tmp_path):
    """Ordinary imports, including importlib used as itself, stay provably safe."""
    write_tree(
        tmp_path,
        {
            "phase2/__init__.py": "",
            "phase2/a.py": "import json\n\nfrom phase2 import b\n",
            "phase2/b.py": "from . import a\n",
            "phase2/c.py": (
                "import importlib\nimport importlib.util\n"
                "from importlib import import_module\n"
                "from importlib.metadata import version as installed_version\n\n"
                "importlib.import_module('json')\nimport_module('json')\n"
                "found = importlib.util.find_spec('json')\n"
            ),
        },
    )
    violations, reached = import_closure(tmp_path, ["phase2"])
    assert not violations, violations
    assert {"phase2", "phase2.a", "phase2.b", "phase2.c"} <= reached.keys()


# --- G4. complaints/ contains no source-slug literal -------------------------------------


def template_dirs() -> list[Path]:
    return [Path(directory) for engine in settings.TEMPLATES for directory in engine["DIRS"]]


def complaints_surface(root: Path, templates: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """The complaints app's Python source, and its other text: non-Python files under
    complaints/ plus the complaints templates in every configured template directory.

    Templates are included because a template ``{% if %}`` on a slug is the same
    branching the invariant forbids in Python.
    """
    files = [
        path
        for path in sorted((root / "complaints").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    python = [path for path in files if path.suffix == ".py"]
    text = [path for path in files if path.suffix != ".py"]
    for directory in templates:
        text += [path for path in sorted((directory / "complaints").rglob("*")) if path.is_file()]
    return python, text


def slug_literal_violations(
    python: Iterable[Path], text: Iterable[Path], slugs: Iterable[str]
) -> list[str]:
    """Every string literal in `python`, and every line of `text`, naming a slug.

    Case-insensitive, so ``"CFPB"`` counts. A Python comment is not a literal and can
    branch on nothing, so it is not scanned. Everything that is not Python is searched
    as bytes, never decoded: a static image or a compiled translation is legitimate in a
    Django app and must not make the scan fail, while a slug stored inside one -- a
    compiled translation carries its strings verbatim -- is still a slug in complaints/.
    """
    lowered = [slug.lower() for slug in slugs]
    encoded = [slug.encode("utf-8") for slug in lowered]
    found = []
    for path in python:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(slug in node.value.lower() for slug in lowered):
                    found.append(f"{path}:{node.lineno}: {node.value!r}")
    for path in text:
        for number, line in enumerate(path.read_bytes().splitlines(), 1):
            if any(slug in line.lower() for slug in encoded):
                found.append(f"{path}:{number}: {line.strip()!r}")
    return found


def test_g4_complaints_contains_no_source_slug_literal():
    """Invariant 1 and addendum §8: no source-specific literal enters complaints/.

    The slugs are PACKS' own keys, so a pack added later is covered without editing
    this test.
    """
    slugs = sorted(PACKS)
    assert slugs and all(isinstance(slug, str) and slug for slug in slugs)
    python, text = complaints_surface(ROOT, template_dirs())
    assert len(python) > 5, python
    assert any(path.suffix == ".html" for path in text), "no complaints template was scanned"
    assert not slug_literal_violations(python, text, slugs), slug_literal_violations(
        python, text, slugs
    )


def copy_complaints_surface(tmp_path: Path) -> tuple[Path, Path]:
    tree = tmp_path / "repo"
    shutil.copytree(
        ROOT / "complaints", tree / "complaints", ignore=shutil.ignore_patterns("__pycache__")
    )
    templates = tree / "templates"
    for directory in template_dirs():
        shutil.copytree(directory / "complaints", templates / "complaints")
    return tree, templates


@pytest.mark.parametrize("slug", sorted(PACKS))
def test_g4_checker_catches_a_planted_slug(tmp_path, slug):
    tree, templates = copy_complaints_surface(tmp_path)
    assert not slug_literal_violations(*complaints_surface(tree, [templates]), PACKS)

    services = tree / "complaints" / "services.py"
    original = services.read_text(encoding="utf-8")
    services.write_text(original + f'\n_PLANTED = "{slug}"\n', encoding="utf-8")
    assert len(slug_literal_violations(*complaints_surface(tree, [templates]), PACKS)) == 1

    services.write_text(original + f'\n_PLANTED = f"{{1}}-{slug.upper()}"\n', encoding="utf-8")
    assert len(slug_literal_violations(*complaints_surface(tree, [templates]), PACKS)) == 1
    services.write_text(original, encoding="utf-8")

    template = next((templates / "complaints").glob("*.html"))
    template.write_text(
        template.read_text(encoding="utf-8")
        + f'\n{{% if domain.slug == "{slug}" %}}{{% endif %}}\n',
        encoding="utf-8",
    )
    assert len(slug_literal_violations(*complaints_surface(tree, [templates]), PACKS)) == 1


BINARY_ASSET = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\xff\xfe\x81\x8d\x90\x9d\x00"
"""Bytes that are not UTF-8 (0x89, 0xff, 0x81 ...), as in any static image."""


def test_g4_checker_accepts_a_binary_file_without_a_slug(tmp_path):
    """A legitimate static asset under complaints/ neither crashes the scan nor fails it."""
    tree, templates = copy_complaints_surface(tmp_path)
    static = tree / "complaints" / "static" / "complaints"
    static.mkdir(parents=True, exist_ok=True)
    (static / "logo.png").write_bytes(BINARY_ASSET)
    (tree / "complaints" / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1\x00\xff\xfe")
    python, text = complaints_surface(tree, [templates])
    assert static / "logo.png" in text, "the binary file must be scanned, not skipped"
    assert not slug_literal_violations(python, text, PACKS)


@pytest.mark.parametrize("slug", sorted(PACKS))
def test_g4_checker_catches_a_slug_inside_a_binary_file(tmp_path, slug):
    """A compiled translation carries its strings verbatim: a slug inside one counts."""
    tree, templates = copy_complaints_surface(tmp_path)
    catalog = tree / "complaints" / "locale" / "en" / "LC_MESSAGES" / "django.mo"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_bytes(BINARY_ASSET + slug.upper().encode("ascii") + b"\x00" + BINARY_ASSET)
    assert len(slug_literal_violations(*complaints_surface(tree, [templates]), PACKS)) == 1


def test_g4_checker_does_not_treat_a_comment_as_a_literal(tmp_path):
    tree, templates = copy_complaints_surface(tmp_path)
    services = tree / "complaints" / "services.py"
    slug = next(iter(PACKS))
    services.write_text(
        services.read_text(encoding="utf-8") + f"\n# {slug} is named here only\n", encoding="utf-8"
    )
    assert not slug_literal_violations(*complaints_surface(tree, [templates]), PACKS)


# --- G5. the application suite passes without the training tier --------------------------

TRAINING_SENTINELS = ("numpy", "sklearn", "onnxruntime", "tokenizers", "pandas", "pyarrow")
"""Import names of the packages plan §E and D38 place in ml.txt and train.txt. They must
be unimportable in the G5 run whether the tier is absent (CI's application job) or
blocked (a full development environment)."""

APPLICATION_TIER = ("django", "rest_framework", "allauth", "pytest_django")
"""Packages base.txt and dev.txt provide. They must stay importable, or the block would be
too wide for a passing suite to prove anything."""

G5_TIMEOUT_SECONDS = 900

G5_MODULE = Path(__file__).resolve().relative_to(ROOT).as_posix()

G5_DESELECT = ("--deselect", f"{G5_MODULE}::test_g5_")
"""How the child leaves out this module's G5 tests, so it does not recurse: a node-id
prefix, which names exactly those tests. A keyword expression such as ``-k "not g5_"``
would be a substring match over every test's name, module and parameters, and would
silently drop an unrelated test such as ``test_log5_rotation`` from the proof."""

G5_BOOTSTRAP = r"""
import importlib, importlib.abc, importlib.machinery, importlib.metadata, importlib.util
import json, re, sys

config = json.loads(sys.argv[1])
blocked = frozenset(config["modules"])
hidden = frozenset(config["distributions"])
training = blocked | frozenset(config["sentinels"])


def canonical(name):
    return re.sub(r"[-_.]+", "-", name or "").lower()


def loaded():
    return sorted(name for name in sys.modules if name.partition(".")[0] in training)


if loaded():
    print("G5-LOADED-AT-STARTUP", loaded())
    sys.exit(3)


class TrainingTierAbsent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, TrainingTierAbsent())
installed = importlib.machinery.PathFinder.find_distributions


def find_distributions(*args, **kwargs):
    for dist in installed(*args, **kwargs):
        if canonical(dist.metadata["Name"]) not in hidden:
            yield dist


importlib.machinery.PathFinder.find_distributions = staticmethod(find_distributions)

for name in sorted(training):
    try:
        importlib.import_module(name)
    except ImportError:
        continue
    print("G5-STILL-IMPORTABLE", name)
    sys.exit(4)
for dist in sorted(hidden):
    try:
        importlib.metadata.distribution(dist)
    except importlib.metadata.PackageNotFoundError:
        continue
    print("G5-STILL-INSTALLED", dist)
    sys.exit(5)
for name in config["application_tier"]:
    # Located through the blocked import system, but not imported: importing a pytest
    # plugin here would stop pytest rewriting its assertions.
    if importlib.util.find_spec(name) is None:
        print("G5-APPLICATION-TIER-MISSING", name)
        sys.exit(7)
print("G5-TRAINING-TIER-ABSENT", json.dumps({"blocked": sorted(blocked), "hidden": sorted(hidden)}))

import pytest

code = int(pytest.main(config["pytest_args"]))
if loaded():
    print("G5-LOADED-DURING-RUN", loaded())
    sys.exit(6)
print("G5-NO-TRAINING-MODULE-LOADED")
sys.exit(code)
"""
"""Runs in the child interpreter. It makes every training-only package unimportable and
hides its installed metadata, proves both, proves the application tier still imports,
then runs pytest in the same process -- so every import the suite makes goes through the
block -- and finally proves no training module was loaded."""


def canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(tier: str) -> set[str]:
    """Canonical names of the distributions a tier installs, following its -r includes."""
    names: set[str] = set()
    for raw in (ROOT / "requirements" / f"{tier}.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        include = re.fullmatch(r"-r\s+(\S+?)\.txt", line)
        if include:
            names |= requirement_names(include.group(1))
        else:
            names.add(canonical_distribution(re.split(r"[\s=<>!~;\[]", line, maxsplit=1)[0]))
    return names


def training_only_distributions() -> set[str]:
    """What train.txt (and the ml.txt it includes) installs that base.txt and dev.txt do not."""
    return requirement_names("train") - requirement_names("dev")


def import_names(distributions: set[str]) -> set[str]:
    """Top-level import names that the installed members of `distributions` provide.

    Read from each distribution's own file list. A distribution that is not installed
    provides nothing to block: it is already absent.
    """
    names = set()
    for dist in metadata.distributions():
        if canonical_distribution(dist.metadata["Name"] or "") not in distributions:
            continue
        for file in dist.files or ():
            top = file.parts[0]
            if top in ("..", "__pycache__") or top.endswith((".dist-info", ".data")):
                continue
            candidate = top.split(".", 1)[0] if len(file.parts) == 1 else top
            if candidate.isidentifier():
                names.add(candidate)
    return names


def application_job_environment() -> dict[str, str]:
    """The parent's environment, minus anything that would steer the child's pytest.

    DJANGO_SETTINGS_MODULE is dropped so conftest.py supplies it, as it does in CI's
    application job. The database is an in-memory SQLite one, so the child can never
    touch, or collide with, a database the parent run is using.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_CORE_")) and key != "DJANGO_SETTINGS_MODULE"
    }
    environment["DATABASE_URL"] = "sqlite://:memory:"
    return environment


def run_without_training_tier(
    pytest_args: Sequence[str], cwd: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run pytest in a child interpreter in which the training tier does not exist.

    ``-I`` isolates the child from PYTHONPATH and the user site, so the block, not the
    parent's environment, decides what imports; ``-B`` writes no bytecode.
    """
    training = training_only_distributions()
    config = {
        "modules": sorted(import_names(training)),
        "distributions": sorted(training),
        "sentinels": list(TRAINING_SENTINELS),
        "application_tier": list(APPLICATION_TIER),
        "pytest_args": list(pytest_args),
    }
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", G5_BOOTSTRAP, json.dumps(config)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=G5_TIMEOUT_SECONDS,
        check=False,
    )


def pytest_counts(output: str) -> dict[str, int]:
    """The outcome counts on pytest's final summary line."""
    summaries = re.findall(r"^=*\s*(\d+ \w+(?:, \d+ \w+)*) in [\d.]+s", output, re.MULTILINE)
    assert summaries, output[-2000:]
    return {word: int(count) for count, word in re.findall(r"(\d+) (\w+)", summaries[-1])}


def test_g5_app_suite_passes_without_the_training_tier():
    """Plan §U Task 22: the application suite passes without train.txt installed.

    Runs CI's application-job selection, ``-m "not ml"``, in a child interpreter in
    which every package only the training tier installs is unimportable and its metadata
    is hidden. Exactly this module's G5 tests are deselected there (G5_DESELECT), so the
    child does not recurse and nothing else escapes the proof.
    """
    result = run_without_training_tier(
        ["-m", "not ml", *G5_DESELECT, "-p", "no:cacheprovider", "-q"],
        ROOT,
        application_job_environment(),
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "G5-TRAINING-TIER-ABSENT" in output, output[-4000:]
    assert "G5-NO-TRAINING-MODULE-LOADED" in output, output[-4000:]
    counts = pytest_counts(result.stdout)
    assert counts.get("passed", 0) > 0, counts
    assert not {"failed", "error", "errors"} & counts.keys(), counts


def test_g5_child_deselects_only_this_modules_g5_tests(tmp_path):
    """G5_DESELECT removes this module's G5 tests and nothing else: not a test whose name
    merely contains "g5_", not a module whose name does, not a G5-named test elsewhere."""
    write_tree(
        tmp_path,
        {
            "pytest.ini": "[pytest]\n",
            G5_MODULE: (
                "def test_g5_app_suite():\n    pass\n\n\n"
                "def test_g5_harness():\n    pass\n\n\n"
                "def test_log5_rotation():\n    pass\n"
            ),
            "tests/test_config5_loader.py": "def test_loads():\n    pass\n",
            "tests/test_elsewhere.py": "def test_g5_named_elsewhere():\n    pass\n",
        },
    )
    collected = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + ["-p", "no:django", *G5_DESELECT],
        cwd=tmp_path,
        env=application_job_environment(),
        capture_output=True,
        text=True,
        timeout=G5_TIMEOUT_SECONDS,
        check=False,
    )
    selected = {line.strip() for line in collected.stdout.splitlines() if "::" in line}
    assert selected == {
        f"{G5_MODULE}::test_log5_rotation",
        "tests/test_config5_loader.py::test_loads",
        "tests/test_elsewhere.py::test_g5_named_elsewhere",
    }, collected.stdout + collected.stderr
    assert "3/5 tests collected (2 deselected)" in collected.stdout, collected.stdout


def test_g5_training_tier_is_derived_from_the_requirements():
    training = training_only_distributions()
    assert {"numpy", "scikit-learn", "onnxruntime", "tokenizers", "pandas", "pyarrow"} <= training
    assert not training & requirement_names("dev")
    assert {"django", "djangorestframework", "pytest", "pytest-django"} <= requirement_names("dev")


def test_g5_harness_fails_an_app_test_that_needs_the_training_tier(tmp_path):
    """The block is real: an application test importing numpy fails under it."""
    write_tree(
        tmp_path,
        {
            "pytest.ini": "[pytest]\n",
            "test_needs_numpy.py": "import numpy\n\n\ndef test_uses_numpy():\n    assert numpy\n",
        },
    )
    result = run_without_training_tier(
        ["-p", "no:cacheprovider", "-p", "no:django", "-q", "test_needs_numpy.py"],
        tmp_path,
        application_job_environment(),
    )
    output = result.stdout + result.stderr
    assert "G5-TRAINING-TIER-ABSENT" in output, output[-4000:]
    assert result.returncode != 0, output[-4000:]
    assert "No module named 'numpy'" in output, output[-4000:]


def test_g5_harness_passes_an_app_test_that_needs_only_the_application_tier(tmp_path):
    write_tree(
        tmp_path,
        {
            "pytest.ini": "[pytest]\n",
            "test_plain.py": (
                "import json\n\n\ndef test_plain():\n    assert json.dumps(1) == '1'\n"
            ),
        },
    )
    result = run_without_training_tier(
        ["-p", "no:cacheprovider", "-p", "no:django", "-q", "test_plain.py"],
        tmp_path,
        application_job_environment(),
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert pytest_counts(result.stdout) == {"passed": 1}


# --- shared by G6, G7 and G8: interfaces compared by resolved type ------------------------


def canonical(annotation: object, home: str) -> str:
    """A resolved type, spelled so that one defined in the module under test reads by name.

    Resolution happens before this is called (typing.get_type_hints), so a string
    annotation and the type it names spell the same.
    """
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin in (types.UnionType, typing.Union):
        return " | ".join(canonical(argument, home) for argument in arguments)
    if origin is not None:
        return f"{canonical(origin, home)}[{', '.join(canonical(a, home) for a in arguments)}]"
    if annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        if annotation.__module__ in ("builtins", home):
            return annotation.__qualname__
        return f"{annotation.__module__}.{annotation.__qualname__}"
    return repr(annotation)


def constructor_shape(cls: type) -> tuple[tuple[str, str], ...] | None:
    """How the class is called: each constructor parameter's name and kind.

    Fields alone do not say this -- ``kw_only=True`` leaves every field as it was and
    breaks every positional construction. Annotations and defaults are left to the
    field comparison, so nothing here depends on how they are spelled.
    """
    try:
        parameters = inspect.signature(cls).parameters.values()
    except (TypeError, ValueError):
        return None
    return tuple((parameter.name, parameter.kind.name) for parameter in parameters)


def dataclass_contract(cls: type) -> dict[str, object]:
    """A dataclass's frozenness, its fields in order as (name, resolved type, required),
    and the shape of its constructor."""
    hints = typing.get_type_hints(cls)
    parameters = getattr(cls, "__dataclass_params__", None)
    return {
        "dataclass": dataclasses.is_dataclass(cls),
        "frozen": bool(parameters and parameters.frozen),
        "fields": tuple(
            (
                field.name,
                canonical(hints[field.name], cls.__module__),
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING,
            )
            for field in dataclasses.fields(cls)
        ),
        "constructor": constructor_shape(cls),
    }


def protocol_contract(protocol: type) -> dict[str, object]:
    """A Protocol's members, each with its parameters and resolved type hints."""
    methods = {}
    for name in sorted(typing.get_protocol_members(protocol)):
        member = getattr(protocol, name)
        parameters = inspect.signature(member).parameters.values()
        hints = typing.get_type_hints(member)
        methods[name] = (
            tuple((parameter.name, parameter.kind.name) for parameter in parameters),
            tuple(
                sorted((key, canonical(value, protocol.__module__)) for key, value in hints.items())
            ),
        )
    return {
        "protocol": bool(getattr(protocol, "_is_protocol", False)),
        "runtime_checkable": bool(getattr(protocol, "_is_runtime_protocol", False)),
        "methods": methods,
    }


def contract(obj: type) -> dict[str, object]:
    if dataclasses.is_dataclass(obj):
        return dataclass_contract(obj)
    if getattr(obj, "_is_protocol", False):
        return protocol_contract(obj)
    raise TypeError(f"{obj!r} is neither a dataclass nor a Protocol")


def dataclass_spec(*fields: tuple[str, str]) -> dict[str, object]:
    """The contract of a frozen dataclass whose fields are all required and positional."""
    return {
        "dataclass": True,
        "frozen": True,
        "fields": tuple((name, annotation, True) for name, annotation in fields),
        "constructor": tuple((name, "POSITIONAL_OR_KEYWORD") for name, _ in fields),
    }


def protocol_spec(**methods: tuple[tuple[str, ...], dict[str, str]]) -> dict[str, object]:
    return {
        "protocol": True,
        "runtime_checkable": False,
        "methods": {
            name: (
                tuple((parameter, "POSITIONAL_OR_KEYWORD") for parameter in parameters),
                tuple(sorted(hints.items())),
            )
            for name, (parameters, hints) in methods.items()
        },
    }


# --- G6. corpus records never enter Complaint -------------------------------------------

COMPLAINT_MODEL = "Complaint"
"""The operational model. Named in any form -- imported, referenced, reached as an
attribute -- it is operational identity inside Phase 2 code."""

COMPLAINT_KEYS = frozenset({"pk", "complaint_id"})
"""Names a Complaint primary key is read or passed under. Only reading one off an object
(``row.pk``, ``match.complaint_id``) or passing one by keyword (``Match(complaint_id=...)``)
is operational identity. A local, a parameter, an import or a dictionary key of the same
name is not: "complaint_id" is also the CFPB API's own field, and binding that field under
its own name is ordinary CFPB normalisation."""

SERVING_MODULE = "ml.base"
"""Addendum §2.2: ml.base.Match is typed ``complaint_id: int``, and the benchmark over
corpus records "must not reuse that type". Reaching it from Phase 2 would type an
evaluation output to Complaint.pk."""


def dotted_reference(node: ast.expr) -> str | None:
    """``a.b.c`` for a chain of names, or None for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = dotted_reference(node.value)
        return f"{owner}.{node.attr}" if owner else None
    return None


def serving_module_names(tree: ast.Module, package: str) -> set[str]:
    """Local names a module binds to ml.base (``from ml import base``, ``import ml.base as
    serving``, or a relative ``from .. import base`` inside ml/)."""
    names = {SERVING_MODULE}
    parent, _, leaf = SERVING_MODULE.rpartition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.asname for a in node.names if a.name == SERVING_MODULE and a.asname)
        elif isinstance(node, ast.ImportFrom) and resolve_import_from(node, package) == parent:
            names.update(a.asname or a.name for a in node.names if a.name == leaf)
    return names


def identity_usage(node: ast.AST, package: str, serving: set[str]) -> str | None:
    """The operational Complaint identity `node` uses, if any (addendum §2.2)."""
    if isinstance(node, ast.Name) and node.id == COMPLAINT_MODEL:
        return COMPLAINT_MODEL
    if isinstance(node, ast.alias) and COMPLAINT_MODEL in (node.name.split(".")[-1], node.asname):
        return COMPLAINT_MODEL
    if isinstance(node, ast.Attribute):
        if node.attr == COMPLAINT_MODEL:
            return COMPLAINT_MODEL
        if node.attr in COMPLAINT_KEYS:
            return f".{node.attr}"
        if node.attr == "Match" and dotted_reference(node.value) in serving:
            return f"{SERVING_MODULE}.Match"
    if isinstance(node, ast.keyword) and node.arg in COMPLAINT_KEYS:
        return f"{node.arg}="
    if isinstance(node, ast.ImportFrom) and resolve_import_from(node, package) == SERVING_MODULE:
        for alias in node.names:
            if alias.name in ("Match", "*"):
                return f"{SERVING_MODULE}.{alias.name}"
    return None


def identity_references(root: Path, packages: Sequence[str]) -> list[str]:
    """Every operational Complaint identity usage in `packages`: the Complaint model, a
    Complaint key read off an object or passed by keyword, or the serving Match typed to
    Complaint.pk. Code is read, never strings, and never a bare name that happens to match:
    see COMPLAINT_KEYS."""
    found = []
    for package in packages:
        for path in sorted((root / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            owner = package_of(path, root)
            serving = serving_module_names(tree, owner)
            for node in ast.walk(tree):
                usage = identity_usage(node, owner, serving)
                if usage:
                    lineno = getattr(node, "lineno", 0)
                    found.append(f"{path.relative_to(root).as_posix()}:{lineno} {usage}")
    return found


RECORD_IDENTITY = (("source", "str", True), ("external_id", "str", True))
"""Addendum §2.2: a corpus record is identified by (source, external_id), two strings."""


def test_g6_corpus_records_never_enter_complaint():
    """Plan coverage matrix, addendum §2 and §2.2: no corpus record becomes a Complaint row,
    and none is given a synthetic Complaint primary key."""
    # Identity is (source, external_id), and RecordRef carries nothing else.
    assert dataclass_contract(RecordRef) == dataclass_spec(
        ("source", "str"), ("external_id", "str")
    )
    assert dataclass_contract(CorpusRecord)["fields"][:2] == RECORD_IDENTITY
    for outcome in (CFPBOutcome, NYC311Outcome):
        assert dataclass_contract(outcome)["fields"][0] == ("external_id", "str", True)
    for cls in (CorpusRecord, CFPBOutcome, NYC311Outcome):
        names = {field.name for field in dataclasses.fields(cls)}
        assert not names & {"id", "pk", "complaint_id", "complaint"}, cls

    record = CorpusRecord(
        source=next(iter(PACKS)),
        external_id="42",
        text="t",
        label="l",
        submitted_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    ref = make_ref(record)
    assert ref == RecordRef(source=record.source, external_id="42")
    assert parse_ref(str(ref)) == ref

    # Complaint has no field that could hold a corpus identity, and its primary key is
    # generated by the database rather than supplied by a caller.
    complaint_fields = {field.name for field in Complaint._meta.get_fields()}
    assert not complaint_fields & {"source", "external_id", "record_ref", "corpus_id"}
    assert isinstance(Complaint._meta.pk, models.AutoField)

    # No Phase 2 module uses operational Complaint identity -- the model, a Complaint key
    # read or passed, or the serving Match typed to Complaint.pk -- and none can import
    # the app that defines Complaint (G3's closure, read for this one app).
    assert not identity_references(ROOT, PHASE_2_PACKAGES), identity_references(
        ROOT, PHASE_2_PACKAGES
    )
    violations, reached = import_closure(ROOT, PHASE_2_PACKAGES)
    assert not [violation for violation in violations if "complaints" in violation]
    assert not [name for name in reached if name.split(".")[0] == "complaints"]


@pytest.mark.parametrize(
    ("files", "scanned", "expected"),
    [
        ({"phase2/a.py": "from somewhere import Complaint\n"}, "phase2", "phase2/a.py:1 Complaint"),
        (
            {"phase2/a.py": "from somewhere import Complaint as Renamed\n"},
            "phase2",
            "phase2/a.py:1 Complaint",
        ),
        (
            {"phase2/a.py": "def model(app):\n    return app.Complaint\n"},
            "phase2",
            "phase2/a.py:2 Complaint",
        ),
        (
            {"phase2/a.py": "def key(match):\n    return match.complaint_id\n"},
            "phase2",
            "phase2/a.py:2 .complaint_id",
        ),
        ({"phase2/a.py": "def key(row):\n    return row.pk\n"}, "phase2", "phase2/a.py:2 .pk"),
        (
            {"phase2/a.py": "def build(**kwargs):\n    return kwargs\n\n\nbuild(complaint_id=3)\n"},
            "phase2",
            "phase2/a.py:5 complaint_id=",
        ),
        (
            {"phase2/a.py": "def get(**kwargs):\n    return kwargs\n\n\nget(pk=3)\n"},
            "phase2",
            "phase2/a.py:5 pk=",
        ),
        ({"phase2/a.py": "from ml.base import Match\n"}, "phase2", "phase2/a.py:1 ml.base.Match"),
        ({"phase2/a.py": "from ml.base import *\n"}, "phase2", "phase2/a.py:1 ml.base.*"),
        (
            {"phase2/a.py": "from ml import base\n\nbase.Match(1, 0.5, 'v')\n"},
            "phase2",
            "phase2/a.py:3 ml.base.Match",
        ),
        (
            {"phase2/a.py": "import ml.base as serving\n\nserving.Match(1, 0.5, 'v')\n"},
            "phase2",
            "phase2/a.py:3 ml.base.Match",
        ),
        (
            {"phase2/a.py": "import ml.base\n\nml.base.Match(1, 0.5, 'v')\n"},
            "phase2",
            "phase2/a.py:3 ml.base.Match",
        ),
        (
            {
                "ml/__init__.py": "",
                "ml/training/__init__.py": "",
                "ml/training/a.py": "from ..base import Match\n",
            },
            "ml/training",
            "ml/training/a.py:1 ml.base.Match",
        ),
    ],
    ids=[
        "complaint-model",
        "complaint-model-renamed",
        "complaint-model-attribute",
        "key-read-complaint-id",
        "key-read-pk",
        "key-passed-complaint-id",
        "key-passed-pk",
        "serving-match-imported",
        "serving-module-star-imported",
        "serving-match-through-module",
        "serving-match-through-alias",
        "serving-match-through-dotted-module",
        "serving-match-relative",
    ],
)
def test_g6_checker_catches_operational_complaint_identity(tmp_path, files, scanned, expected):
    write_tree(tmp_path, {"phase2/__init__.py": "", **files})
    references = identity_references(tmp_path, [scanned])
    assert expected in references, references


def test_g6_checker_passes_cfpb_field_handling_and_unrelated_names(tmp_path):
    """The CFPB API's own field is called "complaint_id". Binding it as a local, a parameter,
    an import or a dictionary key is CFPB normalisation, not Complaint identity; neither is
    another module's Match (re.Match) nor another name from ml.base."""
    write_tree(
        tmp_path,
        {
            "phase2/__init__.py": "",
            "phase2/fields.py": "def complaint_id(row):\n    return row['complaint_id']\n",
            "phase2/cfpb.py": (
                "import re\n\n"
                "from ml.base import TextEmbedder\n"
                "from phase2.fields import complaint_id\n"
                "from phase2.fields import complaint_id as read_complaint_id\n\n"
                "COMPLAINT_ID = 'complaint_id'\n\n\n"
                "def external_id(row, complaint_id=None):\n"
                "    complaint_id = row.get('complaint_id', complaint_id)\n"
                "    record = {'complaint_id': complaint_id}\n"
                "    return str(record['complaint_id'])\n\n\n"
                "def first(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:\n"
                "    return pattern.match(text)\n"
            ),
        },
    )
    assert not identity_references(tmp_path, ["phase2"]), identity_references(tmp_path, ["phase2"])


def test_g6_checker_catches_an_integer_record_identity(monkeypatch):
    module = load_source_as_module(
        "g6_variant",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\nclass RecordRef:\n    source: str\n    external_id: int\n",
        monkeypatch,
    )
    assert dataclass_contract(module.RecordRef)["fields"] != RECORD_IDENTITY


# --- G7 and G8. the Phase 1 serving contract ----------------------------------------------


RISK_FEATURES = (
    ("sla_hours", "int"),
    ("category_mean_resolution_hours", "float"),
    ("category_breach_rate", "float"),
    ("priority_rank", "float"),
    ("age_hours", "float"),
    ("submitted_hour", "int"),
    ("submitted_weekday", "int"),
    ("text_length", "int"),
    ("queue_depth", "int"),
    ("assignee_open_count", "int"),
)

PHASE_1_SERVING: dict[str, dict[str, object]] = {
    "TriagePrediction": dataclass_spec(
        ("category_slug", "str | None"), ("confidence", "float"), ("model_version", "str")
    ),
    "Match": dataclass_spec(
        ("complaint_id", "int"), ("similarity", "float"), ("model_version", "str")
    ),
    "RiskScore": dataclass_spec(("score", "float"), ("band", "str"), ("model_version", "str")),
    "RiskFeatures": dataclass_spec(*RISK_FEATURES),
    "TriageModel": protocol_spec(
        predict=(("self", "text"), {"text": "str", "return": "TriagePrediction"})
    ),
    "DedupIndex": protocol_spec(
        query=(("self", "text", "k"), {"text": "str", "k": "int", "return": "list[Match]"})
    ),
    "RiskModel": protocol_spec(
        predict=(("self", "features"), {"features": "RiskFeatures", "return": "RiskScore"})
    ),
}
"""ml/base.py's serving contract at PHASE_1_BOUNDARY, by resolved type.

Addendum §8: TriageModel, DedupIndex and RiskModel keep their signatures in Phase 2, and
RiskFeatures' field requirements change in Phase 3, not Phase 2. The result types are
included because a signature naming TriagePrediction means nothing if TriagePrediction
itself changes.

D38 added ``from __future__ import annotations`` to ml/base.py so numpy could stay
TYPE_CHECKING-only. Every annotation there is therefore a string at run time -- Match's
``complaint_id`` is ``'int'``, not ``int`` -- so a raw ``__annotations__`` comparison
would report a change that is not one. Types are resolved with typing.get_type_hints
before they are compared."""

PROTOCOL_HEADER = (
    "from __future__ import annotations\n\n"
    "from dataclasses import dataclass\n"
    "from typing import Protocol, runtime_checkable\n\n\n"
    "@dataclass(frozen=True)\nclass Match:\n"
    "    complaint_id: int\n    similarity: float\n    model_version: str\n\n\n"
)
"""Planted variants start from Phase 1's Match, under D38's string annotations."""


def test_g7_match_and_dedup_index_are_unchanged():
    """Plan Task 18 and the coverage matrix: Match and DedupIndex are the Phase 1 ones."""
    assert contract(base.Match) == PHASE_1_SERVING["Match"]
    assert contract(base.DedupIndex) == PHASE_1_SERVING["DedupIndex"]
    # The protocol returns this Match, not a lookalike of the same name.
    assert typing.get_type_hints(base.DedupIndex.query)["return"] == list[base.Match]


@pytest.mark.parametrize(
    "body",
    [
        "complaint_id: str\n    similarity: float\n    model_version: str\n",
        "complaint_id: int\n    similarity: float\n    model_version: str\n    source: str = ''\n",
        "similarity: float\n    complaint_id: int\n    model_version: str\n",
        "record_id: int\n    similarity: float\n    model_version: str\n",
        "complaint_id: int\n    similarity: float\n    model_version: str = 'null'\n",
        "complaint_id: int\n    _: KW_ONLY\n    similarity: float\n    model_version: str\n",
    ],
    ids=["retyped", "extra-field", "reordered", "renamed", "defaulted", "field-keyword-only"],
)
def test_g7_checker_catches_a_changed_match(monkeypatch, body):
    source = (
        "from __future__ import annotations\n\nfrom dataclasses import KW_ONLY, dataclass\n\n\n"
        f"@dataclass(frozen=True)\nclass Match:\n    {body}"
    )
    module = load_source_as_module("g7_variant", source, monkeypatch)
    assert contract(module.Match) != PHASE_1_SERVING["Match"]


def test_g7_checker_catches_an_unfrozen_match(monkeypatch):
    source = PROTOCOL_HEADER.replace("@dataclass(frozen=True)", "@dataclass")
    module = load_source_as_module("g7_variant", source, monkeypatch)
    assert contract(module.Match) != PHASE_1_SERVING["Match"]


def test_g7_checker_catches_a_keyword_only_match(monkeypatch):
    """``kw_only=True`` leaves every field as Phase 1 had it and still breaks every
    positional construction, so only the constructor's shape can see the change."""
    source = PROTOCOL_HEADER.replace(
        "@dataclass(frozen=True)", "@dataclass(frozen=True, kw_only=True)"
    )
    module = load_source_as_module("g7_variant", source, monkeypatch)
    with pytest.raises(TypeError):
        module.Match(1, 0.5, "v")
    actual, expected = contract(module.Match), PHASE_1_SERVING["Match"]
    assert (actual["fields"], actual["frozen"]) == (expected["fields"], expected["frozen"])
    assert actual["constructor"] != expected["constructor"]


@pytest.mark.parametrize(
    "protocol",
    [
        "class DedupIndex(Protocol):\n"
        "    def query(self, text: str, k: int, *, domain: str) -> list[Match]: ...\n",
        "class DedupIndex(Protocol):\n    def query(self, text: str, k: int) -> list[int]: ...\n",
        "class DedupIndex(Protocol):\n"
        "    def query(self, text: str, k: float) -> list[Match]: ...\n",
        "class DedupIndex(Protocol):\n"
        "    def search(self, text: str, k: int) -> list[Match]: ...\n",
        "class DedupIndex(Protocol):\n"
        "    def query(self, text: str, k: int) -> list[Match]: ...\n"
        "    def add(self, text: str) -> None: ...\n",
        "@runtime_checkable\nclass DedupIndex(Protocol):\n"
        "    def query(self, text: str, k: int) -> list[Match]: ...\n",
    ],
    ids=["extra-parameter", "retyped-return", "retyped-k", "renamed", "extra-method", "runtime"],
)
def test_g7_checker_catches_a_changed_dedup_index(monkeypatch, protocol):
    module = load_source_as_module("g7_variant", PROTOCOL_HEADER + protocol, monkeypatch)
    assert contract(module.DedupIndex) != PHASE_1_SERVING["DedupIndex"]


def test_g8_phase_3_serving_interface_is_not_mutated():
    """Plan coverage matrix and addendum §8: the Phase 3 serving interface is Phase 1's."""
    for name, expected in PHASE_1_SERVING.items():
        assert contract(getattr(base, name)) == expected, name
    assert typing.get_type_hints(base.TriageModel.predict)["return"] is base.TriagePrediction
    assert typing.get_type_hints(base.RiskModel.predict) == {
        "features": base.RiskFeatures,
        "return": base.RiskScore,
    }


def test_g8_checker_compares_resolved_types_not_raw_annotations(monkeypatch):
    """Phase 1's own definitions under D38's string annotations compare equal: the
    comparison is of what the annotations mean, not how they are stored."""
    protocol = (
        "class DedupIndex(Protocol):\n    def query(self, text: str, k: int) -> list[Match]: ...\n"
    )
    module = load_source_as_module("g8_resolved", PROTOCOL_HEADER + protocol, monkeypatch)
    assert module.Match.__annotations__["complaint_id"] == "int", "raw annotations are strings"
    assert contract(module.Match) == PHASE_1_SERVING["Match"]
    assert contract(module.DedupIndex) == PHASE_1_SERVING["DedupIndex"]


def risk_features_source(
    fields: Iterable[tuple[str, str]], decorator: str = "@dataclass(frozen=True)"
) -> str:
    body = "".join(f"    {name}: {annotation}\n" for name, annotation in fields)
    return f"{decorator}\nclass RiskFeatures:\n{body}"


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        (
            "RiskFeatures",
            risk_features_source(
                [
                    *RISK_FEATURES[:-2],
                    ("queue_depth", "int | None = None"),
                    ("assignee_open_count", "int | None = None"),
                ]
            ),
        ),
        ("RiskFeatures", risk_features_source(RISK_FEATURES[:-1])),
        (
            "RiskFeatures",
            risk_features_source(RISK_FEATURES, "@dataclass(frozen=True, kw_only=True)"),
        ),
        (
            "TriageModel",
            "class TriagePrediction:\n    pass\n\n\nclass TriageModel(Protocol):\n"
            "    def predict(self, text: str, domain: str) -> TriagePrediction: ...\n",
        ),
        (
            "RiskModel",
            "class RiskFeatures:\n    pass\n\n\nclass RiskModel(Protocol):\n"
            "    def predict(self, features: RiskFeatures) -> float: ...\n",
        ),
        (
            "TriagePrediction",
            "@dataclass(frozen=True)\nclass TriagePrediction:\n"
            "    category_slug: str\n    confidence: float\n    model_version: str\n",
        ),
    ],
    ids=[
        "risk-features-made-optional",
        "risk-features-field-removed",
        "risk-features-made-keyword-only",
        "triage-model-parameter-added",
        "risk-model-return-retyped",
        "triage-prediction-slug-made-required",
    ],
)
def test_g8_checker_catches_a_changed_serving_interface(monkeypatch, name, variant):
    module = load_source_as_module("g8_variant", PROTOCOL_HEADER + variant, monkeypatch)
    assert contract(getattr(module, name)) != PHASE_1_SERVING[name]


def test_g8_pins_are_the_phase_1_boundary(monkeypatch):
    """PHASE_1_SERVING is what ml/base.py defined at PHASE_1_BOUNDARY, read from git."""
    require_phase_1_boundary()
    shown = git("show", f"{PHASE_1_BOUNDARY}:ml/base.py")
    assert shown.returncode == 0, shown.stderr
    module = load_source_as_module("phase_1_ml_base", shown.stdout, monkeypatch)
    for name, expected in PHASE_1_SERVING.items():
        assert contract(getattr(module, name)) == expected, name


# --- G9. git tracks no corpus, model binary or data/ content ------------------------------

TRACKED_ARTIFACT_SUFFIXES = (".parquet", ".joblib", ".onnx")
"""Plan finding I3 and §Y: no Parquet file and no model binary is tracked."""


def tracked_paths(cwd: Path, env: dict[str, str] | None = None) -> list[str]:
    """What git's index tracks, never what happens to be on disk."""
    listed = git("ls-files", "-z", cwd=cwd, env=env)
    assert listed.returncode == 0, listed.stderr
    return [path for path in listed.stdout.split("\0") if path]


def tracked_artifact_violations(paths: Iterable[str]) -> list[str]:
    return [
        path
        for path in paths
        if path.lower().endswith(TRACKED_ARTIFACT_SUFFIXES) or path.split("/", 1)[0] == "data"
    ]


def test_g9_git_tracks_no_corpus_or_model_binary():
    """Plan finding I3 and §Y: git ls-files shows no Parquet, .joblib, .onnx or data/ content.

    This read-only query deliberately inherits the caller's environment: run from a
    pre-commit hook, it then checks the index that is about to be committed.
    """
    paths = tracked_paths(ROOT)
    assert {"manage.py", "ml/registry.py", "pyproject.toml"} <= set(paths), "not the real index"
    assert not tracked_artifact_violations(paths), tracked_artifact_violations(paths)


G9_PLANTED = (
    "data/raw/cfpb/page-0001.json",
    "data/corpus/cfpb/v1/year=2024/part-0000.parquet",
    "ml/artifacts/cfpb/triage/v1/model.joblib",
    "ml/artifacts/embedders/minilm/v1/MODEL.ONNX",
)
G9_ALLOWED = ("ml/artifacts/cfpb/triage/v1/metadata.json", "src/module.py")


def plant_tracked_artifacts(repo: Path) -> list[str]:
    """Create a repository at `repo`, track G9_PLANTED and G9_ALLOWED in it, leave one
    Parquet file on disk untracked, and return what the G9 checker reports.

    Every git command runs with without_git_environment(): an inherited GIT_INDEX_FILE,
    GIT_DIR or GIT_WORK_TREE would otherwise send the planted paths into another
    repository's index -- a developer's, if the suite runs from a pre-commit hook.
    """
    clean = without_git_environment()
    repo.mkdir(parents=True, exist_ok=True)
    assert git("init", "-q", cwd=repo, env=clean).returncode == 0
    write_tree(repo, dict.fromkeys([*G9_PLANTED, *G9_ALLOWED], "x\n"))
    added = git("add", "--", *G9_PLANTED, *G9_ALLOWED, cwd=repo, env=clean)
    assert added.returncode == 0, added.stderr
    # On disk but never added: the index, not the filesystem, is what counts.
    (repo / "untracked.parquet").write_text("x\n", encoding="utf-8")
    return tracked_artifact_violations(tracked_paths(repo, env=clean))


def test_g9_checker_reads_the_git_index_not_the_disk(tmp_path):
    assert sorted(plant_tracked_artifacts(tmp_path / "repo")) == sorted(G9_PLANTED)


def test_g9_checker_cannot_write_to_an_inherited_index(tmp_path, monkeypatch):
    """Repository control inherited from the caller never reaches the temporary repository.

    A bystander repository stands in for the developer's: GIT_DIR, GIT_WORK_TREE and
    GIT_INDEX_FILE all point at it, as a hook or tool may export them. Planting must still
    land in the temporary repository alone, and the bystander's index must not change by
    one byte.
    """
    bystander = tmp_path / "bystander"
    bystander.mkdir()
    clean = without_git_environment()
    assert git("init", "-q", cwd=bystander, env=clean).returncode == 0
    write_tree(bystander, {"kept.txt": "kept\n"})
    assert git("add", "kept.txt", cwd=bystander, env=clean).returncode == 0
    index = bystander / ".git" / "index"
    before = index.read_bytes()

    monkeypatch.setenv("GIT_DIR", str(bystander / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(bystander))
    monkeypatch.setenv("GIT_INDEX_FILE", str(index))

    assert sorted(plant_tracked_artifacts(tmp_path / "repo")) == sorted(G9_PLANTED)
    assert index.read_bytes() == before
    assert tracked_paths(bystander, env=without_git_environment()) == ["kept.txt"]
