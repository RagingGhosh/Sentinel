"""Serving code must never import training code.

`ml/base.py`, `ml/null.py` and `ml/registry.py` are imported by Django at
startup. If any of them reached into `ml.training`, `ml.embedders` or `ingest`,
the web process would pull in scikit-learn, onnxruntime and pandas — defeating
the dependency split and the Phase 3 memory budget.

The check is static. It parses the files rather than importing them, so it
gives the same verdict whether or not the training packages are installed, and
it cannot be satisfied by an environment that happens to have them.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SERVING_MODULES = ("ml/base.py", "ml/null.py", "ml/registry.py")
FORBIDDEN_FOR_SERVING = ("ml.training", "ml.embedders", "ingest")

TRAINING_PACKAGES = ("ingest", "ml/training")

DYNAMIC_IMPORTERS = {"import_module", "__import__"}


def _module_name(path: Path) -> str:
    """Dotted name of a file, relative to the repository root."""
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path) -> set[str]:
    """Every module name imported by a file, statically and dynamically.

    Relative imports are resolved to absolute dotted names. Dynamic imports are
    caught only when the target is a string literal — `import_module(name)` with
    a computed name is beyond static analysis, and this test does not pretend
    otherwise.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dotted = _module_name(path)
    package = dotted if path.name == "__init__.py" else dotted.rsplit(".", 1)[0]
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                trimmed = base[: len(base) - node.level + 1] if node.level > 1 else base
                found.add(".".join([*trimmed, node.module] if node.module else trimmed))
            elif node.module:
                found.add(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in DYNAMIC_IMPORTERS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    found.add(first.value)

    return found


def _package_files(package: str) -> list[Path]:
    return sorted((ROOT / package).rglob("*.py"))


def test_serving_modules_exist():
    """Guard the guard: a typo'd path would make every assertion below vacuous."""
    for rel in SERVING_MODULES:
        assert (ROOT / rel).is_file(), f"{rel} not found — the boundary test would pass vacuously"


def test_serving_does_not_import_training():
    violations = []
    for rel in SERVING_MODULES:
        for imported in _imports(ROOT / rel):
            for banned in FORBIDDEN_FOR_SERVING:
                if imported == banned or imported.startswith(f"{banned}."):
                    violations.append(f"{rel} imports {imported}")
    assert not violations, (
        "serving code must not import training code — this is what keeps "
        f"scikit-learn and onnxruntime out of the web process: {violations}"
    )


def test_training_packages_exist():
    for package in TRAINING_PACKAGES:
        path = ROOT / package
        assert path.is_dir(), f"{package}/ not found"
        assert (path / "__init__.py").is_file(), f"{package}/__init__.py missing"


def test_training_packages_do_not_import_django():
    """Training runs as `python -m`, never as a management command."""
    violations = []
    for package in TRAINING_PACKAGES:
        for path in _package_files(package):
            for imported in _imports(path):
                if imported == "django" or imported.startswith("django."):
                    violations.append(f"{path.relative_to(ROOT).as_posix()} imports {imported}")
    assert not violations, f"ingest/ and ml/training/ must stay Django-independent: {violations}"


def test_relative_imports_are_resolved():
    """The resolver must not silently ignore `from . import x` forms."""
    source = "from . import sibling\nfrom .deeper import thing\n"
    tmp = ROOT / "ml" / "training" / "__init__.py"
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = _module_name(tmp).split(".")
            names.add(".".join([*base, node.module] if node.module else base))
    assert names == {"ml.training", "ml.training.deeper"}, names
