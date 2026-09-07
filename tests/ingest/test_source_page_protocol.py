"""The `SourceAdapter` page type admits both real API shapes.

CFPB's search API returns an Elasticsearch-shaped object; Socrata returns a
top-level array. The protocol originally typed `page` as `Mapping[str, Any]`,
fitted to the only source that existed at the time, and a sequence-shaped
adapter could not satisfy it -- parameters are contravariant, so `Sequence` is
not an acceptable narrowing of `Mapping`.

These tests run mypy rather than asserting the alias exists, because the defect
was invisible at runtime: a sequence-shaped adapter *works* perfectly well when
called, and only a type-checker rejects it. A runtime-only test would have
passed against the broken protocol.
"""

import subprocess
import sys
from pathlib import Path

import pytest

TYPING_FIXTURES = Path(__file__).parent.parent / "typing"
REPO_ROOT = Path(__file__).parent.parent.parent


def run_mypy(fixture: str) -> tuple[int, str]:
    """Type-check one fixture from the repo root, so pyproject config applies."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(TYPING_FIXTURES / fixture)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


@pytest.fixture(scope="module")
def accepts() -> tuple[int, str]:
    return run_mypy("accepts_both_page_shapes.py")


@pytest.fixture(scope="module")
def rejects() -> tuple[int, str]:
    return run_mypy("rejects_narrowed_page.py")


def test_both_page_shapes_satisfy_the_protocol(accepts):
    """A mapping-shaped and a sequence-shaped adapter both type-check."""
    code, output = accepts
    assert code == 0, f"SourcePage must admit both API shapes:\n{output}"


def test_the_real_cfpb_adapter_still_satisfies_the_protocol(accepts):
    """The widened protocol must not break the adapter that already existed.

    `accepts_both_page_shapes.py` assigns a real `CFPBAdapter()` to a
    `SourceAdapter[CFPBOutcome]` variable, so a clean run covers this.
    """
    code, output = accepts
    assert code == 0, output
    source = (TYPING_FIXTURES / "accepts_both_page_shapes.py").read_text(encoding="utf-8")
    assert "cfpb_adapter: SourceAdapter[CFPBOutcome] = CFPBAdapter()" in source


def test_narrowing_the_page_back_to_mapping_is_rejected(rejects):
    """Non-vacuity: the union is load-bearing, not decorative.

    This is the pre-fix protocol expressed as an adapter. If it type-checked,
    `SourcePage` would be admitting a narrowing it must refuse, and a
    sequence-shaped source would silently be impossible again.
    """
    code, output = rejects
    assert code != 0, "an adapter narrowing page to Mapping must not satisfy the protocol"
    assert "rows_from_page" in output, f"the rejection must name the offending member:\n{output}"


def test_source_page_admits_the_two_shapes_at_runtime():
    """The alias itself, checked structurally rather than by name."""
    from collections.abc import Mapping, Sequence
    from typing import get_args

    from ingest.sources.base import SourcePage

    branches = get_args(SourcePage)
    assert len(branches) == 2, f"expected exactly two branches, got {branches}"

    origins = {getattr(b, "__origin__", b) for b in branches}
    assert Mapping in origins, "a mapping-shaped page (CFPB) must be admitted"
    assert Sequence in origins, "a sequence-shaped page (Socrata) must be admitted"


def test_source_page_is_not_a_blanket_any():
    """`Any` would have silenced the error instead of describing the domain."""
    from typing import Any, get_args

    from ingest.sources.base import SourcePage

    assert Any not in get_args(SourcePage)
    assert SourcePage is not Any


def test_the_protocol_module_stays_django_free():
    import ast

    tree = ast.parse((REPO_ROOT / "ingest" / "sources" / "base.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for module in imported:
        assert module.split(".")[0] != "django", f"protocol must not import Django ({module})"
        assert not module.startswith("ml."), f"protocol must not import ml ({module})"
