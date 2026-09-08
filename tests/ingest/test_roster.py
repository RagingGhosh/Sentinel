"""CFPB label roster: derived from data, asserted in both directions.

Every label in this module is synthetic. That is the point: addendum §1 says
"the label roster is derived, not hardcoded... No label list is transcribed into
code from this document", and Task 7's acceptance is that no spec label string
appears as a literal in `ingest/roster.py`. If these tests used the real
vocabulary they would pass just as happily against an implementation that had
the answer written into it.

The two responsibilities are kept apart, as the plan specifies:
`derive_roster` computes the locked roster from per-year label sets;
`assert_roster` checks an observed roster against it.
"""

import ast
from pathlib import Path

import pytest

from ingest.roster import RosterMismatch, assert_roster, derive_roster

ROSTER_SOURCE = Path("ingest/roster.py")


# --- derivation: the intersection across years -------------------------------


def test_the_roster_is_the_intersection_across_years():
    labels_by_year = {
        2023: {"Alpha", "Beta", "Gamma", "Delta"},
        2024: {"Alpha", "Beta", "Gamma", "Epsilon"},
        2025: {"Alpha", "Beta", "Gamma", "Zeta"},
    }
    assert derive_roster(labels_by_year) == frozenset({"Alpha", "Beta", "Gamma"})


def test_a_union_implementation_would_fail_this():
    """The distinction the addendum rests on: a label present in only some years
    is *not* stable, and a union would keep exactly those era-specific labels
    that make a temporal split a vocabulary mismatch rather than a test."""
    labels_by_year = {
        2023: {"Stable", "OnlyIn2023"},
        2024: {"Stable", "OnlyIn2024"},
    }
    derived = derive_roster(labels_by_year)

    assert derived == frozenset({"Stable"})
    union = {label for labels in labels_by_year.values() for label in labels}
    assert derived != union
    assert "OnlyIn2023" not in derived
    assert "OnlyIn2024" not in derived


def test_a_label_missing_from_one_year_is_excluded():
    labels_by_year = {
        2023: {"Kept", "Dropped"},
        2024: {"Kept", "Dropped"},
        2025: {"Kept"},
    }
    assert derive_roster(labels_by_year) == frozenset({"Kept"})


def test_one_year_yields_that_years_labels():
    assert derive_roster({2024: {"Alpha", "Beta"}}) == frozenset({"Alpha", "Beta"})


def test_years_sharing_nothing_yield_an_empty_roster():
    """Mathematically correct, and loud downstream: every observed label then
    reads as unexpected rather than an empty roster passing quietly."""
    assert derive_roster({2023: {"Alpha"}, 2024: {"Beta"}}) == frozenset()


def test_a_year_with_no_labels_empties_the_roster():
    assert derive_roster({2023: {"Alpha"}, 2024: set()}) == frozenset()


def test_no_years_at_all_raises():
    """The intersection of nothing is undefined; returning a roster would be
    inventing one."""
    with pytest.raises(ValueError, match="no years"):
        derive_roster({})


def test_the_result_is_an_immutable_frozenset():
    derived = derive_roster({2024: {"Alpha"}})
    assert isinstance(derived, frozenset)
    with pytest.raises(AttributeError):
        derived.add("Beta")  # type: ignore[attr-defined]


def test_derivation_does_not_mutate_its_input():
    labels_by_year = {2023: {"Alpha", "Beta"}, 2024: {"Alpha"}}
    before = {year: set(labels) for year, labels in labels_by_year.items()}
    derive_roster(labels_by_year)
    assert labels_by_year == before


# --- determinism -------------------------------------------------------------


def test_dictionary_insertion_order_does_not_change_the_result():
    forwards = {2023: {"Alpha", "Beta"}, 2024: {"Alpha", "Gamma"}}
    backwards = {2024: {"Alpha", "Gamma"}, 2023: {"Alpha", "Beta"}}
    assert derive_roster(forwards) == derive_roster(backwards)


def test_set_insertion_order_does_not_change_the_result():
    one = {2024: {"Alpha", "Beta", "Gamma"}}
    other_order: set[str] = set()
    for label in ("Gamma", "Alpha", "Beta"):
        other_order.add(label)
    assert derive_roster(one) == derive_roster({2024: other_order})


def test_repeated_derivation_is_stable():
    labels_by_year = {2023: {"Alpha", "Beta"}, 2024: {"Alpha", "Beta"}}
    assert derive_roster(labels_by_year) == derive_roster(labels_by_year)


# --- validation: both directions ---------------------------------------------


def test_an_exact_match_passes_and_returns_the_roster():
    locked = frozenset({"Alpha", "Beta"})
    observed = {"Alpha": 100, "Beta": 5}
    assert assert_roster(observed, locked) == locked


def test_an_unexpected_label_raises_and_the_message_names_it():
    locked = frozenset({"Alpha", "Beta"})
    observed = {"Alpha": 100, "Beta": 5, "Intruder": 7}

    with pytest.raises(RosterMismatch) as exc:
        assert_roster(observed, locked)

    assert "Intruder" in str(exc.value)
    assert exc.value.unexpected == {"Intruder": 7}
    assert exc.value.missing == frozenset()


def test_the_unexpected_label_carries_its_record_count():
    locked = frozenset({"Alpha"})
    observed = {"Alpha": 1, "Intruder": 4242}

    with pytest.raises(RosterMismatch) as exc:
        assert_roster(observed, locked)

    assert exc.value.unexpected["Intruder"] == 4242
    assert "4242" in str(exc.value), "§1.1 requires the record count be reported"


def test_several_unexpected_labels_each_carry_their_own_count():
    locked = frozenset({"Alpha"})
    observed = {"Alpha": 1, "First": 10, "Second": 20}

    with pytest.raises(RosterMismatch) as exc:
        assert_roster(observed, locked)

    assert exc.value.unexpected == {"First": 10, "Second": 20}
    message = str(exc.value)
    assert "10" in message and "20" in message


def test_a_missing_label_raises_and_the_message_names_it():
    locked = frozenset({"Alpha", "Beta", "Vanished"})
    observed = {"Alpha": 100, "Beta": 5}

    with pytest.raises(RosterMismatch) as exc:
        assert_roster(observed, locked)

    assert "Vanished" in str(exc.value)
    assert exc.value.missing == frozenset({"Vanished"})
    assert exc.value.unexpected == {}


def test_a_swapped_label_raises_even_though_the_count_is_unchanged():
    """The case D13 exists for. A count assertion passes here; membership does
    not. This is exactly how the CFPB taxonomy has changed before."""
    locked = frozenset({"Alpha", "Beta", "Gamma"})
    observed = {"Alpha": 1, "Beta": 2, "Renamed": 3}

    assert len(observed) == len(locked), "same size -- a count check would pass"

    with pytest.raises(RosterMismatch) as exc:
        assert_roster(observed, locked)

    assert exc.value.unexpected == {"Renamed": 3}
    assert exc.value.missing == frozenset({"Gamma"})


def test_the_error_reports_the_two_directions_separately():
    """§1.1 names them as distinct conditions; a merged 'labels differ' message
    would not say whether the taxonomy grew, shrank, or was renamed."""
    locked = frozenset({"Alpha", "Gone"})
    observed = {"Alpha": 1, "New": 2}

    with pytest.raises(RosterMismatch) as exc:
        assert_roster(observed, locked)

    message = str(exc.value)
    unexpected_at = message.find("New")
    missing_at = message.find("Gone")
    assert unexpected_at != -1 and missing_at != -1
    assert "unexpected" in message.lower()
    assert "missing" in message.lower()


def test_an_empty_observed_roster_reports_every_locked_label_missing():
    locked = frozenset({"Alpha", "Beta"})
    with pytest.raises(RosterMismatch) as exc:
        assert_roster({}, locked)
    assert exc.value.missing == locked


def test_validation_never_remaps_drops_or_expands():
    """The three prohibited repairs from §1.1, asserted as absent behaviour:
    an unexpected label raises rather than becoming a roster member."""
    locked = frozenset({"Alpha"})
    with pytest.raises(RosterMismatch):
        assert_roster({"Alpha": 1, "Surprise": 1}, locked)

    # The locked roster is untouched by the failed call.
    assert locked == frozenset({"Alpha"})
    assert "Surprise" not in locked


def test_validation_does_not_mutate_its_inputs():
    locked = frozenset({"Alpha", "Beta"})
    observed = {"Alpha": 1, "Beta": 2}
    assert_roster(observed, locked)
    assert observed == {"Alpha": 1, "Beta": 2}
    assert locked == frozenset({"Alpha", "Beta"})


def test_the_error_is_an_exception_carrying_structured_fields():
    exc = RosterMismatch(unexpected={"A": 1}, missing=frozenset({"B"}))
    assert isinstance(exc, Exception)
    assert exc.unexpected == {"A": 1}
    assert exc.missing == frozenset({"B"})


# --- acceptance: nothing from the taxonomy is written down here ---------------


# These are the only CFPB product strings the addendum spells out. They appear
# here, in a test, to assert their *absence* from the implementation.
SPEC_LABEL_LITERALS = (
    "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit reporting or other personal consumer reports",
    "Credit card",
    "Credit card or prepaid card",
    "Prepaid card",
)


def test_no_spec_label_string_appears_in_the_roster_module():
    """Task 7's stated acceptance criterion, checked literally."""
    source = ROSTER_SOURCE.read_text(encoding="utf-8")
    for label in SPEC_LABEL_LITERALS:
        assert label not in source, f"{label!r} must not be transcribed into roster.py"


def test_the_roster_module_contains_no_hardcoded_label_collection():
    """Structural, so it catches a roster written with labels we never listed.

    Any set/frozenset/list/tuple literal of four or more strings in this module
    would be a transcribed vocabulary; the real one is eleven wide.
    """
    tree = ast.parse(ROSTER_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Set | ast.List | ast.Tuple):
            strings = [
                e for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            assert len(strings) < 4, (
                f"line {node.lineno}: a literal collection of "
                f"{[e.value for e in strings]} looks like a hardcoded roster"
            )


def test_the_module_defines_no_module_level_label_constant():
    import ingest.roster as roster

    for name in dir(roster):
        if name.startswith("_"):
            continue
        value = getattr(roster, name)
        if isinstance(value, (frozenset, set, list, tuple)) and len(value) >= 4:
            if all(isinstance(item, str) for item in value):
                pytest.fail(f"{name} looks like a hardcoded roster: {value!r}")


# --- the Task 5 / Task 7 boundary --------------------------------------------


def test_the_cfpb_adapter_still_holds_no_roster():
    """Task 5 must stay taxonomy-agnostic; Task 7 owns the vocabulary."""
    adapter_source = Path("ingest/sources/cfpb.py").read_text(encoding="utf-8")
    for label in SPEC_LABEL_LITERALS:
        assert label not in adapter_source
    assert "roster" not in adapter_source.lower().split("deliberately not done here.")[0]


def test_an_unexpected_product_reaches_the_roster_layer_unchanged():
    """End to end across the boundary: the adapter passes an unknown product
    through verbatim, and the roster layer is what refuses it. If Task 5 had
    remapped it to a neighbour or to "Other", this failure would never happen
    and the population would have changed with nobody deciding to.
    """
    import json

    from ingest.sources.cfpb import normalize, rows_from_page

    page = json.loads(Path("tests/fixtures/cfpb_page.json").read_text(encoding="utf-8"))
    base = next(iter(rows_from_page(page)))

    record, _ = normalize(dict(base, product="A Product The Roster Never Locked"))
    assert record.label == "A Product The Roster Never Locked", "adapter passed it through"

    locked = derive_roster({2024: {"Known"}, 2025: {"Known"}})
    with pytest.raises(RosterMismatch) as exc:
        assert_roster({record.label: 1}, locked)

    assert exc.value.unexpected == {"A Product The Roster Never Locked": 1}


def test_roster_derivation_does_not_call_the_source_adapter():
    tree = ast.parse(ROSTER_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any("sources" in module for module in imported), (
        "the roster consumes collected label sets, not adapters"
    )


# --- architecture ------------------------------------------------------------


def test_the_module_is_django_free_and_dependency_free():
    tree = ast.parse(ROSTER_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for module in imported:
        root = module.split(".")[0]
        assert root != "django", f"roster must not import Django ({module})"
        assert root not in {"pandas", "pyarrow", "numpy"}, f"no training tier ({module})"
        assert not module.startswith("ml."), f"roster must not import ml ({module})"


def test_the_module_reads_no_clock_and_opens_no_files():
    tree = ast.parse(ROSTER_SOURCE.read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"now", "today", "utcnow"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "open" not in names
