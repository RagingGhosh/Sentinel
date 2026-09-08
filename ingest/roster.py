"""CFPB label roster: derived from data, asserted in both directions.

**No label is written down here, deliberately.** Addendum §1: "the label roster
is derived, not hardcoded... No label list is transcribed into code from this
document; the assumption is executable rather than documentary." A vocabulary
copied into source would agree with the spec forever and with the data never —
and CFPB has renamed this taxonomy at least twice already. A test parses this
module and fails if a literal collection of label-shaped strings appears in it.

Two separate responsibilities, kept separate:

`derive_roster` computes the locked roster as the **intersection** of the
`product` values present in each year of the window. Intersection, not union: a
label present in only some years is not stable, and keeping it is what makes a
temporal split a vocabulary mismatch rather than a generalisation test (§1).

`assert_roster` checks an observed roster against a locked one and fails in
**both directions** (§1.1) — an unexpected label with its record count, and a
locked label that has vanished. Membership, not count: a roster can stay eleven
labels wide while one label is replaced by another, which is precisely how this
taxonomy has changed before (D13). Dropping, remapping, mapping to `other` and
auto-expanding are all prohibited, because each changes the experimental
population underneath a published benchmark with nobody deciding to.

Where the locked roster lives is not this module's business: it comes from the
corpus manifest of the first successful ingest, never from source code.

Pure functions over supplied data. No adapter, no network, no filesystem, no
clock, no database, and Django-independent like the rest of `ingest/`.
"""

from collections.abc import Mapping

__all__ = ["RosterMismatch", "assert_roster", "derive_roster"]


class RosterMismatch(Exception):
    """The observed labels differ from the locked roster, in either direction.

    The two directions are carried separately rather than merged into one
    "labels differ" set, because they mean different things: an unexpected
    label may be a rename or a genuine addition, while a missing one means a
    category vanished from the window. §1.1 requires both to be reported, and
    the unexpected ones to carry their record counts.
    """

    def __init__(self, unexpected: Mapping[str, int], missing: frozenset[str]) -> None:
        self.unexpected: dict[str, int] = dict(unexpected)
        self.missing: frozenset[str] = frozenset(missing)
        super().__init__(self._describe())

    def _describe(self) -> str:
        parts = ["CFPB label roster does not match the locked roster."]

        if self.unexpected:
            listed = ", ".join(
                f"{label!r} ({count} records)" for label, count in sorted(self.unexpected.items())
            )
            parts.append(f"unexpected ({len(self.unexpected)}): {listed}")

        if self.missing:
            listed = ", ".join(repr(label) for label in sorted(self.missing))
            parts.append(f"missing ({len(self.missing)}): {listed}")

        parts.append(
            "A taxonomy change is a spec decision with a version bump, not an "
            "ingest-time inference: amend the addendum, restate the window and "
            "roster, and record that artifacts either side are not comparable."
        )
        return " ".join(parts)


def derive_roster(labels_by_year: Mapping[int, set[str]]) -> frozenset[str]:
    """The labels present in **every** year of the window.

    Raises `ValueError` when given no years at all: the intersection of nothing
    is undefined, and returning either an empty roster or a universal one would
    be inventing an answer. A year whose label set is empty is not an error —
    the intersection is then legitimately empty, and that surfaces loudly at
    `assert_roster` as every locked label having vanished.

    The result is a `frozenset`, so it carries no order to be mistaken for one.
    """
    if not labels_by_year:
        raise ValueError("cannot derive a roster from no years")

    years = iter(labels_by_year.values())
    stable = set(next(years))
    for labels in years:
        stable &= labels
    return frozenset(stable)


def assert_roster(observed: Mapping[str, int], locked: frozenset[str]) -> frozenset[str]:
    """Check observed labels against the locked roster; raise on any difference.

    `observed` maps each label seen in the window to its record count, the same
    shape the corpus manifest records. Counts are carried so the error can name
    how much data an unexpected label represents — one stray row and a renamed
    majority class need different responses, and a bare label name cannot tell
    them apart.

    Returns the locked roster on success, so a caller can bind it in one step.
    Raises `RosterMismatch` otherwise. Nothing is repaired: an unexpected label
    is never dropped, remapped or added to the roster.
    """
    unexpected = {label: count for label, count in observed.items() if label not in locked}
    missing = frozenset(label for label in locked if label not in observed)

    if unexpected or missing:
        raise RosterMismatch(unexpected=unexpected, missing=missing)

    return locked
