"""Corpus record types.

These describe external records ingested from public sources. They are **not**
Django models and never become `Complaint` rows: a corpus of two million CFPB
narratives and twenty-two million 311 requests is training data, not operational
state.

A record is identified by `(source, external_id)` — see `ingest.identity`. No
field here is an integer, so a corpus record can never be mistaken for, or
silently used as, a `Complaint` primary key.

The Phase 1 design's `RawComplaint` is withdrawn. Its `sla_met` field unified
two incomparable constructs: CFPB publishes whether a *company responded* inside
a regulatory window, while NYC 311 publishes how long *resolution took*. Those
are different questions, so each source carries its own outcome type and neither
is forced to have one at all — CFPB has no resolution time in the first place.
"""

from dataclasses import dataclass
from datetime import datetime

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CorpusRecord:
    """What both sources genuinely share. Nothing source-specific belongs here."""

    source: str
    """Domain pack slug: "cfpb" or "nyc311"."""

    external_id: str
    """The source's own identifier, kept as text exactly as published."""

    text: str
    label: str
    """The source's own category label, taken from the locked roster."""

    submitted_at: datetime


@dataclass(frozen=True)
class CFPBOutcome:
    """CFPB's regulatory responsiveness flag, and one provenance field."""

    external_id: str

    timely_response: bool
    """Whether the company responded inside CFPB's window. This is a fact about
    a company's reply, not a measure of resolution, and it is never combined
    with `NYC311Outcome.resolution_hours` under a shared name."""

    sent_to_company_at: datetime | None
    """Provenance evidence only — never a feature and never a target. It exists
    so the `date_received` to `date_sent_to_company` interval is computable from
    the corpus rather than only from the raw cache. Using it as a feature would
    be a defect: it is downstream of intake and unavailable for a live
    complaint."""


@dataclass(frozen=True)
class NYC311Outcome:
    """NYC 311's elapsed resolution time."""

    external_id: str

    closed_at: datetime | None
    """`None` for a request still open at ingest time."""

    resolution_hours: float | None
    """`closed_at - submitted_at` in hours, or `None` when the request is open.
    Not zero: an open request has no resolution time, and defaulting to zero
    would make it look instantly resolved."""
