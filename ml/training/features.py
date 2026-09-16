"""Feature assembly and the declared feature order (plan §J, Task 12, §3.2, D15).

`RiskFeaturesV1` is **five** features, in one fixed order that is part of the
contract and is recorded in an artifact's `feature_spec`:

1. ``submitted_hour``
2. ``submitted_weekday``
3. ``text_length``
4. ``category_mean_resolution_hours``
5. ``category_breach_rate``

`TransferFeaturesV1` is the three of those computable in both corpora, versioned
independently, and used only by §5.4's reduced-feature cross-domain cross-target
robustness probe.

**The spec, not the breadth of the serving interface, decides what is built**
(§3.3, D12). ``build_features`` produces exactly the columns ``spec.names``
names, in exactly that order, and raises `FeatureUnavailable` naming any feature
these inputs cannot supply. It never reorders a mismatched spec, never fills a
column it could not build, and never returns a partial matrix. `ml.base.
RiskFeatures` keeps all ten serving-time fields, but `sla_hours` (D15),
`priority_rank`, `queue_depth`, `assignee_open_count` (§3) and `age_hours`
(§3.1) are not produced here, so a spec naming one of them raises. That is how
D15's removal is enforced rather than merely documented.

**Hour and weekday come from the source's local representation, never from the
UTC one** (§2.4, D21, D32). `CorpusRecord.submitted_at` is the stored UTC
instant, so 311 is converted back to `America/New_York` civil time through
`ingest.sources.nyc311.to_source_local` — the same conversion Task 8's
diagnostic uses, so one feature name has one meaning. CFPB's published offset is
consumed and discarded at normalization, so its stored instant is the only
representation that exists and is used as-is (D32). ``submitted_weekday`` is
Python's ``datetime.weekday()``: Monday 0 through Sunday 6.

**The two target-derived columns are read from Task 11, never recomputed.**
Training rows carry out-of-fold values from `oof_category_aggregates`;
validation and test rows carry the frozen training values from
`apply_category_aggregates`. Both arrive as an `AggregateColumns` aligned by
position to ``records``. Nothing here fits a statistic, consults an outcome, or
touches a threshold — ``build_features`` takes no outcomes at all, which is what
makes a target unable to reach a feature through this module.

**`NaN` survives.** Warm-up rows have no out-of-fold value and an undefined
statistic is `NaN`; both pass through unchanged. No imputation happens here, at
any width, for any column — `HistGradientBoostingClassifier` handles `NaN`
natively, and filling one would reintroduce the leak §6.3 exists to prevent. No
scaling or normalization happens here either.

The matrix is `float64` because it must carry `NaN`, so the three integral
features arrive as whole floats.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ingest.schema import CorpusRecord
from ingest.sources import nyc311
from ml.training.aggregates import AggregateColumns


class FeatureUnavailable(ValueError):
    """A `FeatureSpec` names a feature these inputs cannot supply.

    Raised before any column is built, so a caller never receives a partial or
    silently reordered matrix. A `ValueError` because it reports an unusable
    argument, consistent with the rest of `ml.training`.
    """


@dataclass(frozen=True)
class FeatureSpec:
    """An ordered, versioned feature list — the inference compatibility guard.

    Task 15 writes ``names`` and ``version`` into an artifact's ``metadata.json``
    and enforces them at load time; this module is what constructs exactly those
    features in exactly that order.
    """

    names: tuple[str, ...]
    """Ordered. Position in this tuple is the column index, and nothing else."""
    version: str


RISK_FEATURES_V1 = FeatureSpec(
    names=(
        "submitted_hour",
        "submitted_weekday",
        "text_length",
        "category_mean_resolution_hours",
        "category_breach_rate",
    ),
    version="risk_features_v1",
)
"""Plan §J and addendum §3.2: the five features model v1 accepts, in order."""

TRANSFER_FEATURES_V1 = FeatureSpec(
    names=("submitted_hour", "submitted_weekday", "text_length"),
    version="transfer_features_v1",
)
"""§5.4 and D16: the three features computable in both corpora, versioned
independently of `RISK_FEATURES_V1` so the probe can never be read as the
primary model."""


def _local(record: CorpusRecord) -> datetime:
    """A record's submitted instant in the representation its source's hour and
    weekday are defined against (§2.4, D21, D32)."""
    instant = record.submitted_at
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError(
            f"{record.source}/{record.external_id}: submitted_at is naive "
            f"({instant!r}); a local hour would depend on the machine timezone"
        )
    if record.source == nyc311.SOURCE_SLUG:
        return nyc311.to_source_local(instant)
    return instant


_RECORD_FEATURES: dict[str, Callable[[CorpusRecord], float]] = {
    "submitted_hour": lambda record: float(_local(record).hour),
    "submitted_weekday": lambda record: float(_local(record).weekday()),
    "text_length": lambda record: float(len(record.text)),
}
"""Built from the record alone. `text_length` is the character count of
`CorpusRecord.text` exactly as published — whitespace included, no
normalization, and an empty narrative is 0 rather than missing."""

_AGGREGATE_FEATURES = ("category_mean_resolution_hours", "category_breach_rate")
"""Read from Task 11's `AggregateColumns`. Never computed here."""


def build_features(
    records: Sequence[CorpusRecord],
    aggregates: AggregateColumns | None,
    spec: FeatureSpec,
) -> np.ndarray:
    """Assemble ``spec.names`` as columns, in that order, one row per record.

    ``aggregates`` aligns by position to ``records`` and may be ``None`` when the
    caller has none — a corpus without resolution times, such as CFPB. A spec
    naming an aggregate feature then raises `FeatureUnavailable`, which is how
    §5.4's "the primary model cannot be scored on CFPB at all" is an error rather
    than a silent fallback.

    Raises `FeatureUnavailable` for a feature these inputs cannot supply, and
    ``ValueError`` for aggregates misaligned with ``records`` or a naive
    ``submitted_at``. Every check runs before the first column is built.
    """
    for name in spec.names:
        if name not in _RECORD_FEATURES and name not in _AGGREGATE_FEATURES:
            raise FeatureUnavailable(
                f"{name!r} is not produced by v1 feature assembly; it is not "
                "computable from a corpus record (addendum §3, §3.2, D15)"
            )

    wanted = [name for name in spec.names if name in _AGGREGATE_FEATURES]
    available: dict[str, Sequence[float]] = {}
    if wanted:
        if aggregates is None:
            raise FeatureUnavailable(
                f"{wanted[0]!r} needs category aggregates and none were supplied; "
                "pass out-of-fold aggregates for training rows or frozen "
                "training aggregates for validation and test rows (plan §J)"
            )
        available = {
            "category_mean_resolution_hours": aggregates.category_mean_resolution_hours,
            "category_breach_rate": aggregates.category_breach_rate,
        }
        for name, values in available.items():
            if len(values) != len(records):
                raise ValueError(f"{name} holds {len(values)} values for {len(records)} records")

    matrix = np.empty((len(records), len(spec.names)), dtype=np.float64)
    for index, name in enumerate(spec.names):
        if name in _RECORD_FEATURES:
            builder = _RECORD_FEATURES[name]
            matrix[:, index] = [builder(record) for record in records]
        else:
            matrix[:, index] = available[name]
    return matrix
