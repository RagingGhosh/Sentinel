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
