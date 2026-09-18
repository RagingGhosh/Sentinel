"""Model artifacts whose metadata is authoritative about their features (plan §P, D35).

An artifact is one version directory, laid out as plan §P describes::

    ml/artifacts/<domain>/<model>/<version>/
        model.joblib
        metadata.json

`write_artifact` records a fitted model beside the metadata it is given, and
`load_artifact` returns a `LoadedArtifact` whose `build_features` is **bound to the
artifact's own `feature_spec`** (§3.3, D12). That binding is the point of the
module: a model trained on a subset of features can never be handed a different
vector at load or inference, because the builder constructs exactly the names the
artifact stores, in exactly the stored order, and nothing else — never the
ten-field `ml.base.RiskFeatures` interface, never a sorted or deduplicated list,
never an order read from the model object.

**The load guard.** Loading checks every stored feature name against what this
environment can produce, through Task 12's public `build_features`, and raises
`FeatureSpecMismatch` naming every name it cannot — before the model is
unpickled. The artifact binds the spec but never aggregate values: a spec naming
Task 11's aggregate columns needs the caller to pass ``aggregates`` to
`LoadedArtifact.build_features`, and without them Task 12 raises
`FeatureUnavailable` (the "primary artifact cannot score CFPB" case, plan Task 19).
Writing does not check availability; the guard belongs to load.

**The metadata schema is closed and checked on both sides** (D35). All sixteen base
fields must be present; only ``label_roster``, ``thresholds`` and
``warmup_row_count`` may be ``null``; the four optional fields must be valid when
present; unknown keys are refused. Types are checked shallowly — the inner shape
of ``split``, ``thresholds``, ``metrics`` and the rest belongs to the task that
produces them. Nothing is ever filled in, recomputed or coerced: a missing field,
a non-finite number, a non-string key or a value JSON cannot represent exactly
raises `ArtifactSchemaError`.

**A published version directory is immutable, and metadata is its validity
boundary** (D35, the D27 pattern). Everything is validated before anything is
written. The model is written first and ``metadata.json`` last, through a
temporary file in the same directory and ``os.replace``; a directory without
``metadata.json`` is not an artifact, so a write that fails part-way leaves nothing
loadable. The directory's name must equal ``model_version`` and its parent's name
``model_name``, read from the caller's path normalised as text: links are never
followed, so a link named for one version cannot load another.

**Only load trusted artifacts.** ``model.joblib`` is a pickle: loading it runs code
it contains. Artifacts are produced by this project's own experiments; nothing
here is a defence against a hostile file.

joblib only — ONNX belongs to Task 18. Training-side and Django-independent:
nothing here is imported by serving, and ``ml/registry.py`` is untouched.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import joblib
import numpy as np

from ingest.schema import CorpusRecord
from ml.training import features
from ml.training.aggregates import AggregateColumns

_MODEL_FILENAME = "model.joblib"
_METADATA_FILENAME = "metadata.json"

_REQUIRED_FIELDS = (
    "model_name",
    "model_version",
    "trained_at",
    "git_sha",
    "corpus_id",
    "corpus_schema_version",
    "source_window",
    "split",
    "feature_spec",
    "feature_spec_version",
    "label_roster",
    "thresholds",
    "metrics",
    "warmup_row_count",
    "seeds",
    "dependency_versions",
)
"""Plan §P's base fields. Every artifact carries every one."""

_NULLABLE_FIELDS = frozenset({"label_roster", "thresholds", "warmup_row_count"})
"""An explicit ``null`` means "not applicable to this kind of artifact" (D35)."""

_OPTIONAL_FIELDS = (
    "embedding_dimension",
    "embedding_model_id",
    "embedding_model_sha256",
    "experiment_label",
)
"""Plan §P's embedder and experiment fields: absent, or present and valid."""


class ArtifactSchemaError(ValueError):
    """An artifact's metadata or location breaks the schema D35 fixes."""


class FeatureSpecMismatch(ValueError):
    """The artifact names features this environment cannot produce (§3.3, D12)."""


@dataclass(frozen=True)
class LoadedArtifact:
    """A loaded model with its feature spec bound. Training-side; no serving API."""

    model: Any
    metadata: Mapping[str, Any]
    """Deeply read-only: objects are read-only mappings and arrays are tuples."""
    feature_spec: features.FeatureSpec
    """Exactly the stored names, in the stored order, with the stored version."""

    def build_features(
        self,
        records: Sequence[CorpusRecord],
        aggregates: AggregateColumns | None = None,
    ) -> np.ndarray:
        """Task 12's feature matrix for this artifact's spec, and no other.

        ``aggregates`` are the caller's out-of-fold or frozen training aggregates;
        the artifact never stores them.
        """
        return features.build_features(records, aggregates, self.feature_spec)


def write_artifact(model: Any, metadata: dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Record a fitted model and its metadata as a new, immutable version directory.

    ``path`` is the version directory itself. Raises `ArtifactSchemaError` for
    invalid metadata or a directory that disagrees with it, and `FileExistsError`
    when the directory already holds an artifact file. Nothing is written until
    every check has passed.
    """
    target = Path(path)
    text = _serialized(metadata)
    document = _parsed(text)
    _validate(document)
    _validate_location(target, document)
    for name in (_METADATA_FILENAME, _MODEL_FILENAME):
        if (target / name).exists():
            raise FileExistsError(
                f"{target} already holds {name}; a published version directory is immutable"
            )

    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / _MODEL_FILENAME)

    # metadata.json is the validity boundary, so it is written last and moved into
    # place whole: a reader finds either a complete artifact or no artifact (D27).
    handle, temporary = tempfile.mkstemp(dir=target, prefix=".metadata-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, target / _METADATA_FILENAME)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_artifact(path: str | os.PathLike[str]) -> LoadedArtifact:
    """Load an artifact, refusing it before unpickling if it is invalid or incompatible.

    Raises `FileNotFoundError` for a missing ``metadata.json`` or ``model.joblib``,
    `ArtifactSchemaError` for invalid metadata or a mismatched directory, and
    `FeatureSpecMismatch` naming every stored feature this environment cannot
    produce. Only trusted artifacts may be loaded: the model file is a pickle.
    """
    target = Path(path)
    metadata_file = target / _METADATA_FILENAME
    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"{metadata_file} does not exist; a directory without {_METADATA_FILENAME} "
            "is not an artifact"
        )
    try:
        text = metadata_file.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactSchemaError(f"{metadata_file} is not valid UTF-8") from exc

    document = _parsed(text)
    _validate(document)
    _validate_location(target, document)
    spec = features.FeatureSpec(
        names=tuple(document["feature_spec"]), version=document["feature_spec_version"]
    )
    _require_producible(spec)

    model_file = target / _MODEL_FILENAME
    if not model_file.is_file():
        raise FileNotFoundError(f"{model_file} does not exist")
    return LoadedArtifact(
        model=joblib.load(model_file), metadata=_read_only(document), feature_spec=spec
    )


# --- the feature guard ---------------------------------------------------------------


def _require_producible(spec: features.FeatureSpec) -> None:
    """Every stored name must be buildable here, asked of Task 12 one name at a time."""
    no_rows = AggregateColumns(category_mean_resolution_hours=(), category_breach_rate=())
    unavailable = []
    for name in spec.names:
        try:
            features.build_features((), no_rows, features.FeatureSpec((name,), spec.version))
        except features.FeatureUnavailable:
            unavailable.append(name)
    if unavailable:
        raise FeatureSpecMismatch(
            "the artifact's feature_spec names features this environment cannot "
            f"produce: {', '.join(unavailable)}"
        )


# --- strict JSON -----------------------------------------------------------------------


def _serialized(metadata: dict[str, Any]) -> str:
    """Strict JSON text for ``metadata``, refusing anything it would have to convert."""
    if not isinstance(metadata, dict):
        raise ArtifactSchemaError(f"metadata must be a dict, not {type(metadata).__name__}")
    _require_exact_json(metadata, "metadata")
    try:
        text = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArtifactSchemaError(f"metadata is not strict JSON: {exc}") from exc
    return text + "\n"


def _require_exact_json(value: Any, where: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactSchemaError(f"{where} is {value!r}; strict JSON has no non-finite values")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactSchemaError(f"{where} has the non-string key {key!r}")
            _require_exact_json(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_exact_json(item, f"{where}[{index}]")
        return
    raise ArtifactSchemaError(f"{where} is a {type(value).__name__}, which JSON cannot represent")


def _parsed(text: str) -> dict[str, Any]:
    """Parse strictly: no NaN or Infinity tokens, no repeated keys, an object at the top."""
    try:
        document = json.loads(
            text, parse_constant=_refuse_constant, object_pairs_hook=_refuse_repeated_keys
        )
    except json.JSONDecodeError as exc:
        raise ArtifactSchemaError(f"metadata is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ArtifactSchemaError("metadata must be a JSON object")
    return document


def _refuse_constant(token: str) -> None:
    raise ArtifactSchemaError(f"metadata contains the non-finite token {token}")


def _refuse_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ArtifactSchemaError(f"metadata repeats the key {key!r}")
        document[key] = value
    return document


# --- the schema ----------------------------------------------------------------------------


def _validate(document: Mapping[str, Any]) -> None:
    missing = [field for field in _REQUIRED_FIELDS if field not in document]
    if missing:
        raise ArtifactSchemaError(f"metadata is missing required fields: {', '.join(missing)}")
    known = set(_REQUIRED_FIELDS) | set(_OPTIONAL_FIELDS)
    unknown = [key for key in document if key not in known]
    if unknown:
        raise ArtifactSchemaError(f"metadata has unknown fields: {', '.join(unknown)}")

    for field in _REQUIRED_FIELDS:
        value = document[field]
        if value is None:
            if field in _NULLABLE_FIELDS:
                continue
            raise ArtifactSchemaError(f"{field} must not be null")
        _CHECKS[field](field, value)
    for field in _OPTIONAL_FIELDS:
        if field in document:
            if document[field] is None:
                raise ArtifactSchemaError(f"{field} must not be null when present")
            _CHECKS[field](field, document[field])


def _non_empty_string(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactSchemaError(f"{field} must be a non-empty string; got {value!r}")


def _aware_timestamp(field: str, value: Any) -> None:
    _non_empty_string(field, value)
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactSchemaError(f"{field} must be an ISO-8601 timestamp; got {value!r}") from exc
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ArtifactSchemaError(f"{field} must carry a timezone; got {value!r}")


def _integer(field: str, value: Any, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactSchemaError(f"{field} must be an integer; got {value!r}")
    if minimum is not None and value < minimum:
        raise ArtifactSchemaError(f"{field} must be at least {minimum}; got {value}")


def _json_object(field: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise ArtifactSchemaError(f"{field} must be a JSON object; got {type(value).__name__}")


def _feature_names(field: str, value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ArtifactSchemaError(f"{field} must be a non-empty JSON array of feature names")
    for name in value:
        if not isinstance(name, str) or not name:
            raise ArtifactSchemaError(f"{field} holds {name!r}, which is not a non-empty string")
    if len(set(value)) != len(value):
        raise ArtifactSchemaError(f"{field} repeats a feature name: {value!r}")


def _presence_only(field: str, value: Any) -> None:
    """Shape owned by the producing task; Task 15 checks presence and nullability only."""


_CHECKS = {
    "model_name": _non_empty_string,
    "model_version": _non_empty_string,
    "trained_at": _aware_timestamp,
    "git_sha": _non_empty_string,
    "corpus_id": _non_empty_string,
    "corpus_schema_version": _integer,
    "source_window": _presence_only,
    "split": _json_object,
    "feature_spec": _feature_names,
    "feature_spec_version": _non_empty_string,
    "label_roster": _presence_only,
    "thresholds": _json_object,
    "metrics": _json_object,
    "warmup_row_count": lambda field, value: _integer(field, value, minimum=0),
    "seeds": _json_object,
    "dependency_versions": _json_object,
    "embedding_dimension": lambda field, value: _integer(field, value, minimum=1),
    "embedding_model_id": _non_empty_string,
    "embedding_model_sha256": _non_empty_string,
    "experiment_label": _non_empty_string,
}


def _validate_location(target: Path, document: Mapping[str, Any]) -> None:
    """The directory must be ``<model_name>/<model_version>``; nothing is derived from it.

    The names are read from the path the caller gave, with ``.`` and ``..`` removed as
    text only (D8, D35). Links and junctions are deliberately not resolved: resolving
    would read the names of the link's target, letting ``v2 -> v1`` load v1 as v2.
    """
    location = Path(os.path.abspath(target))
    if location.name != document["model_version"]:
        raise ArtifactSchemaError(
            f"artifact directory {location.name!r} does not match "
            f"model_version {document['model_version']!r}"
        )
    if location.parent.name != document["model_name"]:
        raise ArtifactSchemaError(
            f"artifact parent directory {location.parent.name!r} does not match "
            f"model_name {document['model_name']!r}"
        )


def _read_only(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _read_only(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_read_only(item) for item in value)
    return value
