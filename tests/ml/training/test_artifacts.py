"""Task 15: artifact format, metadata and the feature_spec load guard (plan §P, D35).

Every test writes into pytest's ``tmp_path`` — never under ``ml/artifacts/`` — and
uses a tiny model fitted in-process. Nothing is downloaded and no socket is
opened; one test proves the latter by making every socket connection raise.
"""

import ast
import copy
import dataclasses
import inspect
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")
pytest.importorskip("joblib", reason="joblib lives in requirements/ml.txt")
linear_model = pytest.importorskip("sklearn.linear_model", reason="scikit-learn is in ml.txt")

from ingest.schema import CorpusRecord  # noqa: E402
from ml.training import artifacts  # noqa: E402
from ml.training.aggregates import AggregateColumns  # noqa: E402
from ml.training.artifacts import (  # noqa: E402
    ArtifactSchemaError,
    FeatureSpecMismatch,
    LoadedArtifact,
    load_artifact,
    write_artifact,
)
from ml.training.features import (  # noqa: E402
    RISK_FEATURES_V1,
    TRANSFER_FEATURES_V1,
    FeatureSpec,
    FeatureUnavailable,
    build_features,
)

REQUIRED_KEYS = (
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
NULLABLE_KEYS = ("label_roster", "thresholds", "warmup_row_count")
NON_NULLABLE_KEYS = tuple(key for key in REQUIRED_KEYS if key not in NULLABLE_KEYS)

CUSTOM_ORDER = ["submitted_hour", "text_length", "submitted_weekday"]


def metadata(**overrides: object) -> dict[str, object]:
    """A complete, valid metadata document. Overrides replace or add keys."""
    document: dict[str, object] = {
        "model_name": "probe",
        "model_version": "v1",
        "trained_at": "2026-09-17T12:00:00+00:00",
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        "corpus_id": "c0ffee",
        "corpus_schema_version": 1,
        "source_window": {"start": "2024-01-01T05:00:00+00:00", "end": "2025-01-01T04:59:59+00:00"},
        "split": {"train_end": "2024-09-01T00:00:00+00:00", "counts": {"train": 70}},
        "feature_spec": list(CUSTOM_ORDER),
        "feature_spec_version": "custom_order_v1",
        "label_roster": None,
        "thresholds": None,
        "metrics": {"test": {"pr_auc": {"score": 0.5, "baselines": {"majority": 0.1}}}},
        "warmup_row_count": None,
        "seeds": {"numpy": 0},
        "dependency_versions": {"numpy": "2.5.2", "scikit-learn": "1.9.0"},
    }
    document.update(overrides)
    return document


def version_dir(root: Path, document: dict[str, object]) -> Path:
    """``<root>/<domain>/<model_name>/<model_version>`` as plan §P lays it out."""
    return root / "nyc311" / str(document["model_name"]) / str(document["model_version"])


def records() -> list[CorpusRecord]:
    return [
        CorpusRecord(
            source="nyc311",
            external_id=str(i),
            text="x" * (3 + i),
            label="Noise",
            submitted_at=datetime(2024, 6, 3 + i, 13 + i, 30, tzinfo=UTC),
        )
        for i in range(4)
    ]


def fitted_model():
    X = np.array([[1.0, 4.0, 0.0], [2.0, 5.0, 1.0], [3.0, 6.0, 2.0], [4.0, 7.0, 3.0]])
    return linear_model.LogisticRegression().fit(X, np.array([0, 0, 1, 1]))


def write(root: Path, document: dict[str, object] | None = None, model: object = None) -> Path:
    document = metadata() if document is None else document
    path = version_dir(root, document)
    write_artifact(fitted_model() if model is None else model, document, path)
    return path


def rewrite_metadata(path: Path, text: str | bytes) -> None:
    target = path / "metadata.json"
    target.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)


def make_directory_alias(alias: Path, target: Path) -> None:
    """Point ``alias`` at ``target`` with a symlink, or a junction on Windows.

    Skips the calling test when the platform refuses both, rather than passing it
    without having exercised a link.
    """
    try:
        os.symlink(target, alias, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)], capture_output=True, text=True
        )
        if created.returncode == 0:
            return
    pytest.skip("this platform cannot create a directory symlink or junction for this test")


def remove_directory_alias(alias: Path) -> None:
    """Remove the link itself, never the directory it points at."""
    if sys.platform == "win32":
        os.rmdir(alias)
    else:
        alias.unlink()


class OrderClaimingModel:
    """A picklable model that advertises a different feature order than metadata."""

    feature_names_in_ = np.array(["submitted_weekday", "submitted_hour", "text_length"])

    def predict(self, X):
        return X[:, 0]


class UnpicklesLoudly:
    """Unpickling this raises, so a test can prove when the model is loaded."""

    def __reduce__(self):
        return (_explode, ())


def _explode():
    raise RuntimeError("the model file was unpickled")


# --- structure --------------------------------------------------------------------


def test_write_creates_exactly_the_model_and_metadata_files(tmp_path):
    path = write(tmp_path)
    assert path == tmp_path / "nyc311" / "probe" / "v1"
    assert sorted(p.name for p in path.iterdir()) == ["metadata.json", "model.joblib"]
    assert (path / "metadata.json").is_file()
    assert (path / "model.joblib").is_file()


def test_the_public_api_is_minimal(tmp_path):
    assert list(inspect.signature(write_artifact).parameters) == ["model", "metadata", "path"]
    assert list(inspect.signature(load_artifact).parameters) == ["path"]
    fields = [field.name for field in dataclasses.fields(LoadedArtifact)]
    assert fields == ["model", "metadata", "feature_spec"]
    builder = inspect.signature(LoadedArtifact.build_features).parameters
    assert list(builder) == ["self", "records", "aggregates"]
    assert builder["aggregates"].default is None
    assert not hasattr(LoadedArtifact, "predict")
    assert issubclass(ArtifactSchemaError, ValueError)
    assert issubclass(FeatureSpecMismatch, ValueError)


# --- valid round trip ----------------------------------------------------------------


def test_valid_metadata_loads_and_is_recorded_exactly(tmp_path):
    document = metadata()
    path = write(tmp_path, document)
    loaded = load_artifact(path)
    assert isinstance(loaded, LoadedArtifact)
    on_disk = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk == document


def test_object_key_order_survives_write_and_load(tmp_path):
    """Dict equality ignores order, so the order itself is asserted, at every level."""
    document = metadata(
        label_roster={"Zebra": 1, "Apple": 2, "Mango": 3},
        metrics={
            "test": {"pr_auc": {"score": 0.5, "baselines": {"majority": 0.1}}},
            "validation": {},
            "train": {},
        },
        split={"validation_end": "b", "train_end": "a", "counts": {"test": 3, "train": 7}},
    )
    path = write(tmp_path, document)
    loaded = load_artifact(path)
    assert list(loaded.metadata) == list(document)
    assert list(loaded.metadata["label_roster"]) == ["Zebra", "Apple", "Mango"]
    assert list(loaded.metadata["metrics"]) == ["test", "validation", "train"]
    assert list(loaded.metadata["metrics"]["test"]["pr_auc"]) == ["score", "baselines"]
    assert list(loaded.metadata["split"]) == ["validation_end", "train_end", "counts"]
    assert list(loaded.metadata["split"]["counts"]) == ["test", "train"]
    raw = (path / "metadata.json").read_text(encoding="utf-8")
    assert raw.index('"Zebra"') < raw.index('"Apple"') < raw.index('"Mango"')


def test_write_does_not_mutate_the_supplied_metadata(tmp_path):
    document = metadata()
    snapshot = copy.deepcopy(document)
    write(tmp_path, document)
    assert document == snapshot


def test_round_trip_predictions_are_identical_on_fixed_input(tmp_path):
    model = fitted_model()
    document = metadata()
    path = version_dir(tmp_path, document)
    write_artifact(model, document, path)
    loaded = load_artifact(path)
    X = loaded.build_features(records())
    assert np.array_equal(model.predict_proba(X), loaded.model.predict_proba(X))
    assert np.array_equal(model.predict(X), loaded.model.predict(X))


# --- feature_spec is authoritative -------------------------------------------------


def test_feature_spec_names_and_version_are_preserved(tmp_path):
    loaded = load_artifact(write(tmp_path))
    assert loaded.feature_spec == FeatureSpec(names=tuple(CUSTOM_ORDER), version="custom_order_v1")


def test_feature_order_is_preserved_exactly_through_building(tmp_path):
    loaded = load_artifact(write(tmp_path))
    built = loaded.build_features(records())
    expected = np.column_stack(
        [
            build_features(records(), None, FeatureSpec((name,), "one"))[:, 0]
            for name in CUSTOM_ORDER
        ]
    )
    assert built.shape == (4, 3)
    assert np.array_equal(built, expected)


def test_a_reversed_feature_order_stays_reversed(tmp_path):
    reversed_names = list(reversed(RISK_FEATURES_V1.names))
    document = metadata(feature_spec=reversed_names, feature_spec_version="reversed_v1")
    loaded = load_artifact(write(tmp_path, document))
    assert loaded.feature_spec.names == tuple(reversed_names)
    aggregates = AggregateColumns((1.0, 2.0, 3.0, 4.0), (0.1, 0.2, 0.3, 0.4))
    built = loaded.build_features(records(), aggregates)
    assert np.array_equal(built, build_features(records(), aggregates, loaded.feature_spec))
    assert list(built[0, :2]) == [0.1, 1.0]


def test_the_builder_is_bound_to_the_artifact_spec(tmp_path):
    loaded = load_artifact(write(tmp_path))
    built = loaded.build_features(records())
    assert np.array_equal(built, build_features(records(), None, loaded.feature_spec))
    canonical = FeatureSpec(tuple(sorted(CUSTOM_ORDER)), "sorted")
    assert not np.array_equal(built, build_features(records(), None, canonical))


def test_features_outside_the_spec_are_never_requested(tmp_path):
    """A three-feature spec builds without aggregates; the full RiskFeaturesV1 cannot."""
    loaded = load_artifact(write(tmp_path))
    assert loaded.build_features(records(), None).shape == (4, 3)
    with pytest.raises(FeatureUnavailable):
        build_features(records(), None, RISK_FEATURES_V1)


def test_no_fallback_to_the_ten_field_serving_interface(tmp_path):
    """A transfer artifact builds exactly its three stored columns, in stored order, with
    no aggregates. Falling back to RiskFeaturesV1 would demand aggregates and five
    columns; falling back to the ten-field serving interface is not buildable at all."""
    document = metadata(
        feature_spec=list(TRANSFER_FEATURES_V1.names),
        feature_spec_version=TRANSFER_FEATURES_V1.version,
    )
    loaded = load_artifact(write(tmp_path, document))
    assert loaded.feature_spec == TRANSFER_FEATURES_V1
    built = loaded.build_features(records())
    expected = np.column_stack(
        [
            build_features(records(), None, FeatureSpec((name,), "one"))[:, 0]
            for name in ("submitted_hour", "submitted_weekday", "text_length")
        ]
    )
    assert built.shape == (4, 3)
    assert np.array_equal(built, expected)


def test_feature_order_comes_from_metadata_not_the_model(tmp_path):
    loaded = load_artifact(write(tmp_path, model=OrderClaimingModel()))
    assert loaded.feature_spec.names == tuple(CUSTOM_ORDER)


def test_aggregate_features_build_from_supplied_aggregates_and_fail_without_them(tmp_path):
    """D35: the artifact binds the spec, never aggregate values."""
    document = metadata(
        feature_spec=list(RISK_FEATURES_V1.names), feature_spec_version="risk_features_v1"
    )
    loaded = load_artifact(write(tmp_path, document))
    aggregates = AggregateColumns((1.0, 2.0, 3.0, 4.0), (0.1, 0.2, 0.3, 0.4))
    built = loaded.build_features(records(), aggregates)
    assert np.array_equal(built, build_features(records(), aggregates, RISK_FEATURES_V1))
    with pytest.raises(FeatureUnavailable, match="category_mean_resolution_hours"):
        loaded.build_features(records())


# --- the load guard: FeatureSpecMismatch -----------------------------------------------


def test_an_artifact_declaring_queue_depth_fails_to_load_naming_it(tmp_path):
    """The deliberately mismatched artifact fixture (plan Task 15 acceptance)."""
    document = metadata(feature_spec=["submitted_hour", "queue_depth"])
    path = write(tmp_path, document)
    with pytest.raises(FeatureSpecMismatch, match="queue_depth"):
        load_artifact(path)


def test_every_unavailable_feature_is_named_and_available_ones_are_not(tmp_path):
    document = metadata(feature_spec=["submitted_hour", "queue_depth", "text_length", "sla_hours"])
    path = write(tmp_path, document)
    with pytest.raises(FeatureSpecMismatch) as caught:
        load_artifact(path)
    message = str(caught.value)
    assert "queue_depth" in message and "sla_hours" in message
    assert "submitted_hour" not in message and "text_length" not in message
    assert message.index("queue_depth") < message.index("sla_hours")


def test_an_unavailable_feature_is_never_silently_dropped(tmp_path):
    path = write(tmp_path, metadata(feature_spec=["submitted_hour", "queue_depth"]))
    with pytest.raises(FeatureSpecMismatch):
        load_artifact(path)


def test_the_guard_runs_before_the_model_is_unpickled(tmp_path):
    path = write(tmp_path, metadata(feature_spec=["queue_depth"]), model=UnpicklesLoudly())
    with pytest.raises(FeatureSpecMismatch):
        load_artifact(path)


def test_writing_does_not_check_feature_availability(tmp_path):
    """The guard is a load-time compatibility check (plan Task 15, D35)."""
    path = write(tmp_path, metadata(feature_spec=["queue_depth"]))
    assert (path / "metadata.json").is_file()


# --- required fields and nullability -----------------------------------------------------


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_writing_metadata_missing_a_required_key_raises_naming_it(tmp_path, key):
    document = metadata()
    del document[key]
    path = tmp_path / "nyc311" / "probe" / "v1"
    with pytest.raises(ArtifactSchemaError, match=key):
        write_artifact(fitted_model(), document, path)
    assert not path.exists() or not any(path.iterdir())


def test_loading_metadata_missing_corpus_id_raises_naming_it(tmp_path):
    path = write(tmp_path)
    document = metadata()
    del document["corpus_id"]
    rewrite_metadata(path, json.dumps(document))
    with pytest.raises(ArtifactSchemaError, match="corpus_id"):
        load_artifact(path)


@pytest.mark.parametrize("key", NON_NULLABLE_KEYS)
def test_null_is_refused_for_a_non_nullable_key(tmp_path, key):
    document = metadata(**{key: None})
    with pytest.raises(ArtifactSchemaError, match=key):
        write_artifact(fitted_model(), document, tmp_path / "nyc311" / "probe" / "v1")


@pytest.mark.parametrize("key", NULLABLE_KEYS)
def test_null_is_accepted_for_a_nullable_key(tmp_path, key):
    document = metadata(label_roster=["a"], thresholds={}, warmup_row_count=0)
    document[key] = None
    loaded = load_artifact(write(tmp_path, document))
    assert loaded.metadata[key] is None


def test_non_null_values_are_accepted_for_nullable_keys(tmp_path):
    document = metadata(
        label_roster={"Noise": 3}, thresholds={"global_fallback": 12.5}, warmup_row_count=14
    )
    loaded = load_artifact(write(tmp_path, document))
    assert loaded.metadata["warmup_row_count"] == 14


def test_missing_provenance_is_never_invented(tmp_path):
    document = metadata()
    del document["trained_at"]
    del document["git_sha"]
    with pytest.raises(ArtifactSchemaError):
        write_artifact(fitted_model(), document, tmp_path / "nyc311" / "probe" / "v1")


# --- optional and unknown keys -------------------------------------------------------------


def test_optional_keys_may_be_absent_or_valid(tmp_path):
    document = metadata(
        experiment_label="reduced-feature cross-domain cross-target robustness probe",
        embedding_dimension=384,
        embedding_model_id="checkpoint",
        embedding_model_sha256="ab" * 32,
    )
    loaded = load_artifact(write(tmp_path, document))
    assert loaded.metadata["embedding_dimension"] == 384


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("experiment_label", None),
        ("experiment_label", ""),
        ("embedding_dimension", None),
        ("embedding_dimension", 0),
        ("embedding_dimension", True),
        ("embedding_model_id", ""),
        ("embedding_model_sha256", 5),
    ],
)
def test_an_invalid_optional_key_is_refused(tmp_path, key, bad):
    with pytest.raises(ArtifactSchemaError, match=key):
        write_artifact(fitted_model(), metadata(**{key: bad}), tmp_path / "nyc311" / "probe" / "v1")


def test_an_unknown_top_level_key_is_refused_naming_it(tmp_path):
    with pytest.raises(ArtifactSchemaError, match="corpusid"):
        write_artifact(fitted_model(), metadata(corpusid="x"), tmp_path / "nyc311" / "probe" / "v1")


def test_an_unknown_key_in_a_loaded_document_is_refused(tmp_path):
    path = write(tmp_path)
    rewrite_metadata(path, json.dumps(metadata(surprise=1)))
    with pytest.raises(ArtifactSchemaError, match="surprise"):
        load_artifact(path)


# --- types ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("model_name", ""),
        ("git_sha", 7),
        ("corpus_id", ""),
        ("feature_spec_version", ""),
        ("trained_at", "2026-09-17T12:00:00"),
        ("trained_at", "yesterday"),
        ("corpus_schema_version", "1"),
        ("corpus_schema_version", True),
        ("warmup_row_count", -1),
        ("warmup_row_count", 1.5),
        ("split", ["train"]),
        ("metrics", "good"),
        ("seeds", 0),
        ("dependency_versions", []),
        ("thresholds", [1.0]),
    ],
)
def test_a_wrongly_typed_field_is_refused_naming_it(tmp_path, key, bad):
    with pytest.raises(ArtifactSchemaError, match=key):
        write_artifact(fitted_model(), metadata(**{key: bad}), tmp_path / "nyc311" / "probe" / "v1")


@pytest.mark.parametrize(
    "bad_spec",
    [
        [],
        ["submitted_hour", "submitted_hour"],
        ["submitted_hour", ""],
        ["submitted_hour", 3],
        "submitted_hour",
    ],
)
def test_an_invalid_feature_spec_is_refused_not_repaired(tmp_path, bad_spec):
    with pytest.raises(ArtifactSchemaError, match="feature_spec"):
        write_artifact(
            fitted_model(), metadata(feature_spec=bad_spec), tmp_path / "nyc311" / "probe" / "v1"
        )


@pytest.mark.parametrize(
    "bad_spec",
    [["submitted_hour", "submitted_hour"], [], ["submitted_hour", 3]],
)
def test_a_hand_written_invalid_feature_spec_is_refused_at_load(tmp_path, bad_spec):
    """Validation is not only a write-side courtesy: a file edited on disk is refused too."""
    path = write(tmp_path)
    rewrite_metadata(path, json.dumps(metadata(feature_spec=bad_spec)))
    with pytest.raises(ArtifactSchemaError, match="feature_spec"):
        load_artifact(path)


# --- strict JSON ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_value_is_refused_at_write_and_nothing_is_written(tmp_path, bad):
    document = metadata(
        metrics={"test": {"pr_auc": {"score": bad, "baselines": {"majority": 0.1}}}}
    )
    path = tmp_path / "nyc311" / "probe" / "v1"
    with pytest.raises(ArtifactSchemaError, match=r"metrics\.test\.pr_auc\.score"):
        write_artifact(fitted_model(), document, path)
    assert not path.exists() or not any(path.iterdir())


@pytest.mark.parametrize(
    "bad_value",
    [
        {1: 0.5},
        {"count": np.int64(3)},
        {"when": datetime(2024, 1, 1, tzinfo=UTC)},
        {"pair": {1, 2}},
    ],
)
def test_a_value_json_cannot_represent_exactly_is_refused_not_converted(tmp_path, bad_value):
    with pytest.raises(ArtifactSchemaError):
        write_artifact(
            fitted_model(), metadata(seeds=bad_value), tmp_path / "nyc311" / "probe" / "v1"
        )


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        '["a", "list"]',
        json.dumps(list(REQUIRED_KEYS)),
        json.dumps(metadata()).replace('"score": 0.5', '"score": NaN'),
        json.dumps(metadata()).replace(
            '"corpus_id": "c0ffee"', '"corpus_id": "c0ffee", "corpus_id": "other"'
        ),
    ],
)
def test_malformed_metadata_on_disk_fails_loudly(tmp_path, text):
    path = write(tmp_path)
    rewrite_metadata(path, text)
    with pytest.raises(ArtifactSchemaError):
        load_artifact(path)


def test_metadata_that_is_not_utf8_fails_loudly(tmp_path):
    path = write(tmp_path)
    rewrite_metadata(path, b"\xff\xfe\x00{")
    with pytest.raises(ArtifactSchemaError):
        load_artifact(path)


def test_an_invalid_utf8_byte_inside_otherwise_valid_json_is_not_replaced(tmp_path):
    """Decoding with replacement would load corpus_id as 'c0\\ufffdee' without a word."""
    path = write(tmp_path)
    valid = json.dumps(metadata()).encode("utf-8")
    assert valid.count(b'"c0ffee"') == 1
    rewrite_metadata(path, valid.replace(b'"c0ffee"', b'"c0\xffee"'))
    with pytest.raises(ArtifactSchemaError, match="UTF-8"):
        load_artifact(path)


def test_a_schema_error_is_raised_before_the_model_is_unpickled(tmp_path):
    path = write(tmp_path, model=UnpicklesLoudly())
    document = metadata()
    del document["seeds"]
    rewrite_metadata(path, json.dumps(document))
    with pytest.raises(ArtifactSchemaError, match="seeds"):
        load_artifact(path)


# --- path semantics ----------------------------------------------------------------------------


def test_the_directory_name_must_equal_model_version(tmp_path):
    with pytest.raises(ArtifactSchemaError, match="model_version"):
        write_artifact(fitted_model(), metadata(), tmp_path / "nyc311" / "probe" / "v2")


def test_the_parent_directory_name_must_equal_model_name(tmp_path):
    with pytest.raises(ArtifactSchemaError, match="model_name"):
        write_artifact(fitted_model(), metadata(), tmp_path / "nyc311" / "other" / "v1")


def test_a_renamed_artifact_directory_fails_to_load(tmp_path):
    path = write(tmp_path)
    moved = path.parent / "v9"
    path.rename(moved)
    with pytest.raises(ArtifactSchemaError, match="model_version"):
        load_artifact(moved)


def test_a_string_path_is_accepted(tmp_path):
    path = write(tmp_path)
    assert load_artifact(str(path)).feature_spec.names == tuple(CUSTOM_ORDER)


def test_dot_dot_is_normalised_lexically_before_the_names_are_compared(tmp_path):
    """D35: ``probe/v1/../v1`` names ``probe/v1`` once normalised as text, so it loads,
    even though its unnormalised parent component is ``..``."""
    path = write(tmp_path)
    spelled = path / ".." / "v1"
    assert spelled.parent.name == ".."
    assert load_artifact(spelled).metadata["model_version"] == "v1"


def test_a_link_named_for_another_version_cannot_load_that_version(tmp_path):
    """D8: the caller's path decides identity. A link ``v2 -> v1`` must not load v1 as v2."""
    real = write(tmp_path)
    alias = real.parent / "v2"
    make_directory_alias(alias, real)
    try:
        assert (alias / "metadata.json").is_file(), "the alias must really reach artifact v1"
        with pytest.raises(ArtifactSchemaError, match="model_version"):
            load_artifact(alias)
    finally:
        remove_directory_alias(alias)
    assert load_artifact(real).metadata["model_version"] == "v1"


# --- immutability and the validity boundary ----------------------------------------------------


def test_a_published_version_directory_is_never_overwritten(tmp_path):
    path = write(tmp_path)
    before = (path / "metadata.json").read_bytes(), (path / "model.joblib").read_bytes()
    with pytest.raises(FileExistsError):
        write_artifact(fitted_model(), metadata(git_sha="f" * 40), path)
    assert ((path / "metadata.json").read_bytes(), (path / "model.joblib").read_bytes()) == before


def test_a_directory_holding_only_a_model_file_is_refused(tmp_path):
    path = tmp_path / "nyc311" / "probe" / "v1"
    path.mkdir(parents=True)
    (path / "model.joblib").write_bytes(b"stale")
    with pytest.raises(FileExistsError):
        write_artifact(fitted_model(), metadata(), path)


def test_an_empty_existing_directory_may_be_written(tmp_path):
    path = tmp_path / "nyc311" / "probe" / "v1"
    path.mkdir(parents=True)
    write_artifact(fitted_model(), metadata(), path)
    assert (path / "metadata.json").is_file()


def test_a_failed_metadata_write_leaves_nothing_loadable(tmp_path, monkeypatch):
    path = tmp_path / "nyc311" / "probe" / "v1"

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts.os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        write_artifact(fitted_model(), metadata(), path)
    monkeypatch.undo()
    assert not (path / "metadata.json").exists()
    assert [p.name for p in path.iterdir()] == ["model.joblib"]
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        load_artifact(path)


def test_metadata_is_written_after_the_model(tmp_path, monkeypatch):
    order: list[str] = []
    real_dump = artifacts.joblib.dump
    real_replace = artifacts.os.replace

    def dump(model, target, *args, **kwargs):
        order.append(Path(target).name)
        return real_dump(model, target, *args, **kwargs)

    def replace(source, target):
        order.append(Path(target).name)
        return real_replace(source, target)

    monkeypatch.setattr(artifacts.joblib, "dump", dump)
    monkeypatch.setattr(artifacts.os, "replace", replace)
    write(tmp_path)
    assert order == ["model.joblib", "metadata.json"]


def test_a_missing_model_file_fails_loudly(tmp_path):
    path = write(tmp_path)
    (path / "model.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="model.joblib"):
        load_artifact(path)


def test_a_missing_metadata_file_fails_loudly(tmp_path):
    path = write(tmp_path)
    (path / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        load_artifact(path)


def test_an_unreadable_model_file_fails_loudly(tmp_path):
    """D35 fixes no exception type for a corrupt model; it must not load silently."""
    path = write(tmp_path)
    (path / "model.joblib").write_bytes(b"not a joblib file")
    with pytest.raises(Exception):
        load_artifact(path)


# --- the loaded artifact is read-only ------------------------------------------------------------


def test_the_loaded_artifact_and_its_metadata_are_read_only(tmp_path):
    loaded = load_artifact(write(tmp_path))
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.feature_spec = RISK_FEATURES_V1  # type: ignore[misc]
    with pytest.raises(TypeError):
        loaded.metadata["corpus_id"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded.metadata["seeds"]["numpy"] = 99  # type: ignore[index]
    with pytest.raises(AttributeError):
        loaded.metadata["feature_spec"].append("queue_depth")  # type: ignore[union-attr]


# --- scope: no network, no Django, no serving -----------------------------------------------------


def test_no_network_is_used_to_write_load_or_build(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    loaded = load_artifact(write(tmp_path))
    assert loaded.build_features(records()).shape == (4, 3)


def test_importing_the_artifact_module_pulls_in_no_django_or_heavy_training_stack():
    probe = (
        "import sys; import ml.training.artifacts; "
        "heavy = sorted({m.split('.')[0] for m in sys.modules} & "
        "{'django', 'sklearn', 'scipy', 'pandas', 'pyarrow', 'onnxruntime'}); "
        "print(','.join(heavy))"
    )
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=root,
        env=dict(os.environ),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_the_serving_registry_does_not_use_artifacts():
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / "ml" / "registry.py").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("ml.training") for name in imported)
    assert "load_artifact" not in (root / "ml" / "registry.py").read_text(encoding="utf-8")
