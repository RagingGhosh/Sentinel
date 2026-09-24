"""Task 19: the CFPB outcome sidecar (contract O1).

RED phase. None of the production surface below exists yet, so every test that
reaches it fails. The CFPB target `timely_response` is currently produced by
`normalize` and then discarded -- `ingest/cli.py` gates outcome persistence on
`if source == "nyc311"` -- which is why Task 19 cannot read its own evaluation
target until this lands.

The API surface these tests pin, all of it implied by the frozen contract and
none of it existing yet:

    ingest.storage.CFPB_OUTCOME_ARROW_SCHEMA     exactly three columns
    ingest.storage.write_cfpb_outcome_partition(outcomes, source, year, part, root=)
    ingest.storage.read_cfpb_outcome_parts(paths) -> Iterator[CFPBOutcome]
    ingest.manifest.load_cfpb_outcomes(years=None, root=) -> (manifest, iterator)

NYC 311's `write_outcome_partition`, `read_outcome_parts` and `load_outcomes`
keep their types and semantics untouched; this module proves that too.

Everything is written into `tmp_path`. Nothing is downloaded and no socket is
opened.
"""

import importlib
from datetime import UTC, datetime

import pytest

pytest.importorskip("pyarrow", reason="pyarrow lives in requirements/train.txt")

from ingest.manifest import (  # noqa: E402
    CorpusIntegrityError,
    OutcomeSidecarNotFound,
    build_manifest,
    load_corpus,
    load_outcomes,
    write_manifest,
)
from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome  # noqa: E402
from ingest.storage import (  # noqa: E402
    outcome_partition_path,
    write_outcome_partition,
    write_partition,
)
from tests.ingest.test_cli import a_cfpb_run, cfpb_row  # noqa: E402

SOURCE = "cfpb"
WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
WINDOW_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
SENT = datetime(2024, 3, 16, 13, 0, tzinfo=UTC)

#: The contract's three fields, and only these.
CFPB_OUTCOME_FIELDS = ("external_id", "timely_response", "date_sent_to_company")

TIMESTAMP_DIAGNOSTIC = {
    "verdict": "supported_plausible_event_time",
    "reason": "fixture corpus; no provenance measurement is claimed",
}


# --- the production modules, imported late ------------------------------------------


def storage():
    return importlib.import_module("ingest.storage")


def manifest_module():
    return importlib.import_module("ingest.manifest")


# --- fixtures -----------------------------------------------------------------------


def cfpb_record(external_id: str, *, product: str = "Mortgage") -> CorpusRecord:
    return CorpusRecord(
        source=SOURCE,
        external_id=external_id,
        text=f"narrative {external_id}",
        label=product,
        submitted_at=datetime(2024, 3, 15, 13, 0, tzinfo=UTC),
    )


def cfpb_outcome(external_id: str, *, timely: bool, sent: datetime | None = SENT) -> CFPBOutcome:
    return CFPBOutcome(
        external_id=external_id,
        timely_response=timely,
        sent_to_company_at=sent,
    )


def a_cfpb_sidecar(root, outcomes=None):
    """A real CFPB corpus with an outcome sidecar beside its records."""
    outcomes = (
        outcomes
        if outcomes is not None
        else [
            cfpb_outcome("1", timely=True),
            cfpb_outcome("2", timely=False, sent=None),
        ]
    )
    records = [cfpb_record(outcome.external_id) for outcome in outcomes]
    write_partition(records, SOURCE, 2024, 0, root=root)
    storage().write_cfpb_outcome_partition(outcomes, SOURCE, 2024, 0, root=root)
    manifest = build_manifest(
        SOURCE,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=root,
        ingested_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=root)
    return manifest


# --- A. the persisted schema --------------------------------------------------------


def test_the_cfpb_outcome_schema_has_exactly_the_three_contract_columns():
    """O1 fixes the field set. A fourth column is a schema decision, not a detail."""
    schema = storage().CFPB_OUTCOME_ARROW_SCHEMA
    assert tuple(schema.names) == CFPB_OUTCOME_FIELDS


def test_the_cfpb_outcome_columns_carry_their_contract_types():
    import pyarrow as pa

    schema = storage().CFPB_OUTCOME_ARROW_SCHEMA
    assert schema.field("external_id").type == pa.string()
    assert schema.field("timely_response").type == pa.bool_()
    assert schema.field("date_sent_to_company").type == pa.timestamp("us", tz="UTC")


def test_the_target_column_is_not_nullable():
    """`timely_response` is the evaluation target: a null would be a fabricated label.

    `ingest/sources/cfpb.py` already refuses a row whose `timely` is not exactly
    Yes or No, so an absent target cannot reach persistence.
    """
    assert not storage().CFPB_OUTCOME_ARROW_SCHEMA.field("timely_response").nullable


def test_the_provenance_column_is_nullable():
    """`date_sent_to_company` is provenance only, and CFPB does not always publish it."""
    assert storage().CFPB_OUTCOME_ARROW_SCHEMA.field("date_sent_to_company").nullable


def test_the_cfpb_schema_is_not_the_nyc311_schema():
    """Two sources, two sidecar schemas: Task 17's three columns are untouched (O1)."""
    module = storage()
    assert tuple(module.CFPB_OUTCOME_ARROW_SCHEMA.names) != tuple(module.OUTCOME_ARROW_SCHEMA.names)
    assert tuple(module.OUTCOME_ARROW_SCHEMA.names) == (
        "external_id",
        "resolution_hours",
        "closed_at",
    )


# --- B and I. mapping and round trip -------------------------------------------------


def test_a_cfpb_outcome_round_trips_through_the_sidecar(tmp_path):
    """The persisted column `date_sent_to_company` carries `sent_to_company_at`."""
    module = storage()
    written = [
        cfpb_outcome("1", timely=True, sent=SENT),
        cfpb_outcome("2", timely=False, sent=None),
    ]
    path = module.write_cfpb_outcome_partition(written, SOURCE, 2024, 0, root=tmp_path)
    loaded = list(module.read_cfpb_outcome_parts([path]))

    assert [o.external_id for o in loaded] == ["1", "2"]
    assert [o.timely_response for o in loaded] == [True, False]
    assert loaded[0].sent_to_company_at == SENT
    assert loaded[1].sent_to_company_at is None


def test_the_round_trip_returns_real_cfpb_outcome_objects(tmp_path):
    """Task 19 consumes `CFPBOutcome`; a dict or a tuple would need an adapter."""
    module = storage()
    path = module.write_cfpb_outcome_partition(
        [cfpb_outcome("1", timely=True)], SOURCE, 2024, 0, root=tmp_path
    )
    (loaded,) = module.read_cfpb_outcome_parts([path])
    assert isinstance(loaded, CFPBOutcome)


def test_the_timely_flag_survives_as_a_boolean_not_a_string(tmp_path):
    """`Yes`/`No` were resolved at normalisation; the sidecar stores the decision."""
    module = storage()
    path = module.write_cfpb_outcome_partition(
        [cfpb_outcome("1", timely=False)], SOURCE, 2024, 0, root=tmp_path
    )
    (loaded,) = module.read_cfpb_outcome_parts([path])
    assert loaded.timely_response is False


def test_outcomes_are_written_sorted_by_external_id(tmp_path):
    """Identity is the only stable order an outcome has, as for NYC 311."""
    module = storage()
    path = module.write_cfpb_outcome_partition(
        [cfpb_outcome("3", timely=True), cfpb_outcome("1", timely=False)],
        SOURCE,
        2024,
        0,
        root=tmp_path,
    )
    assert [o.external_id for o in module.read_cfpb_outcome_parts([path])] == ["1", "3"]


def test_a_naive_provenance_timestamp_is_refused(tmp_path):
    """Every timestamp in this corpus is aware; a naive one is a defect upstream."""
    with pytest.raises(ValueError):
        storage().write_cfpb_outcome_partition(
            [cfpb_outcome("1", timely=True, sent=datetime(2024, 3, 16, 13, 0))],
            SOURCE,
            2024,
            0,
            root=tmp_path,
        )


def test_a_nyc311_outcome_is_refused_by_the_cfpb_writer(tmp_path):
    """The two sidecars never accept each other's rows (O1, §4.3)."""
    with pytest.raises(ValueError):
        storage().write_cfpb_outcome_partition(
            [NYC311Outcome(external_id="1", closed_at=None, resolution_hours=None)],
            SOURCE,
            2024,
            0,
            root=tmp_path,
        )


def test_a_cfpb_part_under_another_source_is_refused_by_the_nyc311_reader(tmp_path):
    """Cross-source confusion fails closed at the reader, where it is detectable.

    Neither writer validates the `source` argument -- an outcome carries no source
    field, and `ingest/sources/__init__.py` keeps source names out of storage
    entirely. What is guaranteed instead is that a part is only ever read by the
    reader whose schema it matches: the NYC 311 reader refuses these bytes rather
    than mis-parsing a boolean as a resolution time.
    """
    path = storage().write_cfpb_outcome_partition(
        [cfpb_outcome("1", timely=True)], "nyc311", 2024, 0, root=tmp_path
    )
    from ingest.storage import read_outcome_parts

    with pytest.raises(ValueError):
        list(read_outcome_parts([path]))


# --- C. manifest integration ---------------------------------------------------------


def test_the_sidecar_files_are_listed_in_the_manifest(tmp_path):
    manifest = a_cfpb_sidecar(tmp_path)
    assert manifest.outcome_part_files, "the manifest declares no CFPB outcome sidecar"
    assert all("outcomes" in path for path in manifest.outcome_part_files)


def test_the_sidecar_bytes_bind_the_corpus_identity(tmp_path):
    """Changing an outcome changes `corpus_id`, as it does for NYC 311 (D37.16)."""
    first = a_cfpb_sidecar(tmp_path / "a")
    second = a_cfpb_sidecar(
        tmp_path / "b",
        outcomes=[cfpb_outcome("1", timely=False), cfpb_outcome("2", timely=False, sent=None)],
    )
    assert first.corpus_id != second.corpus_id


def test_a_record_only_cfpb_corpus_still_reads(tmp_path):
    """The sidecar is additive: a corpus without one stays valid for record consumers."""
    write_partition([cfpb_record("1")], SOURCE, 2024, 0, root=tmp_path)
    manifest = build_manifest(
        SOURCE,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=tmp_path,
        ingested_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=tmp_path)
    assert manifest.outcome_part_files == {}
    _, records = load_corpus(SOURCE, root=tmp_path)
    assert [r.external_id for r in records] == ["1"]


# --- D and J. the loader -------------------------------------------------------------


def test_the_cfpb_loader_returns_the_manifest_and_the_outcomes(tmp_path):
    written = a_cfpb_sidecar(tmp_path)
    manifest, outcomes = manifest_module().load_cfpb_outcomes(root=tmp_path)
    loaded = list(outcomes)
    assert manifest.corpus_id == written.corpus_id
    assert [o.external_id for o in loaded] == ["1", "2"]
    assert [o.timely_response for o in loaded] == [True, False]


def test_the_cfpb_loader_is_manifest_backed_not_a_directory_scan(tmp_path):
    """Leaving a part file out of the manifest is a refusal, never a silent read."""
    a_cfpb_sidecar(tmp_path)
    module = manifest_module()
    extra = outcome_partition_path(tmp_path, SOURCE, 2024, 1)
    storage().write_cfpb_outcome_partition(
        [cfpb_outcome("9", timely=True)], SOURCE, 2024, 1, root=tmp_path
    )
    assert extra.is_file()
    with pytest.raises(CorpusIntegrityError):
        list(module.load_cfpb_outcomes(root=tmp_path)[1])


def test_the_cfpb_loader_never_reads_the_raw_cache(tmp_path, monkeypatch):
    """The corpus is the only source of truth; a raw-cache fallback would bypass
    every checksum the manifest exists to enforce (§2.7, D27)."""
    a_cfpb_sidecar(tmp_path)
    import gzip
    import json as json_module

    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "page-0000.json.gz").write_bytes(
        gzip.compress(json_module.dumps({"hits": {"hits": []}}).encode("utf-8"))
    )
    manifest, outcomes = manifest_module().load_cfpb_outcomes(root=tmp_path)
    assert [o.external_id for o in outcomes] == ["1", "2"]


def test_a_years_filter_selects_only_those_partitions(tmp_path):
    module = storage()
    write_partition([cfpb_record("1")], SOURCE, 2024, 0, root=tmp_path)
    module.write_cfpb_outcome_partition(
        [cfpb_outcome("1", timely=True)], SOURCE, 2024, 0, root=tmp_path
    )
    later = CorpusRecord(
        source=SOURCE,
        external_id="2",
        text="narrative 2",
        label="Mortgage",
        submitted_at=datetime(2025, 3, 15, 13, 0, tzinfo=UTC),
    )
    write_partition([later], SOURCE, 2025, 0, root=tmp_path)
    module.write_cfpb_outcome_partition(
        [cfpb_outcome("2", timely=False)], SOURCE, 2025, 0, root=tmp_path
    )
    manifest = build_manifest(
        SOURCE,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=tmp_path,
        ingested_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=tmp_path)
    _, outcomes = manifest_module().load_cfpb_outcomes(years=[2025], root=tmp_path)
    assert [o.external_id for o in outcomes] == ["2"]


# --- G. integrity failures -----------------------------------------------------------


def test_an_absent_cfpb_sidecar_raises_the_typed_absence_error(tmp_path):
    """An absence, never an empty iterator: a caller looping over nothing would
    read "no outcomes" as "every company replied in time"."""
    write_partition([cfpb_record("1")], SOURCE, 2024, 0, root=tmp_path)
    manifest = build_manifest(
        SOURCE,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=tmp_path,
        ingested_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=tmp_path)
    with pytest.raises(OutcomeSidecarNotFound):
        manifest_module().load_cfpb_outcomes(root=tmp_path)


def test_a_missing_listed_part_is_refused(tmp_path):
    a_cfpb_sidecar(tmp_path)
    outcome_partition_path(tmp_path, SOURCE, 2024, 0).unlink()
    with pytest.raises(CorpusIntegrityError):
        list(manifest_module().load_cfpb_outcomes(root=tmp_path)[1])


def test_altered_sidecar_bytes_are_refused_before_anything_is_yielded(tmp_path):
    """Fail closed: the refusal comes before the first outcome, not part-way through."""
    a_cfpb_sidecar(tmp_path)
    path = outcome_partition_path(tmp_path, SOURCE, 2024, 0)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(CorpusIntegrityError):
        manifest_module().load_cfpb_outcomes(root=tmp_path)


def test_the_absence_error_is_distinct_from_a_damaged_sidecar(tmp_path):
    """Absent and corrupt are different operator problems with different fixes."""
    module = manifest_module()
    assert issubclass(OutcomeSidecarNotFound, CorpusIntegrityError)
    assert module.ChecksumMismatch is not OutcomeSidecarNotFound
    assert module.UnlistedPartFile is not OutcomeSidecarNotFound


# --- E. NYC 311 is untouched ---------------------------------------------------------


def test_nyc311_outcomes_still_load_through_their_own_api(tmp_path):
    """O1 preserves Task 17's loader: same name, same type, same semantics."""
    records = [
        CorpusRecord(
            source="nyc311",
            external_id="1",
            text="descriptor 1",
            label="Noise",
            submitted_at=datetime(2024, 3, 15, 13, 0, tzinfo=UTC),
        )
    ]
    outcomes = [
        NYC311Outcome(
            external_id="1",
            closed_at=datetime(2024, 3, 16, 1, 0, tzinfo=UTC),
            resolution_hours=12.0,
        )
    ]
    write_partition(records, "nyc311", 2024, 0, root=tmp_path)
    write_outcome_partition(outcomes, "nyc311", 2024, 0, root=tmp_path)
    manifest = build_manifest(
        "nyc311",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=tmp_path,
        ingested_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=tmp_path)

    _, loaded = load_outcomes("nyc311", root=tmp_path)
    (only,) = list(loaded)
    assert isinstance(only, NYC311Outcome)
    assert only.resolution_hours == 12.0
    assert only.closed_at == datetime(2024, 3, 16, 1, 0, tzinfo=UTC)


def test_the_nyc311_loader_still_returns_nyc311_outcomes_only(tmp_path):
    """Task 17's type contract is not made generic over outcome kinds (O1)."""
    import inspect

    annotation = str(inspect.signature(load_outcomes).return_annotation)
    assert "NYC311Outcome" in annotation
    assert "CFPBOutcome" not in annotation


def test_the_nyc311_writer_still_refuses_a_cfpb_outcome(tmp_path):
    with pytest.raises(ValueError):
        write_outcome_partition([cfpb_outcome("1", timely=True)], "nyc311", 2024, 0, root=tmp_path)


# --- F. the CLI persists what normalisation produced ---------------------------------


def test_a_cfpb_run_persists_the_outcome_stream(tmp_path):
    """The defect O1 exists to fix: `cli.py` currently discards CFPB outcomes."""
    a_cfpb_run(
        tmp_path,
        [[cfpb_row("1", sent="2024-03-16T09:00:00-04:00"), cfpb_row("2")]],
    )
    _, outcomes = manifest_module().load_cfpb_outcomes(root=tmp_path / "corpus")
    loaded = list(outcomes)
    assert [o.external_id for o in loaded] == ["1", "2"]
    assert all(o.timely_response is True for o in loaded)
    assert loaded[0].sent_to_company_at == datetime(2024, 3, 16, 13, 0, tzinfo=UTC)
    assert loaded[1].sent_to_company_at is None


def test_a_cfpb_run_persists_an_untimely_outcome_as_false(tmp_path):
    untimely = cfpb_row("1")
    untimely["timely"] = "No"
    a_cfpb_run(tmp_path, [[untimely, cfpb_row("2")]])
    _, outcomes = manifest_module().load_cfpb_outcomes(root=tmp_path / "corpus")
    assert [o.timely_response for o in outcomes] == [False, True]


def test_a_cfpb_run_binds_its_outcomes_into_the_manifest(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row("1")]])
    assert manifest.outcome_part_files, "the run wrote no CFPB outcome sidecar"


def test_a_replacement_cfpb_run_leaves_no_stale_outcome_partition(tmp_path):
    """D27's ordering: the tree is cleared before the new corpus is written."""
    a_cfpb_run(tmp_path, [[cfpb_row("1"), cfpb_row("2"), cfpb_row("3")]])
    a_cfpb_run(tmp_path / "second", [[cfpb_row("1")]])
    _, outcomes = manifest_module().load_cfpb_outcomes(root=tmp_path / "second" / "corpus")
    assert [o.external_id for o in outcomes] == ["1"]


# --- H. no operational-label pollution -----------------------------------------------


def test_the_timely_flag_never_becomes_the_record_label(tmp_path):
    """`CorpusRecord.label` stays the CFPB product taxonomy Task 16's roster reads."""
    untimely = cfpb_row("1", product="Mortgage")
    untimely["timely"] = "No"
    a_cfpb_run(tmp_path, [[untimely, cfpb_row("2", product="Mortgage")]])
    _, records = load_corpus(SOURCE, root=tmp_path / "corpus")
    labels = [record.label for record in records]
    assert labels == ["Mortgage", "Mortgage"]
    assert not any(label in {"Yes", "No", "True", "False", "timely"} for label in labels)


def test_the_record_label_does_not_vary_with_the_timely_flag(tmp_path):
    """Guard the guard: two records differing only in `timely` share one label."""
    untimely = cfpb_row("1", product="Credit card")
    untimely["timely"] = "No"
    a_cfpb_run(tmp_path, [[untimely, cfpb_row("2", product="Credit card")]])
    _, records = load_corpus(SOURCE, root=tmp_path / "corpus")
    assert {record.label for record in records} == {"Credit card"}


def test_the_corpus_record_schema_has_no_outcome_field():
    """The target lives in the sidecar, never on the record (§4.3)."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(CorpusRecord)}
    assert names == {"source", "external_id", "text", "label", "submitted_at"}
