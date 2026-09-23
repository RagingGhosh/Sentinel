"""Task 18: the MiniLM benchmark arm and its pinned assets (D38).

RED phase. `ml.embedders.minilm` does not exist, so every test that reaches it
fails. The asset-shaped tests build their own fake directories under `tmp_path`;
nothing is downloaded, no socket is opened, and the real 90MB export is never
required for an ordinary run.

The API surface these tests pin, all of it implied by D38 and none of it existing
yet:

    ml.embedders.minilm.CHECKPOINT / REVISION / ONNX_FILENAME
    ml.embedders.minilm.ONNX_SHA256 / TOKENIZER_SHA256
    ml.embedders.minilm.MODEL_VERSION / EMBEDDING_MODEL_ID
    ml.embedders.minilm.MAX_SEQUENCE_TOKENS / PACKAGED_TOKENIZER_MAX / GRAPH_POSITION_LIMIT
    ml.embedders.minilm.ModelAssetUnavailable / ModelAssetMismatch
    ml.embedders.minilm.asset_dir() / load_minilm(...)

Tests needing the real assets skip when they are absent, naming the path they
wanted; `SENTINEL_REQUIRE_MINILM=1` turns that skip into a failure (D38).
"""

import ast
import hashlib
import importlib
import json
import os
import socket
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")

pytestmark = pytest.mark.ml

ROOT = Path(__file__).resolve().parents[3]
EMBEDDERS_DIR = ROOT / "ml" / "embedders"
GITIGNORE = ROOT / ".gitignore"
ML_REQUIREMENTS = ROOT / "requirements" / "ml.txt"
BASE_REQUIREMENTS = ROOT / "requirements" / "base.txt"

# --- D38's frozen asset identity ----------------------------------------------------

CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ONNX_FILENAME = "model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
ONNX_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
MODEL_VERSION = "all_minilm_l6_v2_onnx_v1"
DEFAULT_ASSET_DIR = Path("ml/artifacts/embedders/all_minilm_l6_v2/v1")
ASSET_DIR_ENV = "SENTINEL_MINILM_DIR"
REQUIRE_ENV = "SENTINEL_REQUIRE_MINILM"

#: The three limits D38 refuses to conflate.
PACKAGED_TOKENIZER_MAX = 128
SENTINEL_MAX_TOKENS = 256
GRAPH_POSITION_LIMIT = 512

#: Exports D38 excludes. A benchmark that cannot name its bytes is not a benchmark.
EXCLUDED_EXPORTS = (
    "model_O1.onnx",
    "model_O2.onnx",
    "model_O3.onnx",
    "model_O4.onnx",
    "model_qint8_arm64.onnx",
    "model_qint8_avx512.onnx",
    "model_qint8_avx512_vnni.onnx",
    "model_quint8_avx2.onnx",
)

TOKENIZER = "tokenizers"
TOKENIZER_PIN = "tokenizers==0.23.2"


# --- the production module, imported late -------------------------------------------


def minilm():
    return importlib.import_module("ml.embedders.minilm")


# --- the real assets, when a machine happens to have them ---------------------------


def real_asset_dir() -> Path:
    """Where D38 says the assets live, resolved without importing production.

    Resolved here so the skip decision below does not depend on the module under
    test existing yet.
    """
    override = os.environ.get(ASSET_DIR_ENV)
    return Path(override) if override else ROOT / DEFAULT_ASSET_DIR


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing_assets() -> list[str]:
    """Which pinned assets are absent or wrong, as paths a reader can act on."""
    directory = real_asset_dir()
    problems = []
    for name, expected in ((ONNX_FILENAME, ONNX_SHA256), (TOKENIZER_FILENAME, TOKENIZER_SHA256)):
        path = directory / name
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif digest(path) != expected:
            problems.append(f"{path} does not match the pinned digest {expected}")
    return problems


def require_real_assets() -> Path:
    """Skip unless the pinned assets are present -- or fail, if they are required.

    This is D38's CI rule in executable form: an environment that is supposed to
    hold the assets cannot go green without them.
    """
    problems = missing_assets()
    if not problems:
        return real_asset_dir()
    detail = "; ".join(problems)
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{REQUIRE_ENV}=1 but the pinned MiniLM assets are unusable: {detail}")
    pytest.skip(f"pinned MiniLM assets unavailable: {detail}")


@pytest.fixture
def assets() -> Path:
    return require_real_assets()


@pytest.fixture
def fake_assets(tmp_path, monkeypatch) -> Path:
    """A directory shaped like the real one, with deliberately wrong bytes.

    Enough to exercise every refusal path without the 90MB export.
    """
    directory = tmp_path / "minilm"
    directory.mkdir()
    (directory / ONNX_FILENAME).write_bytes(b"not an onnx graph")
    (directory / TOKENIZER_FILENAME).write_text('{"not": "a tokenizer"}', encoding="utf-8")
    monkeypatch.setenv(ASSET_DIR_ENV, str(directory))
    return directory


# --- the frozen identity ------------------------------------------------------------


def test_the_checkpoint_is_the_one_d38_pinned():
    assert minilm().CHECKPOINT == CHECKPOINT


def test_the_revision_is_pinned():
    """Floating `main` would let the bytes change under a published number."""
    assert minilm().REVISION == REVISION


def test_the_export_is_the_standard_one():
    module = minilm()
    assert module.ONNX_FILENAME == ONNX_FILENAME
    source = Path(module.__file__).read_text(encoding="utf-8")
    for excluded in EXCLUDED_EXPORTS:
        assert excluded not in source, f"D38 excludes {excluded}"


def test_the_onnx_digest_is_the_frozen_value():
    assert minilm().ONNX_SHA256 == ONNX_SHA256


def test_the_tokenizer_digest_is_the_frozen_value():
    assert minilm().TOKENIZER_SHA256 == TOKENIZER_SHA256


def test_the_two_digests_are_distinct_constants():
    """The pair is the identity; one constant reused for both would verify half of it."""
    assert minilm().ONNX_SHA256 != minilm().TOKENIZER_SHA256


def test_the_model_version_is_the_frozen_string():
    assert minilm().MODEL_VERSION == MODEL_VERSION


def test_the_embedding_model_id_is_the_checkpoint():
    assert minilm().EMBEDDING_MODEL_ID == CHECKPOINT


# --- the three sequence limits ------------------------------------------------------


def test_sentinel_truncates_at_two_hundred_and_fifty_six():
    assert minilm().MAX_SEQUENCE_TOKENS == SENTINEL_MAX_TOKENS


def test_the_packaged_tokenizer_default_is_recorded_separately():
    """D38 names 128 as the tokenizer file's own default, not as Sentinel's limit."""
    assert minilm().PACKAGED_TOKENIZER_MAX == PACKAGED_TOKENIZER_MAX


def test_the_graph_positional_limit_is_recorded_separately():
    assert minilm().GRAPH_POSITION_LIMIT == GRAPH_POSITION_LIMIT


def test_the_three_limits_are_not_conflated():
    """The failure this decision exists to prevent: silently obeying the packaged 128."""
    module = minilm()
    assert module.MAX_SEQUENCE_TOKENS != module.PACKAGED_TOKENIZER_MAX
    assert module.MAX_SEQUENCE_TOKENS != module.GRAPH_POSITION_LIMIT
    assert module.PACKAGED_TOKENIZER_MAX < module.MAX_SEQUENCE_TOKENS < module.GRAPH_POSITION_LIMIT


# --- where the assets live ----------------------------------------------------------


def test_the_default_asset_directory_is_the_frozen_path(monkeypatch):
    monkeypatch.delenv(ASSET_DIR_ENV, raising=False)
    resolved = Path(minilm().asset_dir())
    assert resolved.as_posix().endswith(DEFAULT_ASSET_DIR.as_posix()), resolved


def test_the_environment_variable_relocates_the_assets(tmp_path, monkeypatch):
    monkeypatch.setenv(ASSET_DIR_ENV, str(tmp_path / "elsewhere"))
    assert Path(minilm().asset_dir()) == tmp_path / "elsewhere"


# --- refusals -----------------------------------------------------------------------


def test_a_missing_onnx_file_raises_model_asset_unavailable(fake_assets):
    module = minilm()
    (fake_assets / ONNX_FILENAME).unlink()
    with pytest.raises(module.ModelAssetUnavailable) as raised:
        module.load_minilm()
    assert ONNX_FILENAME in str(raised.value), "the refusal must name the missing file"


def test_a_missing_tokenizer_raises_model_asset_unavailable(fake_assets):
    module = minilm()
    (fake_assets / TOKENIZER_FILENAME).unlink()
    with pytest.raises(module.ModelAssetUnavailable) as raised:
        module.load_minilm()
    assert TOKENIZER_FILENAME in str(raised.value)


def test_a_wrong_onnx_digest_raises_model_asset_mismatch(fake_assets, monkeypatch):
    """The tokenizer is made to pass so the ONNX digest is the only thing refused."""
    module = minilm()
    monkeypatch.setattr(
        module, "TOKENIZER_SHA256", digest(fake_assets / TOKENIZER_FILENAME), raising=True
    )
    with pytest.raises(module.ModelAssetMismatch) as raised:
        module.load_minilm()
    assert ONNX_SHA256 in str(raised.value), "the refusal must name the digest it expected"


def test_a_wrong_tokenizer_digest_raises_model_asset_mismatch(fake_assets, monkeypatch):
    module = minilm()
    monkeypatch.setattr(module, "ONNX_SHA256", digest(fake_assets / ONNX_FILENAME), raising=True)
    with pytest.raises(module.ModelAssetMismatch) as raised:
        module.load_minilm()
    assert TOKENIZER_SHA256 in str(raised.value)


def test_a_digest_is_never_adopted_from_whatever_file_is_found(fake_assets):
    """D38 prohibits a first-observed bootstrap: it would verify nothing."""
    module = minilm()
    with pytest.raises(module.ModelAssetMismatch):
        module.load_minilm()
    assert module.ONNX_SHA256 == ONNX_SHA256, "the expected digest was rewritten at run time"
    assert module.TOKENIZER_SHA256 == TOKENIZER_SHA256


def test_the_two_refusals_are_distinct_types():
    """Absent and corrupt are different operator problems and must be distinguishable."""
    module = minilm()
    assert module.ModelAssetUnavailable is not module.ModelAssetMismatch
    assert not issubclass(module.ModelAssetMismatch, module.ModelAssetUnavailable)
    assert not issubclass(module.ModelAssetUnavailable, module.ModelAssetMismatch)


# --- the asset gate itself ----------------------------------------------------------


def test_the_require_flag_turns_a_missing_asset_into_a_failure(tmp_path, monkeypatch):
    """D38's CI rule, tested on the gate rather than on a hypothetical CI run."""
    monkeypatch.setenv(ASSET_DIR_ENV, str(tmp_path / "empty"))
    monkeypatch.setenv(REQUIRE_ENV, "1")
    # `Failed` and `Skipped` derive from BaseException, so the narrower classes are
    # named explicitly: `pytest.raises(Exception)` would let the outcome escape and
    # the test would report the very outcome it is trying to assert.
    with pytest.raises(pytest.fail.Exception) as raised:
        require_real_assets()
    assert REQUIRE_ENV in str(raised.value)


def test_without_the_flag_a_missing_asset_only_skips(tmp_path, monkeypatch):
    monkeypatch.setenv(ASSET_DIR_ENV, str(tmp_path / "empty"))
    monkeypatch.delenv(REQUIRE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception) as raised:
        require_real_assets()
    assert "is missing" in str(raised.value), "the skip must name the path it wanted"


# --- source-level guards ------------------------------------------------------------


def test_no_384_literal_appears_under_ml_embedders():
    """D18: 384 is an observation about one checkpoint, never a constant here."""
    offenders = []
    for path in sorted(EMBEDDERS_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value == 384 and node.value is not True:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert not offenders, f"hardcoded embedding width: {offenders}"


def test_ml_embedders_imports_no_django():
    """D38 puts `ml/embedders` under the Django-independence rule (plan §99)."""
    offenders = []
    for path in sorted(EMBEDDERS_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "django" or name.startswith("django."):
                    offenders.append(f"{path.relative_to(ROOT).as_posix()} imports {name}")
    assert not offenders, offenders


def test_loading_the_module_opens_no_socket(monkeypatch):
    """No test downloads a model, and importing must not reach the network either."""

    def refuse(*args, **kwargs):
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    importlib.reload(minilm())


# --- dependency and repository hygiene, as D38 requires them to end up -------------


def test_the_tokenizer_dependency_is_pinned_in_the_ml_tier():
    lines = [line.strip() for line in ML_REQUIREMENTS.read_text(encoding="utf-8").splitlines()]
    assert TOKENIZER_PIN in lines, f"{TOKENIZER_PIN} belongs in requirements/ml.txt"


def test_the_tokenizer_dependency_never_enters_base():
    """`base.txt` is what production installs; a tokenizer there is a budget regression."""
    text = BASE_REQUIREMENTS.read_text(encoding="utf-8")
    assert TOKENIZER not in text


def test_the_tokenizer_asset_is_excluded_from_git():
    """The weights are already ignored; `tokenizer.json` must be too (D38)."""
    patterns = [
        line.strip()
        for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any(TOKENIZER_FILENAME in pattern for pattern in patterns), (
        "no .gitignore rule excludes the MiniLM tokenizer asset"
    )


# --- behaviour, when the pinned assets are present ----------------------------------


def test_the_pinned_assets_hash_to_the_frozen_digests(assets):
    assert digest(assets / ONNX_FILENAME) == ONNX_SHA256
    assert digest(assets / TOKENIZER_FILENAME) == TOKENIZER_SHA256


def test_the_packaged_tokenizer_really_defaults_to_one_hundred_and_twenty_eight(assets):
    """Guards the guard: if the file did not default to 128 there is nothing to override."""
    packaged = json.loads((assets / TOKENIZER_FILENAME).read_text(encoding="utf-8"))
    assert packaged["truncation"]["max_length"] == PACKAGED_TOKENIZER_MAX


def test_sentinel_overrides_the_packaged_truncation(assets):
    """The failure D38 exists to prevent: obeying 128 while the contract says 256."""
    embedder = minilm().load_minilm()
    assert embedder.effective_max_tokens == SENTINEL_MAX_TOKENS


def test_the_graph_takes_the_three_int64_inputs(assets):
    inputs = {node.name: node.type for node in minilm().load_minilm().session.get_inputs()}
    assert set(inputs) == {"input_ids", "attention_mask", "token_type_ids"}
    assert set(inputs.values()) == {"tensor(int64)"}


def test_the_graph_emits_token_level_hidden_states(assets):
    """The export pools nothing; Sentinel pools outside it (D38)."""
    outputs = minilm().load_minilm().session.get_outputs()
    assert [node.name for node in outputs] == ["last_hidden_state"]
    assert len(outputs[0].shape) == 3


def test_token_type_ids_are_zero_for_single_segment_input(assets):
    encoded = minilm().load_minilm().encode(["a single segment of text"])
    assert np.asarray(encoded.token_type_ids).sum() == 0


def test_truncation_keeps_the_leading_tokens(assets):
    """Right truncation: the head survives, the tail is dropped."""
    embedder = minilm().load_minilm()
    short = embedder.encode(["alpha beta gamma"])
    long = embedder.encode(["alpha beta gamma " + "delta " * 4000])
    assert np.asarray(long.input_ids).shape[1] == SENTINEL_MAX_TOKENS
    head = np.asarray(long.input_ids)[0, :3]
    assert np.array_equal(head, np.asarray(short.input_ids)[0, :3])


def test_the_truncated_input_count_is_observable(assets):
    embedder = minilm().load_minilm()
    embedder.embed(["short text", "long text " + "padding words " * 2000])
    assert embedder.truncated_input_count == 1


def test_padding_does_not_change_the_pooled_vector(assets):
    """Pooling is mask-weighted, so a batch's padding cannot move a vector."""
    embedder = minilm().load_minilm()
    alone = np.asarray(embedder.embed(["a short sentence"]))
    batched = np.asarray(
        embedder.embed(["a short sentence", "a considerably longer sentence " * 20])
    )
    assert np.allclose(alone[0], batched[0], atol=1e-5)


def test_every_vector_is_l2_normalised(assets):
    vectors = np.asarray(minilm().load_minilm().embed(["one text", "another text"]))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_the_output_is_dense_float32_in_input_order(assets):
    embedder = minilm().load_minilm()
    texts = ["first text", "second text", "third text"]
    vectors = embedder.embed(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.dtype == np.float32
    assert vectors.shape[0] == len(texts)
    reversed_rows = np.asarray(embedder.embed(list(reversed(texts))))
    assert np.allclose(np.asarray(vectors), reversed_rows[::-1], atol=1e-6)


def test_the_embedding_dimension_is_observed_from_the_output(assets):
    embedder = minilm().load_minilm()
    assert embedder.embedding_dimension == np.asarray(embedder.embed(["text"])).shape[1]


def test_the_same_inputs_reproduce_under_the_same_pinned_environment(assets):
    """D38's reproducibility conditions -- not a claim about other machines."""
    embedder = minilm().load_minilm()
    first = np.asarray(embedder.embed(["stable text", "second stable text"]))
    second = np.asarray(embedder.embed(["stable text", "second stable text"]))
    assert np.array_equal(first, second)


@pytest.mark.parametrize("bad", ["", "   ", None, 7])
def test_the_input_contract_holds_for_minilm_too(assets, bad):
    with pytest.raises(ValueError):
        minilm().load_minilm().embed(["valid text", bad])


def test_there_is_no_training_path(assets):
    """Inference only (D38): no fit, no fine-tune, no gradient entry point."""
    embedder = minilm().load_minilm()
    for forbidden in ("fit", "fit_transform", "train", "finetune", "fine_tune", "backward"):
        assert not hasattr(embedder, forbidden), f"{forbidden} is not an inference API"
