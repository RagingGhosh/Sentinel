"""Task 18: the TF-IDF benchmark arm and the `TextEmbedder` protocol (D38).

RED phase. Neither `ml.embedders.tfidf` nor `ml.base.TextEmbedder` exists yet, so
every test that reaches production fails. The production module is imported late,
inside `tfidf()`, so the fixture and the source-level guards still run.

The API surface these tests pin, all of it implied by D38 and none of it existing
yet:

    ml.base.TextEmbedder                 protocol: embed, model_version,
                                         embedding_dimension, embedding_model_id
    ml.embedders.tfidf.TFIDF_CONFIG      the frozen vectoriser parameters
    ml.embedders.tfidf.MODEL_VERSION     "tfidf_char_wb_3_5_v1"
    ml.embedders.tfidf.EMBEDDING_MODEL_ID
    ml.embedders.tfidf.fit_tfidf(train_texts, config)

Nothing is downloaded and no socket is opened; the corpus here is a handful of
strings in memory.
"""

import ast
import importlib
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")
pytest.importorskip("sklearn", reason="scikit-learn lives in requirements/ml.txt")

pytestmark = pytest.mark.ml

ROOT = Path(__file__).resolve().parents[3]
BASE_SOURCE = ROOT / "ml" / "base.py"

#: D38's frozen benchmark arm. These are the values, not a restatement of them:
#: a test that read them from the production module would assert nothing.
FROZEN_PARAMETERS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "lowercase": True,
    "min_df": 2,
    "max_df": 1.0,
    "max_features": 2048,
    "norm": "l2",
    "sublinear_tf": False,
}
FROZEN_DTYPE = np.float32
MODEL_VERSION = "tfidf_char_wb_3_5_v1"


# --- the production modules, imported late ------------------------------------------


def tfidf():
    """The Task 18 TF-IDF arm."""
    return importlib.import_module("ml.embedders.tfidf")


def base():
    """`ml.base`, which D38 extends with the `TextEmbedder` protocol only."""
    return importlib.import_module("ml.base")


# --- the fixture texts --------------------------------------------------------------
#
# Three periods with disjoint sentinel substrings. `char_wb` works on character
# n-grams inside word boundaries, so a sentinel word contributes n-grams that no
# other text can produce -- which is what makes "fitted on train text only"
# observable rather than merely asserted.

TRAIN_TEXTS = (
    "billing statement dispute filed against the card issuer",
    "billing statement charge reversed by the card issuer",
    "mortgage escrow shortage notice from the loan servicer",
    "mortgage escrow analysis dispute with the loan servicer",
    "credit report inaccurate tradeline reported by the bureau",
    "credit report mixed file dispute opened with the bureau",
)
VALIDATION_SENTINEL = "vvvalidationonlyvvv"
TEST_SENTINEL = "qqqtestonlyqqq"
VALIDATION_TEXTS = (
    f"billing statement dispute {VALIDATION_SENTINEL} filed late",
    f"mortgage escrow {VALIDATION_SENTINEL} notice received",
)
TEST_TEXTS = (
    f"credit report dispute {TEST_SENTINEL} opened today",
    f"billing statement {TEST_SENTINEL} charge disputed",
)


@pytest.fixture
def embedder():
    return tfidf().fit_tfidf(list(TRAIN_TEXTS), tfidf().TFIDF_CONFIG)


def vectors(embedder, texts):
    return np.asarray(embedder.embed(list(texts)))


# --- the protocol -------------------------------------------------------------------


def test_ml_base_declares_the_text_embedder_protocol():
    """D38 adds `TextEmbedder` to `ml/base.py`; plan Task 18 allows nothing else there."""
    assert hasattr(base(), "TextEmbedder"), "ml.base.TextEmbedder is missing"


def test_the_protocol_declares_exactly_the_four_frozen_members():
    """D38 fixes the surface. A fifth member is a contract change, not a convenience."""
    protocol = base().TextEmbedder
    declared = {
        name for name in vars(protocol) if not name.startswith("_") and name not in {"embed"}
    }
    annotations = set(getattr(protocol, "__annotations__", {}))
    members = declared | annotations | {"embed"}
    assert members == {
        "embed",
        "model_version",
        "embedding_dimension",
        "embedding_model_id",
    }, members


def test_the_protocol_has_no_fit_method():
    """Each arm is constructed already fitted, so an unfitted embedder cannot exist."""
    assert not hasattr(base().TextEmbedder, "fit")
    assert not hasattr(base().TextEmbedder, "fit_transform")


def test_ml_base_never_imports_numpy_at_runtime():
    """The module Django imports at startup stays free of heavy imports (D38).

    Checked by parsing rather than importing: the verdict must be the same whether
    or not numpy happens to be installed in the environment running the test.
    """
    tree = ast.parse(BASE_SOURCE.read_text(encoding="utf-8"))
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            name = getattr(test, "id", None) or getattr(test, "attr", None)
            if name == "TYPE_CHECKING":
                guarded.update(id(child) for child in ast.walk(node) if child is not node)

    runtime_numpy = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        and id(node) not in guarded
        and any(
            (alias.name if isinstance(node, ast.Import) else node.module or "").startswith("numpy")
            for alias in node.names
        )
    ]
    assert not runtime_numpy, "ml/base.py imports numpy outside a TYPE_CHECKING guard"


def test_the_fitted_embedder_satisfies_the_protocol(embedder):
    for member in ("embed", "model_version", "embedding_dimension", "embedding_model_id"):
        assert hasattr(embedder, member), f"the TF-IDF arm is missing {member}"


def test_the_fitted_embedder_exposes_no_fit_seam_of_its_own(embedder):
    """`fit_tfidf` is the only construction path; a public `fit` would reopen it."""
    assert not hasattr(embedder, "fit")


# --- the frozen configuration -------------------------------------------------------


def test_the_configuration_is_exactly_the_one_d38_froze():
    config = tfidf().TFIDF_CONFIG
    for name, expected in FROZEN_PARAMETERS.items():
        assert getattr(config, name) == expected, f"{name} is not D38's value"
    assert config.dtype is FROZEN_DTYPE


def test_the_fitted_vectoriser_carries_those_parameters(embedder):
    """The configuration is not decoration: the fitted object must actually use it."""
    params = embedder.vectorizer.get_params()
    for name, expected in FROZEN_PARAMETERS.items():
        assert params[name] == expected, f"the fitted vectoriser's {name} is not D38's value"
    assert params["dtype"] is FROZEN_DTYPE


def test_the_model_version_is_the_frozen_string():
    assert tfidf().MODEL_VERSION == MODEL_VERSION


def test_the_embedding_model_id_is_the_frozen_string():
    """D38's provenance table: TF-IDF's id is its recipe, there being no model file."""
    assert tfidf().EMBEDDING_MODEL_ID == MODEL_VERSION


def test_the_embedder_reports_those_strings(embedder):
    assert embedder.model_version == MODEL_VERSION
    assert embedder.embedding_model_id == MODEL_VERSION


def test_task_sixteen_s_triage_recipe_is_not_inherited():
    """D38: a representation chosen for a classifier is not one for retrieval.

    Task 16's word(1,2)+char(3,5) pair would be a silent inheritance, so both the
    parameters and the import are checked.
    """
    config = tfidf().TFIDF_CONFIG
    assert config.analyzer != "word"
    assert (config.analyzer, config.ngram_range) != ("char", (3, 5))
    source = Path(tfidf().__file__).read_text(encoding="utf-8")
    assert "triage" not in source, "the benchmark arm must not reach into Task 16"
    assert "TRIAGE_TFIDF_V1" not in source


# --- the output contract ------------------------------------------------------------


def test_the_output_is_a_dense_ndarray(embedder):
    matrix = embedder.embed(list(TRAIN_TEXTS))
    assert isinstance(matrix, np.ndarray), f"expected a dense ndarray, got {type(matrix)}"
    assert not hasattr(matrix, "toarray"), "a sparse matrix is not the frozen return type"


def test_the_output_is_float32(embedder):
    assert vectors(embedder, TRAIN_TEXTS).dtype == np.float32


def test_there_is_one_row_per_input(embedder):
    assert vectors(embedder, TEST_TEXTS).shape[0] == len(TEST_TEXTS)


def test_input_order_is_preserved(embedder):
    """Row i is text i. A sorted or de-duplicated implementation breaks the join."""
    forward = vectors(embedder, TRAIN_TEXTS)
    reversed_rows = vectors(embedder, tuple(reversed(TRAIN_TEXTS)))
    assert np.allclose(forward, reversed_rows[::-1])


def test_every_row_is_l2_normalised(embedder):
    norms = np.linalg.norm(vectors(embedder, TRAIN_TEXTS), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), norms


def test_the_embedding_dimension_is_the_observed_output_width(embedder):
    """D18: read from actual output, never declared."""
    assert embedder.embedding_dimension == vectors(embedder, TRAIN_TEXTS).shape[1]


def test_the_dimension_never_exceeds_max_features(embedder):
    """`max_features` is load-bearing: it bounds the dense matrix D38 requires."""
    assert embedder.embedding_dimension <= FROZEN_PARAMETERS["max_features"]


def test_the_width_is_stable_across_calls(embedder):
    """A per-call width would make the index's recorded dimension meaningless."""
    assert vectors(embedder, TRAIN_TEXTS).shape[1] == vectors(embedder, TEST_TEXTS).shape[1]


# --- fitting on the training period alone (§6.2, vocabulary construction) -----------


def test_the_vocabulary_is_fitted_on_training_text_only():
    """A sentinel appearing only after the training period must not enter the vocabulary."""
    fitted = tfidf().fit_tfidf(list(TRAIN_TEXTS), tfidf().TFIDF_CONFIG)
    vocabulary = set(fitted.vectorizer.vocabulary_)
    leaked = [term for term in vocabulary if "validationonly" in term or "testonly" in term]
    assert not leaked, f"future text reached the fitted vocabulary: {leaked[:5]}"


def test_fitting_on_every_period_would_produce_a_different_vocabulary():
    """Guards the guard: the sentinels must be detectable, or the test above is vacuous."""
    module = tfidf()
    train_only = module.fit_tfidf(list(TRAIN_TEXTS), module.TFIDF_CONFIG)
    everything = module.fit_tfidf(
        list(TRAIN_TEXTS + VALIDATION_TEXTS + TEST_TEXTS), module.TFIDF_CONFIG
    )
    assert set(train_only.vectorizer.vocabulary_) != set(everything.vectorizer.vocabulary_)


def test_embedding_later_text_does_not_extend_the_fitted_vocabulary(embedder):
    """Transform must not refit: the width after unseen text is the fitted width."""
    before = dict(embedder.vectorizer.vocabulary_)
    embedder.embed(list(TEST_TEXTS))
    assert dict(embedder.vectorizer.vocabulary_) == before


def test_unseen_text_still_embeds_without_error(embedder):
    """Evaluation text is embedded through the training vocabulary, not rejected."""
    assert vectors(embedder, TEST_TEXTS).shape == (len(TEST_TEXTS), embedder.embedding_dimension)


def test_the_same_inputs_give_the_same_vectors(embedder):
    """Determinism under one fitted embedder (D38's reproducibility conditions)."""
    assert np.array_equal(vectors(embedder, TEST_TEXTS), vectors(embedder, TEST_TEXTS))


def test_two_fits_on_the_same_training_text_agree():
    module = tfidf()
    first = module.fit_tfidf(list(TRAIN_TEXTS), module.TFIDF_CONFIG)
    second = module.fit_tfidf(list(TRAIN_TEXTS), module.TFIDF_CONFIG)
    assert first.embedding_dimension == second.embedding_dimension
    assert np.array_equal(
        np.asarray(first.embed(list(TEST_TEXTS))), np.asarray(second.embed(list(TEST_TEXTS)))
    )


# --- the input contract -------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", " ", "\t", "\n  \n"])
def test_empty_or_whitespace_text_is_refused(embedder, bad):
    """D38: non-empty strings only. An all-zero row has no defined cosine."""
    with pytest.raises(ValueError):
        embedder.embed([*TRAIN_TEXTS[:1], bad])


@pytest.mark.parametrize("bad", [None, 3, b"bytes", ["nested"]])
def test_a_non_string_element_is_refused(embedder, bad):
    with pytest.raises(ValueError):
        embedder.embed([*TRAIN_TEXTS[:1], bad])


def test_a_refused_batch_embeds_nothing(embedder):
    """Validation happens before work, so a refusal leaves no partial result."""
    with pytest.raises(ValueError):
        embedder.embed(["valid text here", ""])
    assert vectors(embedder, TRAIN_TEXTS[:1]).shape[0] == 1
