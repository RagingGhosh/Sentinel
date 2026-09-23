"""The MiniLM arm of the duplicate-retrieval benchmark (D38, addendum §5.3).

The export Sentinel runs is the transformer body only: it emits token-level
hidden states and performs neither pooling nor normalisation. Both are applied
here, mask-weighted then L2, which reproduces the checkpoint's own published
representation rather than inventing one.

Three sequence limits exist and are deliberately kept apart. The packaged
tokenizer file truncates and pads at `PACKAGED_TOKENIZER_MAX`; Sentinel
truncates at `MAX_SEQUENCE_TOKENS`, which this module sets explicitly because
loading the file as shipped would quietly shorten every input; and the graph
itself refuses anything past `GRAPH_POSITION_LIMIT`. Only the middle one is
Sentinel's choice.

The weights and the tokenizer are external assets, pinned by digest and
verified before any inference. Nothing here downloads: the tokenizer is read
from a local file, and no hub client is imported or reached.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
"""D38's checkpoint. It fixes which model Sentinel benchmarks and says nothing
about its output width, which stays an observation (D18)."""

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
"""The repository revision both assets came from. A floating branch would let
the bytes behind a published number change."""

ONNX_FILENAME = "model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

ONNX_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
"""The pair of digests is the model identity: either file swapped alone would
produce silently wrong vectors, so both are verified on every load."""

MODEL_VERSION = "all_minilm_l6_v2_onnx_v1"
EMBEDDING_MODEL_ID = CHECKPOINT

ASSET_SUBPATH = Path("ml") / "artifacts" / "embedders" / "all_minilm_l6_v2" / "v1"
ASSET_DIR_ENV = "SENTINEL_MINILM_DIR"

MAX_SEQUENCE_TOKENS = 256
"""Sentinel's benchmark limit, and only that: it is neither the packaged
tokenizer's default nor the graph's maximum."""

PACKAGED_TOKENIZER_MAX = 128
"""What the checkpoint's own tokenizer.json truncates and pads to. Recorded so
the override below is visibly an override."""

GRAPH_POSITION_LIMIT = 512
"""The export's positional capacity, which it enforces by failing rather than by
truncating. Recorded, never relied on."""

TRUNCATION_SIDE = "right"

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"
HIDDEN_STATE = "last_hidden_state"


class ModelAssetError(RuntimeError):
    """An external model asset cannot be trusted, so nothing is inferred from it."""


class ModelAssetUnavailable(ModelAssetError):
    """A pinned asset is not where it should be.

    Distinct from a mismatch on purpose: an absent file and a corrupt one are
    different operator problems with different fixes.
    """


class ModelAssetMismatch(ModelAssetError):
    """A pinned asset is present but is not the bytes this decision names."""


class EmbeddingDimensionMismatch(ValueError):
    """An embedding's width is not the width something was built with (D18).

    Declared here because it belongs to the embedding contract; the retrieval
    index is what raises it, comparing against the width it recorded.
    """


@dataclass(frozen=True)
class Encoded:
    """One tokenized batch, rectangular and ready for the graph."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray
    truncated: int


def asset_dir() -> Path:
    """Where the pinned assets live.

    `SENTINEL_MINILM_DIR` relocates them -- for a shared cache, or a machine that
    keeps large binaries elsewhere -- and is the only environment variable this
    module reads.
    """
    override = os.environ.get(ASSET_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / ASSET_SUBPATH


class MiniLMEmbedder:
    """A loaded `TextEmbedder` over the pinned export.

    Constructed only by `load_minilm`, after both digests have passed. Inference
    only: there is no fitting, no gradient and no training mode to reach.
    """

    def __init__(self, session: Any, tokenizer: Any, embedding_dimension: int) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.embedding_dimension = embedding_dimension
        self.model_version = MODEL_VERSION
        self.embedding_model_id = EMBEDDING_MODEL_ID
        self.effective_max_tokens = MAX_SEQUENCE_TOKENS
        self.truncated_input_count = 0

    def encode(self, texts: Sequence[str]) -> Encoded:
        """Tokenize a batch under Sentinel's limit, padded to its longest member.

        Padding makes the tensor rectangular, which the graph requires; it cannot
        move a vector, because pooling below is weighted by the attention mask.
        """
        values = _validated(texts)
        encodings = self.tokenizer.encode_batch(values)
        # Counted from the attention mask, not from `len(ids)`: padding makes every
        # row in a batch as long as its longest member, so a padded short text
        # would otherwise be reported as truncated.
        truncated = sum(
            1 for encoding in encodings if sum(encoding.attention_mask) >= MAX_SEQUENCE_TOKENS
        )
        self.truncated_input_count += truncated
        return Encoded(
            input_ids=np.asarray([e.ids for e in encodings], dtype=np.int64),
            attention_mask=np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
            token_type_ids=np.asarray([e.type_ids for e in encodings], dtype=np.int64),
            truncated=truncated,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Dense, `float32`, L2-normalised, one row per input, in input order."""
        encoded = self.encode(texts)
        if encoded.input_ids.shape[0] == 0:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)
        hidden = self._hidden_states(encoded)
        return _pooled(hidden, encoded.attention_mask)

    def _hidden_states(self, encoded: Encoded) -> np.ndarray:
        """The graph's own output: token-level states, pooled by nobody but us."""
        feed = {
            INPUT_IDS: encoded.input_ids,
            ATTENTION_MASK: encoded.attention_mask,
            TOKEN_TYPE_IDS: encoded.token_type_ids,
        }
        outputs = self.session.run([HIDDEN_STATE], feed)
        return np.asarray(outputs[0], dtype=np.float32)


def load_minilm(directory: str | os.PathLike[str] | None = None) -> MiniLMEmbedder:
    """Verify both assets, then load them.

    Digest verification happens before the graph is opened, so unknown bytes are
    never executed. The expected digests come from this module's constants and
    are never taken from the files found -- a first-observed bootstrap would
    verify nothing.
    """
    root = Path(directory) if directory is not None else asset_dir()
    onnx_path = root / ONNX_FILENAME
    tokenizer_path = root / TOKENIZER_FILENAME
    # Presence for both before digests for either: an absent asset and a corrupt
    # one are different operator problems, and reporting "wrong bytes" for the
    # file that is there would hide the file that is not.
    _require_present(onnx_path)
    _require_present(tokenizer_path)
    _require_digest(onnx_path, ONNX_SHA256)
    _require_digest(tokenizer_path, TOKENIZER_SHA256)

    options = ort.SessionOptions()
    # One thread each way: within a single pinned environment this is what makes
    # repeated runs reproduce. No claim is made about other machines.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    packaged = tokenizer.padding or {}
    # The packaged file truncates and pads at its own length. Overriding both is
    # the point: loaded as shipped it would silently shorten every input, and the
    # published limit would not be the one in force.
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_TOKENS, direction=TRUNCATION_SIDE)
    tokenizer.enable_padding(
        direction=TRUNCATION_SIDE,
        pad_id=packaged.get("pad_id", 0),
        pad_type_id=packaged.get("pad_type_id", 0),
        pad_token=packaged.get("pad_token", "[PAD]"),
    )

    embedder = MiniLMEmbedder(session, tokenizer, embedding_dimension=0)
    embedder.embedding_dimension = _observed_width(embedder)
    embedder.truncated_input_count = 0
    return embedder


def _observed_width(embedder: MiniLMEmbedder) -> int:
    """The width the model actually produced, measured once at load (D18).

    Read from a real forward pass rather than from the graph's declared shape or
    from a constant, so an export whose pooling differs is described correctly
    instead of being assumed.
    """
    encoded = embedder.encode(["dimension probe"])
    return int(embedder._hidden_states(encoded).shape[-1])


def _pooled(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Attention-mask weighted mean, then L2 -- the checkpoint's representation.

    Weighting by the mask is what makes padding inert: a padded position
    contributes zero to both the sum and the divisor, so a text embedded alone
    and the same text embedded beside a longer one give the same vector.
    """
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (hidden * mask).sum(axis=1)
    # Every encoding carries at least its special tokens, so the divisor is never
    # zero; the floor keeps a degenerate mask from producing infinities.
    counts = np.maximum(mask.sum(axis=1), 1.0)
    means = summed / counts
    norms = np.maximum(np.linalg.norm(means, axis=1, keepdims=True), np.finfo(np.float32).tiny)
    return np.asarray(means / norms, dtype=np.float32)


def _require_present(path: Path) -> None:
    """Refuse an asset that is not there, naming the path an operator must fill."""
    if not path.is_file():
        raise ModelAssetUnavailable(
            f"the pinned MiniLM asset {path.name} is missing from {path.parent}; "
            f"obtain it from {CHECKPOINT} at revision {REVISION} "
            f"(or point {ASSET_DIR_ENV} at a directory holding it)"
        )


def _require_digest(path: Path, expected: str) -> None:
    """Refuse an asset present as bytes this decision did not name."""
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ModelAssetMismatch(
            f"{path} does not match the pinned digest: expected {expected}, "
            f"observed {observed}. The weights and the tokenizer are pinned "
            "together, and neither digest is ever taken from the file found"
        )


def _validated(texts: Sequence[str]) -> list[str]:
    """Every element checked before any tokenization, so a refusal infers nothing."""
    if isinstance(texts, str):
        raise ValueError("texts must be a sequence of strings, not a single string")
    values = list(texts)
    for position, text in enumerate(values):
        if not isinstance(text, str):
            raise ValueError(
                f"texts[{position}] is {type(text).__name__}, not a string; "
                "the embedder accepts strings only"
            )
        if not text.strip():
            raise ValueError(
                f"texts[{position}] is empty or whitespace only; an all-zero row "
                "has no defined cosine similarity"
            )
    return values
