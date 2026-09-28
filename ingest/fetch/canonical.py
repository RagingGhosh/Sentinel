"""The byte forms D46 freezes, and nothing else.

An acquisition is identified by the SHA256 of its record's bytes, so those bytes
cannot be left to whatever a serializer happens to emit. The rules, from D46:

* values are objects, arrays, strings, integers, booleans and null only -- no
  floating-point number, NaN or infinity;
* keys sorted by Unicode code point at every level, no whitespace between tokens,
  non-ASCII characters written as UTF-8 rather than escaped, strings kept exactly;
* UTF-8 with no byte-order mark, followed by exactly one line feed;
* timestamps are UTC strings with exactly six fractional digits;
* digests are 64 lowercase hexadecimal characters.

`page_digest` is the page identity ``ingest.cli.page_checksum`` computes. It is
restated here rather than imported because this package may not import
``ingest.cli`` (D44); a test in ``tests/ingest/test_cli.py`` holds the two equal.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

DIGEST = re.compile(r"[0-9a-f]{64}")
"""A SHA256 digest as D46 writes it: 64 lowercase hexadecimal characters."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _refuse_floats(value: Any, where: str = "record") -> None:
    """D46 admits no floating-point value anywhere in a record or journal line."""
    if isinstance(value, float):
        raise ValueError(f"{where} holds a floating-point value ({value!r}); D46 forbids them")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} has a non-string key {key!r}")
            _refuse_floats(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _refuse_floats(item, f"{where}[{index}]")
    elif not (value is None or isinstance(value, (str, int, bool))):
        raise ValueError(f"{where} holds {type(value).__name__}, which JSON cannot carry")


def canonical_json(value: Any) -> str:
    """The frozen JSON text, without the final line feed."""
    _refuse_floats(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_bytes(value: Any) -> bytes:
    """The frozen bytes of a record or a journal line: canonical JSON, UTF-8, one LF."""
    return canonical_json(value).encode("utf-8") + b"\n"


def page_digest(page: Any) -> str:
    """A page's identity, byte for byte the rule ``ingest.cli.page_checksum`` applies."""
    text = json.dumps(page, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_hex(text.encode("utf-8"))


def format_timestamp(moment: datetime) -> str:
    """UTC, ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00``, always six fractional digits.

    A naive value is refused rather than assumed to be UTC, for the reason the CFPB
    adapter refuses one: an invented offset moves every derived instant.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"timestamp {moment!r} carries no offset")
    return moment.astimezone(UTC).isoformat(timespec="microseconds")
