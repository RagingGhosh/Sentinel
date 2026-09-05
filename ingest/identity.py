"""Corpus record identity.

A corpus record is identified by `(source, external_id)` and by nothing else.
The pair matters: the two sources number their records independently, so
`cfpb:1` and `nyc311:1` are different records that happen to share a digit.

The string form is `"<source>:<external_id>"`. Parsing splits on the **first**
colon only, because a source's own identifier may contain one.
"""

from dataclasses import dataclass

from ingest.schema import CorpusRecord

SEPARATOR = ":"


@dataclass(frozen=True)
class RecordRef:
    """A hashable reference to one corpus record.

    Frozen so it can key a dictionary or join an evaluation set. Deliberately
    carries no integer: evaluation output referring to corpus records must never
    be typed to a `Complaint` primary key.
    """

    source: str
    external_id: str

    def __str__(self) -> str:
        return f"{self.source}{SEPARATOR}{self.external_id}"


def make_ref(record: CorpusRecord) -> RecordRef:
    """The reference identifying `record`."""
    return RecordRef(source=record.source, external_id=record.external_id)


def parse_ref(value: str) -> RecordRef:
    """Inverse of `str(ref)`.

    Splits on the first separator only, so an external id containing a colon
    survives a round trip intact.
    """
    source, found, external_id = value.partition(SEPARATOR)
    if not found:
        raise ValueError(
            f"not a record reference: {value!r} names no source "
            f"(expected '<source>{SEPARATOR}<external_id>')"
        )
    return RecordRef(source=source, external_id=external_id)
