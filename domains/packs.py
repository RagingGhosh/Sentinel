"""Domain packs.

A domain's *identity* lives in the database as a Domain row. A domain's
*behavior* lives here in code. Nothing outside this module may branch on a
domain slug.

Phase 1 packs carry only identity. Phase 2 adds dataset adapters; Phase 3
adds model bundles.
"""

from typing import ClassVar, Protocol, runtime_checkable


class UnknownDomainError(KeyError):
    """Raised when a domain slug has no registered pack."""


@runtime_checkable
class DomainPack(Protocol):
    slug: ClassVar[str]
    display_name: ClassVar[str]


class CFPBPack:
    slug: ClassVar[str] = "cfpb"
    display_name: ClassVar[str] = "Consumer Financial"


class NYC311Pack:
    slug: ClassVar[str] = "nyc311"
    display_name: ClassVar[str] = "Civic Services"


PACKS: dict[str, type[DomainPack]] = {
    CFPBPack.slug: CFPBPack,
    NYC311Pack.slug: NYC311Pack,
}


def get_pack(slug: str) -> type[DomainPack]:
    try:
        return PACKS[slug]
    except KeyError as exc:
        raise UnknownDomainError(f"No domain pack registered for slug {slug!r}") from exc
