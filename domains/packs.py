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
    # (category slug, display name, sla_hours) per demo category. Consumed only by
    # seed_demo, which iterates PACKS generically rather than branching on slug.
    demo_categories: ClassVar[tuple[tuple[str, str, int], ...]]


class CFPBPack:
    slug: ClassVar[str] = "cfpb"
    display_name: ClassVar[str] = "Consumer Financial"
    demo_categories: ClassVar[tuple[tuple[str, str, int], ...]] = (
        ("mortgage", "Mortgage", 72),
        ("credit_card", "Credit card", 48),
        ("debt_collection", "Debt collection", 72),
        ("other", "Other", 96),
    )


class NYC311Pack:
    slug: ClassVar[str] = "nyc311"
    display_name: ClassVar[str] = "Civic Services"
    demo_categories: ClassVar[tuple[tuple[str, str, int], ...]] = (
        ("noise", "Noise", 24),
        ("street_condition", "Street condition", 120),
        ("sanitation", "Sanitation", 48),
        ("other", "Other", 96),
    )


PACKS: dict[str, type[DomainPack]] = {
    CFPBPack.slug: CFPBPack,
    NYC311Pack.slug: NYC311Pack,
}


def get_pack(slug: str) -> type[DomainPack]:
    try:
        return PACKS[slug]
    except KeyError as exc:
        raise UnknownDomainError(f"No domain pack registered for slug {slug!r}") from exc
