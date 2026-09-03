import pytest

from domains.packs import PACKS, UnknownDomainError, get_pack


def test_registered_pack_resolves_by_slug():
    pack = get_pack("cfpb")
    assert pack.slug == "cfpb"


def test_unknown_slug_raises_a_named_error():
    with pytest.raises(UnknownDomainError) as exc:
        get_pack("does-not-exist")
    assert "does-not-exist" in str(exc.value)


def test_every_pack_key_matches_its_slug():
    """A mismatch here would make get_pack return the wrong pack silently."""
    for key, pack in PACKS.items():
        assert key == pack.slug


def test_packs_satisfy_the_protocol():
    for pack in PACKS.values():
        assert isinstance(pack.slug, str)
        assert isinstance(pack.display_name, str)
