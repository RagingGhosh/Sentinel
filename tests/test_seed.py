import pytest
from django.core.management import call_command

from complaints.models import Complaint
from domains.models import Domain
from domains.packs import PACKS


@pytest.mark.django_db
def test_seed_creates_a_domain_row_for_every_registered_pack():
    call_command("seed_demo")
    assert set(Domain.objects.values_list("slug", flat=True)) == set(PACKS)


@pytest.mark.django_db
def test_seed_creates_complaints():
    call_command("seed_demo")
    assert Complaint.objects.exists()


@pytest.mark.django_db
def test_seed_is_idempotent():
    """The deploy runs this on every boot; it must not multiply rows."""
    call_command("seed_demo")
    first = Complaint.objects.count()
    call_command("seed_demo")
    assert Complaint.objects.count() == first


class _ThirdPack:
    """A stand-in for a pack registered after this test was written.

    Used only to prove the seed command cannot silently skip a pack's
    categories -- it must not need updating by name every time PACKS grows.
    """

    slug = "thirdpack"
    display_name = "Third Pack"
    demo_categories = (("widget", "Widget", 24),)


@pytest.mark.django_db
def test_seed_gives_every_registered_pack_its_demo_categories(monkeypatch):
    """A pack with categories left unseeded cannot be triaged -- this must never regress,
    including for a pack registered after this test was written."""
    monkeypatch.setitem(PACKS, _ThirdPack.slug, _ThirdPack)
    call_command("seed_demo")
    for slug, pack in PACKS.items():
        domain = Domain.objects.get(slug=slug)
        seeded_slugs = set(domain.categories.values_list("slug", flat=True))
        expected_slugs = {cat_slug for cat_slug, _, _ in pack.demo_categories}
        assert seeded_slugs == expected_slugs
