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
