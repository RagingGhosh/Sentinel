import pytest

from complaints import services
from complaints.models import ComplaintEvent, EventKind, Status
from tests.factories import ComplaintFactory, UserFactory


@pytest.mark.django_db
def test_marking_a_duplicate_points_at_the_canonical():
    canonical = ComplaintFactory()
    duplicate = ComplaintFactory(domain=canonical.domain)
    services.mark_duplicate(duplicate, canonical, UserFactory())
    duplicate.refresh_from_db()
    assert duplicate.status == Status.DUPLICATE
    assert duplicate.duplicate_of == canonical


@pytest.mark.django_db
def test_marking_a_duplicate_writes_both_a_duplicate_and_a_status_event():
    canonical = ComplaintFactory()
    duplicate = ComplaintFactory(domain=canonical.domain)
    services.mark_duplicate(duplicate, canonical, UserFactory())
    kinds = set(ComplaintEvent.objects.filter(complaint=duplicate).values_list("kind", flat=True))
    assert kinds == {EventKind.DUPLICATE, EventKind.STATUS}


@pytest.mark.django_db
def test_a_complaint_cannot_be_its_own_duplicate():
    complaint = ComplaintFactory()
    with pytest.raises(services.InvalidTransition):
        services.mark_duplicate(complaint, complaint, UserFactory())


@pytest.mark.django_db
def test_cannot_point_at_a_complaint_that_is_itself_a_duplicate():
    """Prevents chains: C -> B -> A. Every duplicate names a real canonical."""
    canonical = ComplaintFactory()
    first = ComplaintFactory(domain=canonical.domain)
    second = ComplaintFactory(domain=canonical.domain)
    services.mark_duplicate(first, canonical, UserFactory())
    with pytest.raises(services.InvalidTransition):
        services.mark_duplicate(second, first, UserFactory())


@pytest.mark.django_db
def test_cannot_mark_a_duplicate_across_domains():
    canonical = ComplaintFactory()
    other_domain_complaint = ComplaintFactory()
    with pytest.raises(services.InvalidTransition):
        services.mark_duplicate(other_domain_complaint, canonical, UserFactory())
