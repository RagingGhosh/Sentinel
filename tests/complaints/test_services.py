from datetime import timedelta

import pytest
from django.utils import timezone

from complaints import services
from complaints.models import ComplaintEvent, EventKind, Priority, Status
from tests.factories import CategoryFactory, ComplaintFactory, PredictionFactory, UserFactory


@pytest.mark.django_db
def test_legal_transition_writes_an_event():
    complaint = ComplaintFactory()
    actor = UserFactory()
    services.transition(complaint, Status.IN_REVIEW, actor)
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_REVIEW
    event = ComplaintEvent.objects.get(complaint=complaint, kind=EventKind.STATUS)
    assert (event.from_value, event.to_value) == (Status.SUBMITTED, Status.IN_REVIEW)
    assert event.actor == actor


@pytest.mark.django_db
def test_illegal_transition_is_refused():
    complaint = ComplaintFactory()
    with pytest.raises(services.InvalidTransition):
        services.transition(complaint, Status.CLOSED, UserFactory())


@pytest.mark.django_db
def test_illegal_transition_leaves_no_event_behind():
    """A refused transition must not half-apply."""
    complaint = ComplaintFactory()
    with pytest.raises(services.InvalidTransition):
        services.transition(complaint, Status.CLOSED, UserFactory())
    complaint.refresh_from_db()
    assert complaint.status == Status.SUBMITTED
    assert ComplaintEvent.objects.filter(complaint=complaint).count() == 0


@pytest.mark.django_db
def test_triage_sets_the_sla_clock_from_human_confirmation():
    """due_at derives from triaged_at, which is the human decision moment."""
    category = CategoryFactory(sla_hours=48)
    complaint = ComplaintFactory(domain=category.domain)
    before = timezone.now()

    services.triage(complaint, category, Priority.HIGH, UserFactory())

    complaint.refresh_from_db()
    assert complaint.category == category
    assert complaint.priority == Priority.HIGH
    assert complaint.status == Status.IN_REVIEW
    assert complaint.triaged_at >= before
    assert complaint.due_at == complaint.triaged_at + timedelta(hours=48)


@pytest.mark.django_db
def test_triage_links_the_prediction_it_accepted():
    category = CategoryFactory(slug="mortgage", sla_hours=72)
    complaint = ComplaintFactory(domain=category.domain)
    prediction = PredictionFactory(
        complaint=complaint, payload={"category_slug": "mortgage", "confidence": 0.9}
    )

    services.triage(complaint, category, Priority.LOW, UserFactory(), prediction=prediction)

    event = ComplaintEvent.objects.get(complaint=complaint, kind=EventKind.CATEGORY)
    assert event.was_prediction_accepted is True


@pytest.mark.django_db
def test_triage_records_an_override_when_the_human_disagrees():
    category = CategoryFactory(slug="credit_card", sla_hours=72)
    complaint = ComplaintFactory(domain=category.domain)
    prediction = PredictionFactory(
        complaint=complaint, payload={"category_slug": "mortgage", "confidence": 0.9}
    )

    services.triage(complaint, category, Priority.LOW, UserFactory(), prediction=prediction)

    event = ComplaintEvent.objects.get(complaint=complaint, kind=EventKind.CATEGORY)
    assert event.was_prediction_accepted is False


@pytest.mark.django_db
def test_triage_rejects_a_category_from_another_domain():
    """Cross-domain categories would corrupt every per-domain metric."""
    complaint = ComplaintFactory()
    foreign_category = CategoryFactory()
    with pytest.raises(services.InvalidTransition):
        services.triage(complaint, foreign_category, Priority.LOW, UserFactory())


@pytest.mark.django_db
def test_resolve_stamps_resolved_at():
    category = CategoryFactory(sla_hours=24)
    complaint = ComplaintFactory(domain=category.domain)
    actor = UserFactory()
    services.triage(complaint, category, Priority.LOW, actor)
    services.transition(complaint, Status.IN_PROGRESS, actor)
    services.resolve(complaint, actor)
    complaint.refresh_from_db()
    assert complaint.status == Status.RESOLVED
    assert complaint.resolved_at is not None
