import pytest

from complaints.models import ComplaintEvent, EventKind
from tests.factories import ComplaintFactory, PredictionFactory, UserFactory


@pytest.mark.django_db
def test_event_without_a_prediction_has_no_acceptance_verdict():
    event = ComplaintEvent.objects.create(
        complaint=ComplaintFactory(),
        kind=EventKind.CATEGORY,
        from_value=None,
        to_value="mortgage",
        actor=UserFactory(),
    )
    assert event.was_prediction_accepted is None


@pytest.mark.django_db
def test_matching_decision_counts_as_acceptance():
    complaint = ComplaintFactory()
    prediction = PredictionFactory(
        complaint=complaint, payload={"category_slug": "mortgage", "confidence": 0.9}
    )
    event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.CATEGORY,
        to_value="mortgage",
        actor=UserFactory(),
        prediction=prediction,
    )
    assert event.was_prediction_accepted is True


@pytest.mark.django_db
def test_differing_decision_counts_as_override():
    """Overrides are the retraining signal, so they must be identifiable."""
    complaint = ComplaintFactory()
    prediction = PredictionFactory(
        complaint=complaint, payload={"category_slug": "mortgage", "confidence": 0.9}
    )
    event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.CATEGORY,
        to_value="credit_card",
        actor=UserFactory(),
        prediction=prediction,
    )
    assert event.was_prediction_accepted is False
