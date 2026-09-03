import pytest

from complaints.models import ComplaintEvent, EventKind, PredictionKind
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


@pytest.mark.django_db
def test_dedup_decision_matching_a_suggested_complaint_counts_as_acceptance():
    complaint = ComplaintFactory()
    canonical = ComplaintFactory()
    prediction = PredictionFactory(
        complaint=complaint,
        kind=PredictionKind.DEDUP,
        payload={"matches": [{"complaint_id": canonical.id, "similarity": 0.95}]},
    )
    event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.DUPLICATE,
        to_value=str(canonical.id),
        actor=UserFactory(),
        prediction=prediction,
    )
    assert event.was_prediction_accepted is True


@pytest.mark.django_db
def test_dedup_decision_naming_an_unsuggested_complaint_counts_as_override():
    complaint = ComplaintFactory()
    canonical = ComplaintFactory()
    other = ComplaintFactory()
    prediction = PredictionFactory(
        complaint=complaint,
        kind=PredictionKind.DEDUP,
        payload={"matches": [{"complaint_id": canonical.id, "similarity": 0.95}]},
    )
    event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.DUPLICATE,
        to_value=str(other.id),
        actor=UserFactory(),
        prediction=prediction,
    )
    assert event.was_prediction_accepted is False


@pytest.mark.django_db
def test_risk_prediction_has_no_acceptance_verdict():
    """A risk score is not something a human accepts or overrides."""
    complaint = ComplaintFactory()
    prediction = PredictionFactory(
        complaint=complaint,
        kind=PredictionKind.RISK,
        payload={"band": "high", "score": 0.8},
    )
    event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.PRIORITY,
        to_value="high",
        actor=UserFactory(),
        prediction=prediction,
    )
    assert event.was_prediction_accepted is None
