import pytest

from complaints.models import ImmutableRecordError, Prediction, PredictionKind
from tests.factories import ComplaintFactory


@pytest.mark.django_db
def test_prediction_records_the_model_version_that_produced_it():
    complaint = ComplaintFactory()
    prediction = Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.TRIAGE,
        payload={"category_slug": "mortgage", "confidence": 0.91},
        model_name="triage",
        model_version="v1",
    )
    assert prediction.model_version == "v1"


@pytest.mark.django_db
def test_predictions_cannot_be_updated():
    """Predictions are evidence. Rewriting one would falsify the audit trail."""
    complaint = ComplaintFactory()
    prediction = Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.TRIAGE,
        payload={"category_slug": "mortgage", "confidence": 0.91},
        model_name="triage",
        model_version="v1",
    )
    prediction.model_version = "v2"
    with pytest.raises(ImmutableRecordError):
        prediction.save()


@pytest.mark.django_db
def test_predictions_cannot_be_deleted():
    complaint = ComplaintFactory()
    prediction = Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.DEDUP,
        payload={"matches": []},
        model_name="dedup",
        model_version="v1",
    )
    with pytest.raises(ImmutableRecordError):
        prediction.delete()
