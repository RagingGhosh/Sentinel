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


@pytest.mark.django_db
def test_bulk_update_is_blocked():
    """QuerySet.update() compiles straight to SQL, bypassing instance save()."""
    complaint = ComplaintFactory()
    Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.TRIAGE,
        payload={"category_slug": "mortgage", "confidence": 0.91},
        model_name="triage",
        model_version="v1",
    )
    with pytest.raises(ImmutableRecordError):
        Prediction.objects.filter(complaint=complaint).update(model_version="v2")


@pytest.mark.django_db
def test_bulk_delete_is_blocked():
    """QuerySet.delete() compiles straight to SQL, bypassing instance delete()."""
    complaint = ComplaintFactory()
    Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.DEDUP,
        payload={"matches": []},
        model_name="dedup",
        model_version="v1",
    )
    with pytest.raises(ImmutableRecordError):
        Prediction.objects.filter(complaint=complaint).delete()


@pytest.mark.django_db
def test_deleting_the_complaint_still_cascades_to_its_predictions():
    """Django's cascade goes through Collector.delete_batch(), not QuerySet.delete(),
    so it must not be blocked by the bulk-delete guard above."""
    complaint = ComplaintFactory()
    Prediction.objects.create(
        complaint=complaint,
        kind=PredictionKind.TRIAGE,
        payload={"category_slug": "mortgage", "confidence": 0.91},
        model_name="triage",
        model_version="v1",
    )
    complaint.delete()
    assert Prediction.objects.filter(complaint_id=complaint.id).count() == 0
