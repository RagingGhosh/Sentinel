import pytest
from django.db import IntegrityError

from complaints.models import Complaint, Status
from tests.factories import ComplaintFactory


@pytest.mark.django_db
def test_duplicate_status_requires_a_canonical_complaint():
    complaint = ComplaintFactory()
    complaint.status = Status.DUPLICATE
    with pytest.raises(IntegrityError):
        complaint.save()


@pytest.mark.django_db
def test_a_complaint_cannot_duplicate_itself():
    complaint = ComplaintFactory()
    complaint.status = Status.DUPLICATE
    complaint.duplicate_of = complaint
    with pytest.raises(IntegrityError):
        complaint.save()


@pytest.mark.django_db
def test_duplicate_with_a_canonical_is_valid():
    canonical = ComplaintFactory()
    duplicate = ComplaintFactory()
    duplicate.status = Status.DUPLICATE
    duplicate.duplicate_of = canonical
    duplicate.save()
    assert Complaint.objects.filter(status=Status.DUPLICATE).count() == 1


@pytest.mark.django_db
def test_new_complaints_start_submitted_and_untriaged():
    complaint = ComplaintFactory()
    assert complaint.status == Status.SUBMITTED
    assert complaint.triaged_at is None
    assert complaint.due_at is None
    assert complaint.category is None
