"""The only place complaint state changes.

Every mutation writes a ComplaintEvent, and every mutation is atomic, so a
refused change leaves neither state nor audit trail behind.
"""

from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from complaints.models import (
    Complaint,
    ComplaintEvent,
    EventKind,
    Prediction,
    Priority,
    Status,
)
from domains.models import Category


class InvalidTransition(Exception):
    """Raised when a requested change is not legal from the current state."""


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    Status.SUBMITTED: {Status.IN_REVIEW, Status.DUPLICATE},
    Status.IN_REVIEW: {Status.IN_PROGRESS, Status.DUPLICATE},
    Status.IN_PROGRESS: {Status.RESOLVED, Status.DUPLICATE},
    Status.RESOLVED: {Status.CLOSED, Status.IN_PROGRESS},
    Status.CLOSED: set(),
    Status.DUPLICATE: set(),
}


@transaction.atomic
def transition(
    complaint: Complaint, to_status: str, actor, note: str = ""
) -> ComplaintEvent:
    from_status = complaint.status
    if to_status not in ALLOWED_TRANSITIONS[from_status]:
        raise InvalidTransition(f"Cannot move a complaint from {from_status} to {to_status}")

    complaint.status = to_status
    if to_status == Status.RESOLVED and complaint.resolved_at is None:
        complaint.resolved_at = timezone.now()
    complaint.save()

    return ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.STATUS,
        from_value=from_status,
        to_value=to_status,
        actor=actor,
        note=note,
    )


@transaction.atomic
def triage(
    complaint: Complaint,
    category: Category,
    priority: str,
    actor,
    prediction: Prediction | None = None,
) -> ComplaintEvent:
    """Human confirmation of category and priority. Starts the SLA clock.

    triaged_at is set here and nowhere else. A model predicting instantly does
    not start the clock; a human confirming does.
    """
    if category.domain_id != complaint.domain_id:
        raise InvalidTransition(
            f"Category {category} belongs to another domain than complaint #{complaint.pk}"
        )
    if priority not in Priority.values:
        raise InvalidTransition(f"Unknown priority {priority!r}")

    previous_category = complaint.category.slug if complaint.category else None
    previous_priority = complaint.priority

    now = timezone.now()
    complaint.category = category
    complaint.priority = priority
    complaint.triaged_at = now
    complaint.due_at = now + timedelta(hours=category.sla_hours)
    complaint.save()

    category_event = ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.CATEGORY,
        from_value=previous_category,
        to_value=category.slug,
        actor=actor,
        prediction=prediction,
    )
    ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.PRIORITY,
        from_value=previous_priority,
        to_value=priority,
        actor=actor,
    )
    if complaint.status == Status.SUBMITTED:
        transition(complaint, Status.IN_REVIEW, actor, note="Triaged")

    return category_event


@transaction.atomic
def assign(complaint: Complaint, assignee, actor) -> ComplaintEvent:
    previous = complaint.assignee.username if complaint.assignee else None
    complaint.assignee = assignee
    complaint.save()
    return ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.ASSIGNMENT,
        from_value=previous,
        to_value=assignee.username if assignee else None,
        actor=actor,
    )


def resolve(complaint: Complaint, actor, note: str = "") -> ComplaintEvent:
    return transition(complaint, Status.RESOLVED, actor, note=note)
