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
def transition(complaint: Complaint, to_status: str, actor, note: str = "") -> ComplaintEvent:
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
    """Human confirmation of category and priority. Starts the SLA clock once.

    triaged_at is set on the *first* triage and is immutable after that: the
    first human confirmation is the moment the clock starts, and it never
    moves, even across a later re-triage. due_at, by contrast, is recomputed
    from triaged_at on *every* triage call, as triaged_at + the confirmed
    category's sla_hours -- so correcting a mis-categorisation adjusts the
    deadline to the corrected SLA without ever discarding elapsed time.

    This is a deliberate controller ruling, not an oversight. Restarting the
    clock on re-triage would erase an in-flight breach; freezing due_at too
    would lock in a wrong deadline after a legitimate category correction.
    Both were rejected. The consequence, not a bug: a complaint re-triaged
    into a shorter-SLA category can come out already overdue. That is the
    honest reading of a late correction, and `due_at` is left in the past
    rather than bumped to "now" -- resolved_at > due_at must stay a reliable
    breach signal for the Phase 3 risk model's training labels.
    """
    if category.domain_id != complaint.domain_id:
        raise InvalidTransition(
            f"Category {category} belongs to another domain than complaint #{complaint.pk}"
        )
    if priority not in Priority.values:
        raise InvalidTransition(f"Unknown priority {priority!r}")

    previous_category = complaint.category.slug if complaint.category else None
    previous_priority = complaint.priority

    triaged_at = complaint.triaged_at or timezone.now()
    complaint.category = category
    complaint.priority = priority
    complaint.triaged_at = triaged_at
    complaint.due_at = triaged_at + timedelta(hours=category.sla_hours)
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


@transaction.atomic
def mark_duplicate(
    complaint: Complaint,
    canonical: Complaint,
    actor,
    prediction: Prediction | None = None,
) -> ComplaintEvent:
    """Mark `complaint` as a duplicate of `canonical`.

    The database enforces "not itself" and "must have a canonical". The chain
    rule needs a query, so it lives here: a canonical may not itself be a
    duplicate, which keeps every duplicate exactly one hop from a real
    complaint and makes cycles impossible.
    """
    if complaint.pk == canonical.pk:
        raise InvalidTransition("A complaint cannot be a duplicate of itself")
    if canonical.duplicate_of_id is not None or canonical.status == Status.DUPLICATE:
        raise InvalidTransition(
            f"Complaint #{canonical.pk} is itself a duplicate; point at its canonical instead"
        )
    if complaint.domain_id != canonical.domain_id:
        raise InvalidTransition("Complaints in different domains cannot be duplicates")

    from_status = complaint.status
    if Status.DUPLICATE not in ALLOWED_TRANSITIONS[from_status]:
        raise InvalidTransition(f"Cannot mark a {from_status} complaint as a duplicate")

    complaint.duplicate_of = canonical
    complaint.status = Status.DUPLICATE
    complaint.save()

    ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.STATUS,
        from_value=from_status,
        to_value=Status.DUPLICATE,
        actor=actor,
    )
    return ComplaintEvent.objects.create(
        complaint=complaint,
        kind=EventKind.DUPLICATE,
        from_value=None,
        to_value=str(canonical.pk),
        actor=actor,
        prediction=prediction,
    )
