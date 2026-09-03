from django.conf import settings
from django.db import models

from domains.models import Category, Domain


class Status(models.TextChoices):
    SUBMITTED = "submitted", "Submitted"
    IN_REVIEW = "in_review", "In review"
    IN_PROGRESS = "in_progress", "In progress"
    RESOLVED = "resolved", "Resolved"
    CLOSED = "closed", "Closed"
    DUPLICATE = "duplicate", "Duplicate"


class Priority(models.TextChoices):
    LOW = "low", "Low"
    MEDIUM = "medium", "Medium"
    HIGH = "high", "High"
    CRITICAL = "critical", "Critical"


PRIORITY_RANK = {
    Priority.LOW: 0.25,
    Priority.MEDIUM: 0.5,
    Priority.HIGH: 0.75,
    Priority.CRITICAL: 1.0,
}


class Complaint(models.Model):
    domain = models.ForeignKey(Domain, on_delete=models.PROTECT, related_name="complaints")

    # Human-owned ground truth. Never written by a model.
    category = models.ForeignKey(
        Category, on_delete=models.PROTECT, null=True, blank=True, related_name="complaints"
    )
    priority = models.CharField(max_length=20, choices=Priority.choices, null=True, blank=True)

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.SUBMITTED)
    title = models.CharField(max_length=300)
    body = models.TextField()

    submitted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="submitted_complaints"
    )
    assignee = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assigned_complaints",
    )
    duplicate_of = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True, related_name="duplicates"
    )

    # float32 bytes, not JSON: 1,536 bytes against roughly 9KB serialized.
    embedding = models.BinaryField(null=True, blank=True, editable=False)

    created_at = models.DateTimeField(auto_now_add=True)
    triaged_at = models.DateTimeField(null=True, blank=True)
    due_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        permissions = [
            ("view_queue", "Can view the agent work queue"),
            ("triage_complaint", "Can confirm category and priority"),
            ("assign_complaint", "Can assign complaints to agents"),
            ("resolve_complaint", "Can resolve and close complaints"),
            ("mark_duplicate", "Can mark a complaint as a duplicate"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(status=Status.DUPLICATE) | models.Q(duplicate_of__isnull=False),
                name="duplicate_requires_canonical",
            ),
            models.CheckConstraint(
                condition=~models.Q(duplicate_of=models.F("id")),
                name="duplicate_of_is_not_self",
            ),
        ]

    def __str__(self) -> str:
        return f"#{self.pk} {self.title[:60]}"

    @property
    def is_overdue(self) -> bool:
        from django.utils import timezone

        return (
            self.due_at is not None
            and self.resolved_at is None
            and self.due_at < timezone.now()
        )


class ImmutableRecordError(Exception):
    """Raised on any attempt to modify an append-only record."""


class PredictionKind(models.TextChoices):
    TRIAGE = "triage", "Triage"
    DEDUP = "dedup", "Duplicate detection"
    RISK = "risk", "SLA risk"


class Prediction(models.Model):
    """Append-only model output.

    A Prediction records what a model said and which artifact said it. It is
    never written into Complaint.category or Complaint.priority — those belong
    to a human. Keeping the two apart is what makes live evaluation possible:
    joining predictions to the human decisions in ComplaintEvent gives a real
    accuracy figure rather than a test-set one.
    """

    complaint = models.ForeignKey(Complaint, on_delete=models.CASCADE, related_name="predictions")
    kind = models.CharField(max_length=20, choices=PredictionKind.choices)
    payload = models.JSONField()
    model_name = models.CharField(max_length=100)
    model_version = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["complaint", "kind", "-created_at"])]
        permissions = [("view_ml_metrics", "Can view model metrics")]

    def save(self, *args, **kwargs):
        if self.pk is not None:
            raise ImmutableRecordError("Prediction rows are append-only and cannot be updated")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ImmutableRecordError("Prediction rows are append-only and cannot be deleted")

    def __str__(self) -> str:
        return f"{self.kind}@{self.model_version} for #{self.complaint_id}"
