import pytest
from django.contrib.auth.models import Group
from django.urls import reverse
from django.utils import timezone

from complaints.models import Complaint, Status
from complaints.permissions import AGENT, bootstrap_groups
from complaints.views import QueueView
from tests.factories import CategoryFactory, ComplaintFactory, DomainFactory, UserFactory


@pytest.mark.django_db
def test_submit_page_requires_login(client):
    response = client.get(reverse("complaint-submit"))
    assert response.status_code == 302
    assert "/accounts/" in response.url


@pytest.mark.django_db
def test_submitting_creates_a_complaint_in_submitted_state(client):
    bootstrap_groups()
    user = UserFactory()
    domain = DomainFactory()
    client.force_login(user)
    response = client.post(
        reverse("complaint-submit"),
        {"domain": domain.pk, "title": "Double charged", "body": "Billed twice in March."},
    )
    assert response.status_code == 302
    complaint = Complaint.objects.get()
    assert complaint.status == Status.SUBMITTED
    assert complaint.submitted_by == user
    assert complaint.category is None


@pytest.mark.django_db
def test_submitter_cannot_open_another_users_complaint(client):
    bootstrap_groups()
    intruder = UserFactory()
    someone_elses = ComplaintFactory()
    client.force_login(intruder)
    response = client.get(reverse("complaint-detail", args=[someone_elses.pk]))
    assert response.status_code == 404


@pytest.mark.django_db
def test_queue_is_closed_to_submitters(client):
    bootstrap_groups()
    client.force_login(UserFactory())
    assert client.get(reverse("complaint-queue")).status_code == 403


@pytest.mark.django_db
def test_queue_lists_open_complaints_for_agents(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    ComplaintFactory(title="Open one")
    client.force_login(agent)
    response = client.get(reverse("complaint-queue"))
    assert response.status_code == 200
    assert b"Open one" in response.content


@pytest.mark.django_db
def test_agent_triage_form_moves_the_complaint_to_in_review(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    category = CategoryFactory(sla_hours=12)
    complaint = ComplaintFactory(domain=category.domain)
    client.force_login(agent)
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "triage", "category": category.pk, "priority": "high"},
    )
    assert response.status_code == 302
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_REVIEW
    assert complaint.due_at is not None


@pytest.mark.django_db
def test_submit_missing_title_does_not_create_a_complaint_or_500(client):
    bootstrap_groups()
    user = UserFactory()
    domain = DomainFactory()
    client.force_login(user)
    client.raise_request_exception = False
    response = client.post(
        reverse("complaint-submit"),
        {"domain": domain.pk, "body": "Billed twice in March."},
    )
    assert response.status_code in (200, 302)
    assert Complaint.objects.count() == 0


@pytest.mark.django_db
def test_triage_missing_priority_does_not_change_the_complaint_or_500(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    category = CategoryFactory(sla_hours=12)
    complaint = ComplaintFactory(domain=category.domain)
    client.force_login(agent)
    client.raise_request_exception = False
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "triage", "category": category.pk},
    )
    assert response.status_code in (200, 302)
    complaint.refresh_from_db()
    assert complaint.status == Status.SUBMITTED
    assert complaint.due_at is None


@pytest.mark.django_db
def test_agent_can_start_work_on_an_in_review_complaint(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    complaint = ComplaintFactory(status=Status.IN_REVIEW)
    client.force_login(agent)
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "start_work"},
    )
    assert response.status_code == 302
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_PROGRESS


@pytest.mark.django_db
def test_non_agent_cannot_start_work(client):
    bootstrap_groups()
    submitter = UserFactory()
    complaint = ComplaintFactory(status=Status.IN_REVIEW, submitted_by=submitter)
    client.force_login(submitter)
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "start_work"},
    )
    assert response.status_code == 302
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_REVIEW


@pytest.mark.django_db
def test_agent_can_assign_a_complaint(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    assignee = UserFactory()
    assignee.groups.add(Group.objects.get(name=AGENT))
    complaint = ComplaintFactory()
    client.force_login(agent)
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "assign", "assignee": assignee.pk},
    )
    assert response.status_code == 302
    complaint.refresh_from_db()
    assert complaint.assignee == assignee


@pytest.mark.django_db
def test_non_agent_cannot_assign(client):
    bootstrap_groups()
    submitter = UserFactory()
    other = UserFactory()
    complaint = ComplaintFactory(submitted_by=submitter)
    client.force_login(submitter)
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "assign", "assignee": other.pk},
    )
    assert response.status_code == 302
    complaint.refresh_from_db()
    assert complaint.assignee is None


@pytest.mark.django_db
def test_agent_can_mark_a_complaint_duplicate(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    canonical = ComplaintFactory()
    duplicate = ComplaintFactory(domain=canonical.domain)
    client.force_login(agent)
    response = client.post(
        reverse("complaint-detail", args=[duplicate.pk]),
        {"action": "mark_duplicate", "canonical": canonical.pk},
    )
    assert response.status_code == 302
    duplicate.refresh_from_db()
    assert duplicate.status == Status.DUPLICATE
    assert duplicate.duplicate_of == canonical


@pytest.mark.django_db
def test_non_agent_cannot_mark_duplicate(client):
    bootstrap_groups()
    submitter = UserFactory()
    canonical = ComplaintFactory()
    duplicate = ComplaintFactory(domain=canonical.domain, submitted_by=submitter)
    client.force_login(submitter)
    response = client.post(
        reverse("complaint-detail", args=[duplicate.pk]),
        {"action": "mark_duplicate", "canonical": canonical.pk},
    )
    assert response.status_code == 302
    duplicate.refresh_from_db()
    assert duplicate.status == Status.SUBMITTED


@pytest.mark.django_db
def test_agent_can_close_a_resolved_complaint(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    complaint = ComplaintFactory(status=Status.RESOLVED, resolved_at=timezone.now())
    client.force_login(agent)
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "close"},
    )
    assert response.status_code == 302
    complaint.refresh_from_db()
    assert complaint.status == Status.CLOSED


@pytest.mark.django_db
def test_non_agent_cannot_close(client):
    bootstrap_groups()
    submitter = UserFactory()
    complaint = ComplaintFactory(
        status=Status.RESOLVED, resolved_at=timezone.now(), submitted_by=submitter
    )
    client.force_login(submitter)
    response = client.post(
        reverse("complaint-detail", args=[complaint.pk]),
        {"action": "close"},
    )
    assert response.status_code == 302
    complaint.refresh_from_db()
    assert complaint.status == Status.RESOLVED


@pytest.mark.django_db
def test_queue_orders_untriaged_complaints_before_triaged_ones_using_explicit_nulls_first():
    """NULL due_at sorts first on SQLite by default but last on PostgreSQL. The
    ordering must say NULLS FIRST explicitly so both backends agree."""
    sql = str(QueueView().get_queryset().query)
    assert "NULLS FIRST" in sql.upper()


@pytest.mark.django_db
def test_submit_rejects_an_inactive_domain(client):
    user = UserFactory()
    domain = DomainFactory(is_active=False)
    client.force_login(user)
    response = client.post(
        reverse("complaint-submit"),
        {"domain": domain.pk, "title": "Should not work", "body": "Inactive domain."},
    )
    assert response.status_code == 404
    assert Complaint.objects.count() == 0


@pytest.mark.django_db
def test_full_lifecycle_is_walkable_through_the_views(client):
    bootstrap_groups()
    agent = UserFactory()
    agent.groups.add(Group.objects.get(name=AGENT))
    category = CategoryFactory(sla_hours=48)
    complaint = ComplaintFactory(domain=category.domain)
    client.force_login(agent)
    detail_url = reverse("complaint-detail", args=[complaint.pk])

    client.post(detail_url, {"action": "triage", "category": category.pk, "priority": "high"})
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_REVIEW

    client.post(detail_url, {"action": "start_work"})
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_PROGRESS

    client.post(detail_url, {"action": "resolve"})
    complaint.refresh_from_db()
    assert complaint.status == Status.RESOLVED

    client.post(detail_url, {"action": "close"})
    complaint.refresh_from_db()
    assert complaint.status == Status.CLOSED
