import pytest
from django.contrib.auth.models import Group
from django.urls import reverse

from complaints.models import Complaint, Status
from complaints.permissions import AGENT, bootstrap_groups
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
