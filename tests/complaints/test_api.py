import pytest
from django.contrib.auth.models import Group
from rest_framework.test import APIClient

from complaints.models import Priority, Status
from complaints.permissions import AGENT, bootstrap_groups
from tests.factories import CategoryFactory, ComplaintFactory, DomainFactory, UserFactory


@pytest.fixture
def agent(db):
    bootstrap_groups()
    user = UserFactory()
    user.groups.add(Group.objects.get(name=AGENT))
    return type(user).objects.get(pk=user.pk)


@pytest.fixture
def api():
    return APIClient()


@pytest.mark.django_db
def test_anonymous_users_cannot_list_complaints(api):
    assert api.get("/api/complaints/").status_code in (401, 403)


@pytest.mark.django_db
def test_submitters_see_only_their_own_complaints(api):
    bootstrap_groups()
    mine = UserFactory()
    ComplaintFactory(submitted_by=mine)
    ComplaintFactory()  # someone else's
    api.force_authenticate(mine)
    response = api.get("/api/complaints/")
    assert response.status_code == 200
    assert len(response.data["results"]) == 1


@pytest.mark.django_db
def test_agents_see_every_complaint(api, agent):
    ComplaintFactory()
    ComplaintFactory()
    api.force_authenticate(agent)
    response = api.get("/api/complaints/")
    assert len(response.data["results"]) == 2


@pytest.mark.django_db
def test_submitter_cannot_triage(api):
    bootstrap_groups()
    submitter = UserFactory()
    category = CategoryFactory()
    complaint = ComplaintFactory(domain=category.domain)
    api.force_authenticate(submitter)
    response = api.post(
        f"/api/complaints/{complaint.pk}/triage/",
        {"category": category.pk, "priority": Priority.HIGH},
        format="json",
    )
    assert response.status_code == 403


@pytest.mark.django_db
def test_agent_can_triage_and_the_sla_clock_starts(api, agent):
    category = CategoryFactory(sla_hours=24)
    complaint = ComplaintFactory(domain=category.domain)
    api.force_authenticate(agent)
    response = api.post(
        f"/api/complaints/{complaint.pk}/triage/",
        {"category": category.pk, "priority": Priority.HIGH},
        format="json",
    )
    assert response.status_code == 200
    complaint.refresh_from_db()
    assert complaint.status == Status.IN_REVIEW
    assert complaint.due_at is not None


@pytest.mark.django_db
def test_triage_with_a_foreign_domain_category_is_a_400_not_a_500(api, agent):
    complaint = ComplaintFactory()
    foreign = CategoryFactory()
    api.force_authenticate(agent)
    response = api.post(
        f"/api/complaints/{complaint.pk}/triage/",
        {"category": foreign.pk, "priority": Priority.LOW},
        format="json",
    )
    assert response.status_code == 400


@pytest.mark.django_db
def test_creating_a_complaint_records_the_submitter(api):
    bootstrap_groups()
    user = UserFactory()
    domain = DomainFactory()
    api.force_authenticate(user)
    response = api.post(
        "/api/complaints/",
        {"domain": domain.pk, "title": "Wrong charge", "body": "I was billed twice."},
        format="json",
    )
    assert response.status_code == 201
    assert response.data["submitted_by"] == user.username


@pytest.mark.django_db
def test_healthz_reports_model_registry_state(api):
    response = api.get("/healthz")
    assert response.status_code == 200
    assert response.data["models"]["cfpb"]["triage"] == "null"
