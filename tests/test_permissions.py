import pytest
from django.contrib.auth.models import Group

from complaints.permissions import ADMIN, AGENT, SUBMITTER, bootstrap_groups
from tests.factories import UserFactory

# (group, permission, is_granted) for every meaningful pair, negatives included.
MATRIX = [
    (SUBMITTER, "complaints.add_complaint", True),
    (SUBMITTER, "complaints.view_complaint", True),
    (SUBMITTER, "complaints.view_queue", False),
    (SUBMITTER, "complaints.triage_complaint", False),
    (SUBMITTER, "complaints.assign_complaint", False),
    (SUBMITTER, "complaints.resolve_complaint", False),
    (SUBMITTER, "complaints.mark_duplicate", False),
    (SUBMITTER, "domains.manage_domain", False),
    (SUBMITTER, "complaints.view_ml_metrics", False),
    (AGENT, "complaints.add_complaint", True),
    (AGENT, "complaints.view_complaint", True),
    (AGENT, "complaints.view_queue", True),
    (AGENT, "complaints.triage_complaint", True),
    (AGENT, "complaints.assign_complaint", True),
    (AGENT, "complaints.resolve_complaint", True),
    (AGENT, "complaints.mark_duplicate", True),
    (AGENT, "domains.manage_domain", False),
    (AGENT, "complaints.view_ml_metrics", False),
    (ADMIN, "complaints.view_queue", True),
    (ADMIN, "complaints.triage_complaint", True),
    (ADMIN, "complaints.assign_complaint", True),
    (ADMIN, "complaints.resolve_complaint", True),
    (ADMIN, "complaints.mark_duplicate", True),
    (ADMIN, "domains.manage_domain", True),
    (ADMIN, "complaints.view_ml_metrics", True),
]


@pytest.mark.django_db
@pytest.mark.parametrize("group_name,permission,granted", MATRIX)
def test_permission_matrix(group_name, permission, granted):
    bootstrap_groups()
    user = UserFactory()
    user.groups.add(Group.objects.get(name=group_name))
    user = type(user).objects.get(pk=user.pk)  # drop the permission cache
    assert user.has_perm(permission) is granted


@pytest.mark.django_db
def test_bootstrap_is_idempotent():
    bootstrap_groups()
    bootstrap_groups()
    assert Group.objects.filter(name__in=[SUBMITTER, AGENT, ADMIN]).count() == 3
