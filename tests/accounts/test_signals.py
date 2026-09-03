import pytest
from django.contrib.auth.models import User

from complaints.permissions import SUBMITTER, bootstrap_groups


@pytest.mark.django_db
def test_new_users_become_submitters():
    bootstrap_groups()
    user = User.objects.create_user(username="newcomer", email="new@example.com")
    assert user.groups.filter(name=SUBMITTER).exists()
    assert user.has_perm("complaints.add_complaint")


@pytest.mark.django_db
def test_new_users_are_not_agents():
    bootstrap_groups()
    user = User.objects.create_user(username="newcomer2", email="new2@example.com")
    assert not user.has_perm("complaints.triage_complaint")


@pytest.mark.django_db
def test_superusers_are_not_downgraded():
    bootstrap_groups()
    admin = User.objects.create_superuser(username="root", email="root@example.com", password="x")
    assert admin.has_perm("complaints.triage_complaint")
