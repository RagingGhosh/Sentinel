import pytest
from django.db import IntegrityError

from domains.models import Category, Domain


@pytest.mark.django_db
def test_domain_slug_is_unique():
    Domain.objects.create(slug="cfpb", name="Consumer Financial")
    with pytest.raises(IntegrityError):
        Domain.objects.create(slug="cfpb", name="Duplicate")


@pytest.mark.django_db
def test_category_slug_unique_per_domain_not_globally():
    """The same category slug in two domains is legitimate."""
    cfpb = Domain.objects.create(slug="cfpb", name="Consumer Financial")
    nyc = Domain.objects.create(slug="nyc311", name="Civic Services")
    Category.objects.create(domain=cfpb, slug="other", name="Other", sla_hours=72)
    Category.objects.create(domain=nyc, slug="other", name="Other", sla_hours=48)
    assert Category.objects.count() == 2


@pytest.mark.django_db
def test_category_slug_collides_within_one_domain():
    cfpb = Domain.objects.create(slug="cfpb", name="Consumer Financial")
    Category.objects.create(domain=cfpb, slug="mortgage", name="Mortgage", sla_hours=72)
    with pytest.raises(IntegrityError):
        Category.objects.create(domain=cfpb, slug="mortgage", name="Dup", sla_hours=24)


@pytest.mark.django_db
def test_sla_hours_must_be_positive():
    cfpb = Domain.objects.create(slug="cfpb", name="Consumer Financial")
    with pytest.raises(IntegrityError):
        Category.objects.create(domain=cfpb, slug="bad", name="Bad", sla_hours=0)
