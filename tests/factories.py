import factory
from django.contrib.auth.models import User

from complaints.models import Complaint
from domains.models import Category, Domain


class UserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda o: f"{o.username}@example.com")


class DomainFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Domain

    slug = factory.Sequence(lambda n: f"domain{n}")
    name = factory.Sequence(lambda n: f"Domain {n}")


class CategoryFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Category

    domain = factory.SubFactory(DomainFactory)
    slug = factory.Sequence(lambda n: f"category{n}")
    name = factory.Sequence(lambda n: f"Category {n}")
    sla_hours = 72


class ComplaintFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Complaint

    domain = factory.SubFactory(DomainFactory)
    submitted_by = factory.SubFactory(UserFactory)
    title = factory.Sequence(lambda n: f"Complaint {n}")
    body = "The servicer applied my payment to the wrong account and will not correct it."
