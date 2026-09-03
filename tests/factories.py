import factory
from django.contrib.auth.models import User

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
