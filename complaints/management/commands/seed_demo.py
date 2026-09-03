"""Idempotent demo data for local exploration and the deployed demo instance.

Every complaint is created through complaints/services.py, never by direct
field assignment, so seeded complaints carry a genuine ComplaintEvent history
instead of fabricated state. Every get_or_create is keyed so re-running this
command (the deploy runs it on every boot) never multiplies rows.

This command understands the concept of a domain pack -- it iterates PACKS --
but never the meaning of any particular one. No domain slug is ever written
here; both the taxonomy (domains.packs.DomainPack.demo_categories) and the
seeded content below are keyed off whatever packs happen to be registered, so
registering a new pack seeds a full demo set for it automatically.
"""

from django.contrib.auth.models import Group, User
from django.core.management.base import BaseCommand

from complaints import services
from complaints.models import Complaint, Priority, Status
from complaints.permissions import AGENT, SUBMITTER, bootstrap_groups
from domains.models import Category, Domain
from domains.packs import PACKS


class Command(BaseCommand):
    help = "Seed demo domains, categories, users and complaints. Safe to run repeatedly."

    def handle(self, *args, **options):
        bootstrap_groups()

        domains: dict[str, Domain] = {}
        for slug, pack in PACKS.items():
            domain, _ = Domain.objects.get_or_create(
                slug=slug, defaults={"name": pack.display_name}
            )
            domains[slug] = domain

        categories: dict[str, dict[str, Category]] = {}
        for slug, pack in PACKS.items():
            domain = domains[slug]
            categories[slug] = {}
            for cat_slug, name, sla_hours in pack.demo_categories:
                category, _ = Category.objects.get_or_create(
                    domain=domain,
                    slug=cat_slug,
                    defaults={"name": name, "sla_hours": sla_hours},
                )
                categories[slug][cat_slug] = category

        agent, created = User.objects.get_or_create(
            username="demo-agent", defaults={"email": "demo-agent@example.com"}
        )
        if created:
            agent.set_unusable_password()
            agent.save()
        agent.groups.add(Group.objects.get(name=AGENT))

        submitter, created = User.objects.get_or_create(
            username="demo-user", defaults={"email": "demo-user@example.com"}
        )
        if created:
            submitter.set_unusable_password()
            submitter.save()
        submitter.groups.add(Group.objects.get(name=SUBMITTER))

        for slug, pack in PACKS.items():
            domain = domains[slug]
            pack_categories = list(categories[slug].values())
            if not pack_categories:
                continue
            first_category = pack_categories[0]
            second_category = pack_categories[1] if len(pack_categories) > 1 else first_category

            # One complaint left untouched, as submitted by the demo user.
            Complaint.objects.get_or_create(
                title=f"Unreviewed {domain.name} report",
                defaults={
                    "domain": domain,
                    "submitted_by": submitter,
                    "body": f"A newly submitted complaint awaiting triage in {domain.name}.",
                },
            )

            # One complaint triaged via the service layer.
            triaged, created = Complaint.objects.get_or_create(
                title=f"{first_category.name} issue reported in {domain.name}",
                defaults={
                    "domain": domain,
                    "submitted_by": submitter,
                    "body": f"A {first_category.name.lower()} complaint awaiting agent review.",
                },
            )
            if created:
                services.triage(triaged, first_category, Priority.MEDIUM, actor=agent)

            # One complaint carried all the way through to resolved.
            resolved_complaint, created = Complaint.objects.get_or_create(
                title=f"{second_category.name} issue resolved in {domain.name}",
                defaults={
                    "domain": domain,
                    "submitted_by": submitter,
                    "body": f"A {second_category.name.lower()} complaint that has been "
                    "fully worked.",
                },
            )
            if created:
                services.triage(resolved_complaint, second_category, Priority.HIGH, actor=agent)
                services.transition(resolved_complaint, Status.IN_PROGRESS, actor=agent)
                services.resolve(resolved_complaint, actor=agent, note="Resolved during demo seed.")

            # One complaint marked as a duplicate of the first triaged complaint.
            duplicate_complaint, created = Complaint.objects.get_or_create(
                title=f"Duplicate {first_category.name} report in {domain.name}",
                defaults={
                    "domain": domain,
                    "submitted_by": submitter,
                    "body": "Same issue as another report already on file.",
                },
            )
            if created:
                services.mark_duplicate(duplicate_complaint, triaged, actor=agent)

        self.stdout.write(self.style.SUCCESS("Demo data seeded."))
