"""Idempotent demo data for local exploration and the deployed demo instance.

Every complaint is created through complaints/services.py, never by direct
field assignment, so seeded complaints carry a genuine ComplaintEvent history
instead of fabricated state. Every get_or_create is keyed so re-running this
command (the deploy runs it on every boot) never multiplies rows.
"""

from django.contrib.auth.models import Group, User
from django.core.management.base import BaseCommand

from complaints import services
from complaints.models import Complaint, Priority, Status
from complaints.permissions import AGENT, SUBMITTER, bootstrap_groups
from domains.models import Category, Domain
from domains.packs import PACKS

# slug -> [(category slug, display name, sla_hours), ...]
CATEGORIES: dict[str, list[tuple[str, str, int]]] = {
    "cfpb": [
        ("mortgage", "Mortgage", 72),
        ("credit_card", "Credit card", 48),
        ("debt_collection", "Debt collection", 72),
        ("other", "Other", 96),
    ],
    "nyc311": [
        ("noise", "Noise", 24),
        ("street_condition", "Street condition", 120),
        ("sanitation", "Sanitation", 48),
        ("other", "Other", 96),
    ],
}


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
        for domain_slug, entries in CATEGORIES.items():
            domain = domains[domain_slug]
            categories[domain_slug] = {}
            for cat_slug, name, sla_hours in entries:
                category, _ = Category.objects.get_or_create(
                    domain=domain,
                    slug=cat_slug,
                    defaults={"name": name, "sla_hours": sla_hours},
                )
                categories[domain_slug][cat_slug] = category

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

        cfpb = domains["cfpb"]
        nyc311 = domains["nyc311"]

        # Two complaints left untouched, as submitted by the demo user.
        for title, domain, body in [
            (
                "Mortgage servicer misapplied my payment",
                cfpb,
                "My servicer applied last month's payment to the wrong account and "
                "will not correct it despite two calls.",
            ),
            (
                "Loud construction noise every night this week",
                nyc311,
                "Construction crew has been running heavy equipment well past "
                "the permitted hours, every night this week.",
            ),
        ]:
            Complaint.objects.get_or_create(
                title=title,
                defaults={
                    "domain": domain,
                    "submitted_by": submitter,
                    "body": body,
                },
            )

        # Two complaints triaged via the service layer.
        triaged_specs = [
            (
                "Credit card charged twice for the same purchase",
                cfpb,
                "I was billed twice for the same purchase and the merchant "
                "dispute has gone nowhere.",
                categories["cfpb"]["credit_card"],
                Priority.MEDIUM,
            ),
            (
                "Pothole on Main Street damaging cars",
                nyc311,
                "A large pothole has formed on Main Street and has already damaged several cars.",
                categories["nyc311"]["street_condition"],
                Priority.HIGH,
            ),
        ]
        triaged_complaints = []
        for title, domain, body, category, priority in triaged_specs:
            complaint, created = Complaint.objects.get_or_create(
                title=title,
                defaults={
                    "domain": domain,
                    "submitted_by": submitter,
                    "body": body,
                },
            )
            if created:
                services.triage(complaint, category, priority, actor=agent)
            triaged_complaints.append(complaint)

        # One complaint carried all the way through to resolved.
        resolved_title = "Debt collector calling after being asked to stop"
        resolved_complaint, created = Complaint.objects.get_or_create(
            title=resolved_title,
            defaults={
                "domain": cfpb,
                "submitted_by": submitter,
                "body": "A debt collector keeps calling after I sent a written "
                "request to stop contact.",
            },
        )
        if created:
            services.triage(
                resolved_complaint,
                categories["cfpb"]["debt_collection"],
                Priority.HIGH,
                actor=agent,
            )
            services.transition(resolved_complaint, Status.IN_PROGRESS, actor=agent)
            services.resolve(
                resolved_complaint, actor=agent, note="Collector confirmed cease of contact."
            )

        # One complaint marked as a duplicate of the first triaged complaint.
        canonical = triaged_complaints[0]
        duplicate_title = "Charged twice again on my credit card statement"
        duplicate_complaint, created = Complaint.objects.get_or_create(
            title=duplicate_title,
            defaults={
                "domain": cfpb,
                "submitted_by": submitter,
                "body": "Same double-charge issue as my other report on this statement.",
            },
        )
        if created:
            services.mark_duplicate(duplicate_complaint, canonical, actor=agent)

        self.stdout.write(self.style.SUCCESS("Demo data seeded."))
