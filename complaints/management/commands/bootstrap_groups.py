from django.core.management.base import BaseCommand

from complaints.permissions import bootstrap_groups


class Command(BaseCommand):
    help = "Create the Submitter, Agent and Admin groups with their permissions."

    def handle(self, *args, **options):
        bootstrap_groups()
        self.stdout.write(self.style.SUCCESS("Groups bootstrapped."))
