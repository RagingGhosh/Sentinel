import logging

from django.contrib.auth.models import Group, User
from django.db.models.signals import post_save
from django.dispatch import receiver

from complaints.permissions import SUBMITTER

logger = logging.getLogger(__name__)


@receiver(post_save, sender=User)
def assign_default_group(sender, instance: User, created: bool, **kwargs) -> None:
    """Every new account starts as a Submitter. Elevation is deliberate."""
    if not created:
        return
    try:
        instance.groups.add(Group.objects.get(name=SUBMITTER))
    except Group.DoesNotExist:
        logger.warning("Submitter group missing; run `manage.py bootstrap_groups`")
