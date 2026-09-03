"""Roles are bundles of permissions, never hardcoded checks.

Authorization asks `user.has_perm("complaints.triage_complaint")`, never
`user.role == "agent"`. Adding a Supervisor tier later is a new bundle here,
with no change to any call site.
"""

from django.contrib.auth.models import Group, Permission

SUBMITTER = "Submitter"
AGENT = "Agent"
ADMIN = "Admin"

_SUBMITTER_PERMS = {
    "complaints.add_complaint",
    "complaints.view_complaint",
}

_AGENT_PERMS = _SUBMITTER_PERMS | {
    "complaints.view_queue",
    "complaints.change_complaint",
    "complaints.triage_complaint",
    "complaints.assign_complaint",
    "complaints.resolve_complaint",
    "complaints.mark_duplicate",
}

_ADMIN_PERMS = _AGENT_PERMS | {
    "domains.manage_domain",
    "complaints.view_ml_metrics",
}

GROUP_PERMISSIONS: dict[str, set[str]] = {
    SUBMITTER: _SUBMITTER_PERMS,
    AGENT: _AGENT_PERMS,
    ADMIN: _ADMIN_PERMS,
}


def bootstrap_groups() -> None:
    """Create the three groups and set their permissions. Idempotent."""
    for group_name, codenames in GROUP_PERMISSIONS.items():
        group, _ = Group.objects.get_or_create(name=group_name)
        permissions = []
        for dotted in codenames:
            app_label, codename = dotted.split(".")
            permissions.append(
                Permission.objects.get(content_type__app_label=app_label, codename=codename)
            )
        group.permissions.set(permissions)
