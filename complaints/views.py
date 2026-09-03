from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin
from django.contrib.auth.models import User
from django.db.models import F, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.views.generic import ListView

from complaints import services
from complaints.models import Complaint, Priority, Status
from domains.models import Category, Domain


def _assignable_users():
    """Users who hold complaints.view_queue: a reasonable pool for assignment."""
    return (
        User.objects.filter(
            Q(groups__permissions__codename="view_queue")
            | Q(user_permissions__codename="view_queue")
            | Q(is_superuser=True)
        )
        .distinct()
        .order_by("username")
    )


class ComplaintListView(LoginRequiredMixin, ListView):
    """A submitter's own complaints."""

    template_name = "complaints/list.html"
    context_object_name = "complaints"
    paginate_by = 25

    def get_queryset(self):
        return Complaint.objects.filter(submitted_by=self.request.user).select_related(
            "domain", "category"
        )


class QueueView(LoginRequiredMixin, PermissionRequiredMixin, ListView):
    """The agent work queue: everything not yet finished."""

    permission_required = "complaints.view_queue"
    raise_exception = True
    template_name = "complaints/queue.html"
    context_object_name = "complaints"
    paginate_by = 25

    def get_queryset(self):
        return (
            Complaint.objects.exclude(status__in=[Status.CLOSED, Status.DUPLICATE])
            .select_related("domain", "category", "assignee")
            .order_by(F("due_at").asc(nulls_first=True), "-created_at")
        )


@login_required
def submit(request):
    if request.method == "POST":
        domain_id = request.POST.get("domain", "").strip()
        title = request.POST.get("title", "").strip()
        body = request.POST.get("body", "").strip()
        if not domain_id or not title or not body:
            messages.error(request, "Domain, title and body are all required.")
            return redirect("complaint-submit")
        complaint = Complaint.objects.create(
            domain=get_object_or_404(Domain, pk=domain_id, is_active=True),
            title=title,
            body=body,
            submitted_by=request.user,
        )
        return redirect("complaint-detail", pk=complaint.pk)
    return render(
        request, "complaints/submit.html", {"domains": Domain.objects.filter(is_active=True)}
    )


@login_required
def detail(request, pk):
    queryset = Complaint.objects.select_related("domain", "category", "submitted_by")
    if not request.user.has_perm("complaints.view_queue"):
        queryset = queryset.filter(submitted_by=request.user)
    complaint = get_object_or_404(queryset, pk=pk)

    if request.method == "POST":
        action = request.POST.get("action")
        try:
            if action == "triage" and request.user.has_perm("complaints.triage_complaint"):
                category_id = request.POST.get("category", "").strip()
                priority = request.POST.get("priority", "").strip()
                if not category_id or not priority:
                    messages.error(request, "Category and priority are both required to triage.")
                else:
                    category = get_object_or_404(Category, pk=category_id)
                    services.triage(complaint, category, priority, request.user)
            elif action == "start_work" and request.user.has_perm("complaints.triage_complaint"):
                services.transition(complaint, Status.IN_PROGRESS, request.user)
            elif action == "assign" and request.user.has_perm("complaints.assign_complaint"):
                assignee_id = request.POST.get("assignee", "").strip()
                if not assignee_id:
                    messages.error(request, "Choose someone to assign this complaint to.")
                else:
                    assignee = get_object_or_404(User, pk=assignee_id)
                    services.assign(complaint, assignee, request.user)
            elif action == "mark_duplicate" and request.user.has_perm("complaints.mark_duplicate"):
                canonical_id = request.POST.get("canonical", "").strip()
                if not canonical_id:
                    messages.error(
                        request, "Enter the canonical complaint's id to mark this as a duplicate."
                    )
                else:
                    canonical = get_object_or_404(Complaint, pk=canonical_id)
                    services.mark_duplicate(complaint, canonical, request.user)
            elif action == "resolve" and request.user.has_perm("complaints.resolve_complaint"):
                services.resolve(complaint, request.user)
            elif action == "close" and request.user.has_perm("complaints.resolve_complaint"):
                services.transition(complaint, Status.CLOSED, request.user)
        except services.InvalidTransition as exc:
            messages.error(request, str(exc))
        return redirect("complaint-detail", pk=complaint.pk)

    legal_targets = services.ALLOWED_TRANSITIONS[complaint.status]
    return render(
        request,
        "complaints/detail.html",
        {
            "complaint": complaint,
            "events": complaint.events.select_related("actor", "prediction"),
            "categories": Category.objects.filter(domain=complaint.domain),
            "priorities": Priority.choices,
            "assignable_users": _assignable_users(),
            "can_start_work": (
                request.user.has_perm("complaints.triage_complaint")
                and Status.IN_PROGRESS in legal_targets
            ),
            "can_assign": (
                request.user.has_perm("complaints.assign_complaint")
                and complaint.status not in (Status.CLOSED, Status.DUPLICATE)
            ),
            "can_mark_duplicate": (
                request.user.has_perm("complaints.mark_duplicate")
                and Status.DUPLICATE in legal_targets
            ),
            "can_resolve": (
                request.user.has_perm("complaints.resolve_complaint")
                and Status.RESOLVED in legal_targets
            ),
            "can_close": (
                request.user.has_perm("complaints.resolve_complaint")
                and Status.CLOSED in legal_targets
            ),
        },
    )
