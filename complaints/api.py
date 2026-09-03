from django.contrib.auth.models import User
from django.db.models import QuerySet
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from complaints import services
from complaints.models import Complaint, Priority, Status
from domains.models import Category
from ml.registry import registry_status


class ComplaintSerializer(serializers.ModelSerializer):
    submitted_by = serializers.SlugRelatedField(slug_field="username", read_only=True)
    is_overdue = serializers.BooleanField(read_only=True)

    class Meta:
        model = Complaint
        fields = [
            "id",
            "domain",
            "category",
            "priority",
            "status",
            "title",
            "body",
            "submitted_by",
            "assignee",
            "duplicate_of",
            "created_at",
            "triaged_at",
            "due_at",
            "resolved_at",
            "is_overdue",
        ]
        read_only_fields = [
            "category",
            "priority",
            "status",
            "submitted_by",
            "assignee",
            "duplicate_of",
            "created_at",
            "triaged_at",
            "due_at",
            "resolved_at",
        ]


class TriageSerializer(serializers.Serializer):
    category = serializers.PrimaryKeyRelatedField(queryset=Category.objects.all())
    priority = serializers.ChoiceField(choices=Priority.choices)


class AssignSerializer(serializers.Serializer):
    assignee = serializers.PrimaryKeyRelatedField(queryset=User.objects.all())


class MarkDuplicateSerializer(serializers.Serializer):
    canonical = serializers.PrimaryKeyRelatedField(queryset=Complaint.objects.all())


class ComplaintViewSet(viewsets.ModelViewSet):
    serializer_class = ComplaintSerializer
    http_method_names = ["get", "post", "head", "options"]

    def get_queryset(self) -> QuerySet[Complaint]:
        queryset = Complaint.objects.select_related("domain", "category", "submitted_by")
        if self.request.user.has_perm("complaints.view_queue"):
            return queryset
        return queryset.filter(submitted_by=self.request.user)

    def perform_create(self, serializer) -> None:
        serializer.save(submitted_by=self.request.user)

    @action(detail=True, methods=["post"])
    def triage(self, request, pk=None):
        if not request.user.has_perm("complaints.triage_complaint"):
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        complaint = self.get_object()
        payload = TriageSerializer(data=request.data)
        payload.is_valid(raise_exception=True)
        try:
            services.triage(
                complaint,
                payload.validated_data["category"],
                payload.validated_data["priority"],
                request.user,
            )
        except services.InvalidTransition as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        complaint.refresh_from_db()
        return Response(ComplaintSerializer(complaint).data)

    @action(detail=True, methods=["post"])
    def resolve(self, request, pk=None):
        if not request.user.has_perm("complaints.resolve_complaint"):
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        complaint = self.get_object()
        try:
            services.resolve(complaint, request.user, note=request.data.get("note", ""))
        except services.InvalidTransition as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        complaint.refresh_from_db()
        return Response(ComplaintSerializer(complaint).data)

    @action(detail=True, methods=["post"])
    def start_work(self, request, pk=None):
        if not request.user.has_perm("complaints.triage_complaint"):
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        complaint = self.get_object()
        try:
            services.transition(complaint, Status.IN_PROGRESS, request.user)
        except services.InvalidTransition as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        complaint.refresh_from_db()
        return Response(ComplaintSerializer(complaint).data)

    @action(detail=True, methods=["post"])
    def assign(self, request, pk=None):
        if not request.user.has_perm("complaints.assign_complaint"):
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        complaint = self.get_object()
        payload = AssignSerializer(data=request.data)
        payload.is_valid(raise_exception=True)
        services.assign(complaint, payload.validated_data["assignee"], request.user)
        complaint.refresh_from_db()
        return Response(ComplaintSerializer(complaint).data)

    @action(detail=True, methods=["post"])
    def mark_duplicate(self, request, pk=None):
        if not request.user.has_perm("complaints.mark_duplicate"):
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        complaint = self.get_object()
        payload = MarkDuplicateSerializer(data=request.data)
        payload.is_valid(raise_exception=True)
        try:
            services.mark_duplicate(complaint, payload.validated_data["canonical"], request.user)
        except services.InvalidTransition as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        complaint.refresh_from_db()
        return Response(ComplaintSerializer(complaint).data)

    @action(detail=True, methods=["post"])
    def close(self, request, pk=None):
        if not request.user.has_perm("complaints.resolve_complaint"):
            return Response({"detail": "Not permitted."}, status=status.HTTP_403_FORBIDDEN)

        complaint = self.get_object()
        try:
            services.transition(complaint, Status.CLOSED, request.user)
        except services.InvalidTransition as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        complaint.refresh_from_db()
        return Response(ComplaintSerializer(complaint).data)


@api_view(["GET"])
@permission_classes([AllowAny])
def healthz(request):
    """Reports which model version serves each domain.

    A deployment that lost its artifacts says so here rather than silently
    getting worse.
    """
    return Response({"status": "ok", "models": registry_status()})
