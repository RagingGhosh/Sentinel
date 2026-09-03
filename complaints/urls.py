from django.urls import include, path
from rest_framework.routers import DefaultRouter

from complaints.api import ComplaintViewSet, healthz
from complaints.views import ComplaintListView, QueueView, detail, submit

# basename is "api-complaint", not the brief's "complaint": Task 13 registers
# server-rendered HTML views under the names "complaint-list"/"complaint-detail",
# and reverse() must resolve those unambiguously rather than depending on
# urlpattern registration order.
router = DefaultRouter()
router.register("complaints", ComplaintViewSet, basename="api-complaint")

urlpatterns = [
    path("api/", include(router.urls)),
    path("healthz", healthz, name="healthz"),
]

urlpatterns += [
    path("", ComplaintListView.as_view(), name="complaint-list"),
    path("complaints/submit/", submit, name="complaint-submit"),
    path("complaints/queue/", QueueView.as_view(), name="complaint-queue"),
    path("complaints/<int:pk>/", detail, name="complaint-detail"),
]
