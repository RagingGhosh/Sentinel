from django.urls import include, path
from rest_framework.routers import DefaultRouter

from complaints.api import ComplaintViewSet, healthz

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

# Task 13 appends server-rendered HTML routes here with urlpatterns += [...].
