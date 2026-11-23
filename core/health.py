"""Unauthenticated liveness and readiness probes for a backend-only deployment.

They expose no tenant data, paths, versions or secrets: only coarse states, so a
load balancer, systemd watchdog or the integrating portal can check availability.
"""
from django.db import connection
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView


class LiveView(APIView):
    authentication_classes = []
    permission_classes = [permissions.AllowAny]
    throttle_classes = []

    def get(self, request):
        return Response({"status": "ok"})


class ReadyView(APIView):
    """200 when the database answers and every enabled model has finished warming."""
    authentication_classes = []
    permission_classes = [permissions.AllowAny]
    throttle_classes = []

    def get(self, request):
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            database = "ok"
        except Exception:
            database = "unavailable"
        from core.ml import preload
        from core.services import rag_admission
        models = dict(preload.status)
        models_ready = all(v in ("ready", "disabled") for v in models.values())
        ready = database == "ok" and models_ready
        return Response({"status": "ready" if ready else "not_ready", "database": database,
                         "models": models, "rag_busy": rag_admission.busy()},
                        status=200 if ready else 503)
