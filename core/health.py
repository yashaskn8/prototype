"""Unauthenticated liveness and readiness probes for a backend-only deployment.

They expose no tenant data, paths, versions or secrets: only coarse states, so a
load balancer, systemd watchdog or the integrating portal can check availability.
"""
import time

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


_MIGRATIONS_CACHE = {"checked": 0.0, "pending": None}


def _pending_migrations() -> bool:
    """True when the database schema is behind the code (cached for 60 s)."""
    now = time.monotonic()
    if _MIGRATIONS_CACHE["pending"] is None or now - _MIGRATIONS_CACHE["checked"] > 60:
        from django.db.migrations.executor import MigrationExecutor
        executor = MigrationExecutor(connection)
        _MIGRATIONS_CACHE["pending"] = bool(executor.migration_plan(executor.loader.graph.leaf_nodes()))
        _MIGRATIONS_CACHE["checked"] = now
    return _MIGRATIONS_CACHE["pending"]


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
        schema = "unknown"
        if database == "ok":
            try:
                schema = "migrations_pending" if _pending_migrations() else "ok"
            except Exception:
                schema = "unknown"
        from core.ml import preload
        models = dict(preload.status)
        models_ready = all(v in ("ready", "disabled") for v in models.values())
        index = preload.index_state if models.get("bge") != "disabled" else "disabled"
        ready = (database == "ok" and schema == "ok" and models_ready and index in ("ready", "disabled"))
        # Coarse states only: no paths, versions, counts or live usage (rag_busy).
        return Response({"status": "ready" if ready else "not_ready", "database": database,
                         "schema": schema, "models": models, "index": index},
                        status=200 if ready else 503)
