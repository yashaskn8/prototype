from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView

from .security import require_api_context
from .services.customer_answer import open_citation


class CustomerSourceView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        data = open_citation(request.query_params.get("citation", ""), require_api_context(request))
        response = Response(data)
        response["Cache-Control"] = "no-store, private"
        response["Vary"] = "Cookie"
        return response
