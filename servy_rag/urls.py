from django.conf import settings
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.http import FileResponse, Http404, HttpResponse
from django.urls import include, path


def serve_react_app(request, path=""):
    dist_dir = settings.BASE_DIR / "frontend" / "dist"
    if path:
        target = (dist_dir / path).resolve()
        # Path traversal guard
        if str(target).startswith(str(dist_dir.resolve())) and target.exists() and not target.is_dir():
            return FileResponse(target.open("rb"))
    index_file = dist_dir / "index.html"
    if index_file.exists():
        return HttpResponse(index_file.read_text(encoding="utf-8"), content_type="text/html")
    raise Http404("React frontend has not been built yet. Run 'npm run build' inside 'frontend/'.")


urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/login/", auth_views.LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("accounts/logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("api/", include("core.api_urls")),
    path("app/", serve_react_app, name="react_app"),
    path("app/<path:path>", serve_react_app, name="react_app_path"),
    path("", include("core.urls")),
]
