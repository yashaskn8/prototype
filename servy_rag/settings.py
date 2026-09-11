from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
_default_secret = "servy-rag-local-demo-secret-key-change-me"
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", _default_secret)
if not DEBUG and SECRET_KEY == _default_secret:
    raise RuntimeError("Set DJANGO_SECRET_KEY before running with DJANGO_DEBUG=0")

ALLOWED_HOSTS = [h.strip() for h in os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if h.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "core.apps.CoreConfig",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "servy_rag.urls"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
        "core.context_processors.tenant_context",
    ]},
}]
WSGI_APPLICATION = "servy_rag.wsgi.application"
ASGI_APPLICATION = "servy_rag.asgi.application"

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3", "OPTIONS": {"timeout": 20}}}
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/accounts/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/accounts/login/"

# Security defaults suitable for local development and ready for hardening in production.
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = False
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SECURE_REFERRER_POLICY = "same-origin"
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"
DATA_UPLOAD_MAX_MEMORY_SIZE = 12 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 3 * 1024 * 1024
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = os.getenv("DJANGO_SECURE_SSL_REDIRECT", "1") == "1"
    SECURE_HSTS_SECONDS = int(os.getenv("DJANGO_SECURE_HSTS_SECONDS", "3600"))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True

# CORS and CSRF trusted origins for React dev server and production
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Django REST Framework configuration
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "customer_rag": "20/minute",
        "engineer_rag": "30/minute",
        "login": "10/minute",
        "upload": "10/hour",
    },
}

CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Local RAG / LLM settings. Docker is not required.
SERVY_LLM_PROVIDER = os.getenv("SERVY_LLM_PROVIDER", "extractive")  # extractive|auto|ollama
SERVY_OLLAMA_URL = os.getenv("SERVY_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
SERVY_OLLAMA_MODEL = os.getenv("SERVY_OLLAMA_MODEL", "qwen2.5:3b")
SERVY_OLLAMA_TIMEOUT = int(os.getenv("SERVY_OLLAMA_TIMEOUT", "30"))
SERVY_RAG_TOP_K = int(os.getenv("SERVY_RAG_TOP_K", "6"))
SERVY_RAG_MIN_SCORE = float(os.getenv("SERVY_RAG_MIN_SCORE", "0.055"))
SERVY_MAX_KB_UPLOAD_BYTES = int(os.getenv("SERVY_MAX_KB_UPLOAD_BYTES", str(10 * 1024 * 1024)))
SERVY_MAX_IMPORT_BYTES = int(os.getenv("SERVY_MAX_IMPORT_BYTES", str(5 * 1024 * 1024)))
SERVY_MAX_DOCUMENT_TEXT_CHARS = int(os.getenv("SERVY_MAX_DOCUMENT_TEXT_CHARS", "500000"))
SERVY_MAX_PDF_PAGES = int(os.getenv("SERVY_MAX_PDF_PAGES", "120"))
SERVY_MAX_CSV_ROWS = int(os.getenv("SERVY_MAX_CSV_ROWS", "10000"))
SERVY_MAX_DOCX_UNCOMPRESSED_BYTES = int(os.getenv("SERVY_MAX_DOCX_UNCOMPRESSED_BYTES", str(40 * 1024 * 1024)))
SERVY_MAX_IMPORT_ROWS = int(os.getenv("SERVY_MAX_IMPORT_ROWS", "10000"))
SERVY_MAX_XLSX_UNCOMPRESSED_BYTES = int(os.getenv("SERVY_MAX_XLSX_UNCOMPRESSED_BYTES", str(30 * 1024 * 1024)))
SERVY_RAG_RATE_LIMIT_PER_MINUTE = int(os.getenv("SERVY_RAG_RATE_LIMIT_PER_MINUTE", "20"))
SERVY_ALLOW_REMOTE_LLM = os.getenv("SERVY_ALLOW_REMOTE_LLM", "0") == "1"
