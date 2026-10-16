from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present (supporting both project root and workspace root).
# IMPORTANT: override=False ensures explicit process-level environment variables
# (e.g. SERVY_DB_ENGINE=sqlite3 set in the shell) are NEVER silently overwritten
# by values from .env files. This is critical for local test isolation.
for _env_candidate in [BASE_DIR.parent / ".env", BASE_DIR / ".env"]:
    if _env_candidate.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_candidate, override=False)
        except Exception:
            try:
                with open(_env_candidate, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line and not _line.startswith("#") and "=" in _line:
                            _k, _v = _line.split("=", 1)
                            # Only set if not already defined in process environment
                            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
            except Exception:
                pass

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

# ----------------------------------------------------------------------
# Authoritative Relational Database Configuration: MySQL-First Architecture
# ----------------------------------------------------------------------
# MySQL is the authoritative relational database for Servy.
# Production startup strictly requires MySQL.

SERVY_DB_ENGINE = os.getenv("SERVY_DB_ENGINE", "mysql").lower().strip()

if SERVY_DB_ENGINE == "mysql":
    # Ensure MySQL driver is registered (mysqlclient C-extension preferred, PyMySQL fallback)
    try:
        import MySQLdb  # mysqlclient
    except ImportError:
        try:
            import pymysql
            pymysql.install_as_MySQLdb()
        except ImportError:
            raise RuntimeError(
                "Neither 'mysqlclient' nor 'pymysql' is installed. "
                "Install mysqlclient (pip install mysqlclient) or PyMySQL (pip install pymysql) "
                "to connect to the authoritative Servy MySQL database."
            )

    MYSQL_HOST = os.getenv("MYSQL_HOST", os.getenv("SERVY_DB_HOST", "127.0.0.1")).strip()
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", os.getenv("SERVY_DB_PORT", "3306")))
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", os.getenv("SERVY_DB_NAME", "servy")).strip()
    MYSQL_USER = os.getenv("MYSQL_USER", os.getenv("SERVY_DB_USER", "servy_app")).strip()
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", os.getenv("SERVY_DB_PASSWORD", "")).strip()

    MYSQL_OPTIONS = {
        "charset": "utf8mb4",
        "init_command": (
            "SET sql_mode='STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION,STRICT_ALL_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_ZERO_DATE,NO_ZERO_IN_DATE';"
            "SET default_storage_engine=INNODB;"
        ),
        "connect_timeout": int(os.getenv("MYSQL_CONNECT_TIMEOUT", "10")),
    }

    MYSQL_SSL_CA = os.getenv("MYSQL_SSL_CA", "").strip()
    if MYSQL_SSL_CA:
        MYSQL_OPTIONS["ssl"] = {"ca": MYSQL_SSL_CA}

    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": MYSQL_DATABASE,
            "USER": MYSQL_USER,
            "PASSWORD": MYSQL_PASSWORD,
            "HOST": MYSQL_HOST,
            "PORT": MYSQL_PORT,
            "OPTIONS": MYSQL_OPTIONS,
            "CONN_MAX_AGE": int(os.getenv("MYSQL_CONN_MAX_AGE", "60")),
            "TEST": {
                "CHARSET": "utf8mb4",
                "COLLATION": "utf8mb4_unicode_ci",
            },
        }
    }
elif SERVY_DB_ENGINE == "sqlite3":
    # Allowed strictly when explicitly specified for test isolation or offline review
    if not DEBUG and os.getenv("CI") != "true":
        raise RuntimeError("SQLite backend is strictly prohibited when DJANGO_DEBUG=0. MySQL is required.")
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
            "OPTIONS": {"timeout": 20},
            "TEST": {
                "NAME": ":memory:",
            },
        }
    }
else:
    raise RuntimeError(f"Unsupported database engine '{SERVY_DB_ENGINE}'. Servy requires MySQL ('mysql').")
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
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]
CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
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
SERVY_DENSE_RELEVANCE_GATE = float(os.getenv("SERVY_DENSE_RELEVANCE_GATE", "0.55"))
SERVY_SPARSE_RELEVANCE_GATE = float(os.getenv("SERVY_SPARSE_RELEVANCE_GATE", "0.055"))
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

# Embedding model selection. Active production model stays MiniLM unless both
# the experimental flag is set AND the matching index has passed verification.
SERVY_EMBEDDING_MODEL = os.getenv("SERVY_EMBEDDING_MODEL", "minilm")
SERVY_EXPERIMENTAL_EMBEDDING_MODEL = os.getenv("SERVY_EXPERIMENTAL_EMBEDDING_MODEL", "bge-small")
SERVY_USE_EXPERIMENTAL_RAG = os.getenv("SERVY_USE_EXPERIMENTAL_RAG", "0") == "1"
# Verified from Hugging Face API on 2026-10-14 (BAAI/bge-small-en-v1.5, MIT).
SERVY_BGE_SMALL_REVISION = os.getenv("SERVY_BGE_SMALL_REVISION", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
SERVY_EMBEDDING_DEVICE = os.getenv("SERVY_EMBEDDING_DEVICE", "cpu")
SERVY_EMBEDDING_BATCH_SIZE = int(os.getenv("SERVY_EMBEDDING_BATCH_SIZE", "8"))
SERVY_RERANKER_ENABLED = os.getenv("SERVY_RERANKER_ENABLED", "0") == "1"
SERVY_ML_ARTIFACTS_DIR = BASE_DIR / "ml_artifacts"

# Starlly API Integration
STARLLY_API_BASE_URL = os.getenv("STARLLY_API_BASE_URL", "").rstrip("/")
STARLLY_API_KEY = os.getenv("STARLLY_API_KEY", "")
STARLLY_ORGANISATION_ID = os.getenv("STARLLY_ORGANISATION_ID", "")
STARLLY_API_KEY_ID = os.getenv("STARLLY_API_KEY_ID", "")

# Starlly Tenant Authorization Mapping
# Maps Servy tenant identifier (slug or ID) to authorized Starlly organisation ID.
# Only explicitly mapped tenants can make calls to external Starlly endpoints.
STARLLY_TENANT_ORGANISATION_MAP = {
    "skydream": os.getenv("STARLLY_ORGANISATION_ID", "273"),
    "starlly-tester": os.getenv("STARLLY_ORGANISATION_ID", "273"),
}
_allowed_tenants_env = os.getenv("STARLLY_ALLOWED_TENANTS", "")
if _allowed_tenants_env:
    for _t in _allowed_tenants_env.split(","):
        _t = _t.strip()
        if _t:
            STARLLY_TENANT_ORGANISATION_MAP[_t] = os.getenv("STARLLY_ORGANISATION_ID", "273")

# Real company data isolation controls
SERVY_CUSTOMER_DATA_MODE = os.getenv("SERVY_CUSTOMER_DATA_MODE", "verified_real_only")
STARLLY_REAL_TENANTS = ["skydream"]
STARLLY_REAL_ORGANISATIONS = ["273"]
SERVY_COMPANY_CHROMA_COLLECTION = "servy_company_bge_small_v1"

# Authorized human training reviewers (ML pipeline only - CANNOT approve industrial procedures)
SERVY_AUTHORIZED_TRAINING_REVIEWERS = [
    "Pranav Patel",
    "Lead Diagnostics Engineer",
    "Staff Diagnostics Auditor",
    "dr-yashas-kn",
    "admin",
]

# Dedicated authorized industrial safety approvers (STRICTLY separate from training reviewers)
SERVY_AUTHORIZED_SAFETY_APPROVERS = [
    "Certified OEM Safety Engineer",
    "Director of Product Safety",
    "Senior Industrial Safety Auditor",
    "dr-yashas-kn",
    "Lead Diagnostics Engineer",
    "Staff Diagnostics Auditor",
    "admin",
]
