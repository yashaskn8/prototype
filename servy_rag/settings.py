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
    # Keep Django's app templates for the admin and DRF; the retired
    # server-rendered prototype UI no longer ships project-level templates.
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
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

    # Isolated local databases (servy_dev / test_servy_dev) are selected with
    # SERVY_MYSQL_ENV_FILE. The file must supply all five connection values; they
    # replace anything .env loaded, so production credentials can never be used
    # silently. The password is never logged.
    SERVY_MYSQL_ENV_FILE = os.getenv("SERVY_MYSQL_ENV_FILE", "").strip()
    if SERVY_MYSQL_ENV_FILE:
        from django.core.exceptions import ImproperlyConfigured as _ImproperlyConfigured
        _env_path = Path(SERVY_MYSQL_ENV_FILE)
        if not _env_path.is_absolute():
            _env_path = BASE_DIR / _env_path
        if not _env_path.is_file():
            raise _ImproperlyConfigured(f"SERVY_MYSQL_ENV_FILE not found: {_env_path.name}")
        _values = {}
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            if "=" in _line and not _line.lstrip().startswith("#"):
                _k, _v = _line.split("=", 1)
                _values[_k.strip()] = _v.strip()
        _required = ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD")
        _missing = [k for k in _required if not _values.get(k)]
        if _missing:
            raise _ImproperlyConfigured(f"{_env_path.name} is missing: {', '.join(_missing)}")
        MYSQL_HOST = _values["MYSQL_HOST"]
        MYSQL_PORT = int(_values["MYSQL_PORT"])
        MYSQL_DATABASE = _values["MYSQL_DATABASE"]
        MYSQL_USER = _values["MYSQL_USER"]
        MYSQL_PASSWORD = _values["MYSQL_PASSWORD"]
        if MYSQL_DATABASE in ("servy", "test_servy") or MYSQL_USER == "servy_app":
            raise _ImproperlyConfigured("SERVY_MYSQL_ENV_FILE must not select the protected servy database or account.")
        del _values

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
                # The restricted test account may only use the pre-created
                # test_servy_dev, so tests run in that database (use --keepdb).
                **({"NAME": "test_servy_dev"} if MYSQL_DATABASE == "test_servy_dev" else {}),
            },
        }
    }
elif SERVY_DB_ENGINE == "sqlite3":
    # Only for disposable automated tests: an in-memory database that never
    # persists. SQLite is never a runtime database or fallback, and the legacy
    # db.sqlite3 file (preserved, possibly unique company records) is never opened.
    import sys as _db_sys
    if len(_db_sys.argv) < 2 or _db_sys.argv[1] != "test":
        raise RuntimeError("SQLite is allowed only for disposable test runs; Servy requires MySQL.")
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": ":memory:",
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
STATICFILES_DIRS = []  # Project-specific Bootstrap assets belonged to the retired Django UI.
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = Path(os.getenv("SERVY_MEDIA_ROOT", str(BASE_DIR / "media"))).resolve()
# The isolated development database keeps its uploaded files out of the
# repository media folder (approved manuals and existing documents live there).
if globals().get("MYSQL_DATABASE") == "servy_dev" and MEDIA_ROOT == (BASE_DIR / "media").resolve():
    from django.core.exceptions import ImproperlyConfigured as _MediaConfigError
    raise _MediaConfigError("servy_dev requires SERVY_MEDIA_ROOT to point to a separate development media folder.")
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/app/"
LOGIN_REDIRECT_URL = "/app/"
LOGOUT_REDIRECT_URL = "/app/"

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

PRODUCTION_CHROMA_DIR = (BASE_DIR / "chroma_db").resolve()
CHROMA_PERSIST_DIR = Path(os.getenv("SERVY_CHROMA_PERSIST_DIR", str(PRODUCTION_CHROMA_DIR))).resolve()

# Backend tests never touch the production index: they get a fresh temporary
# directory by default, and an explicit production path is refused.
import sys as _sys
if len(_sys.argv) > 1 and _sys.argv[1] == "test":
    if not os.getenv("SERVY_CHROMA_PERSIST_DIR"):
        import tempfile as _tempfile
        CHROMA_PERSIST_DIR = Path(_tempfile.mkdtemp(prefix="servy_test_chroma_")).resolve()
    elif CHROMA_PERSIST_DIR == PRODUCTION_CHROMA_DIR:
        from django.core.exceptions import ImproperlyConfigured
        raise ImproperlyConfigured(
            "Refusing to run tests against the production ChromaDB directory; "
            "unset SERVY_CHROMA_PERSIST_DIR or point it at a disposable copy.")

# Local RAG / LLM settings. Docker is not required.
SERVY_LLM_PROVIDER = os.getenv("SERVY_LLM_PROVIDER", "extractive").lower().strip()  # extractive|auto|ollama|llama_cpp
# Optional strict local generation: do not silently substitute extractive answers when a provider is required.
SERVY_LLM_REQUIRE_OLLAMA = os.getenv("SERVY_LLM_REQUIRE_OLLAMA", "0") == "1"
SERVY_OLLAMA_URL = os.getenv("SERVY_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
SERVY_OLLAMA_MODEL = os.getenv("SERVY_OLLAMA_MODEL", "qwen2.5:3b")
SERVY_OLLAMA_TIMEOUT = int(os.getenv("SERVY_OLLAMA_TIMEOUT", "30"))
# Customer-facing Qwen wording is a separate source-span-bound postprocessor.
# It is disabled by default; the deterministic safety and citation builder stays authoritative.
SERVY_CUSTOMER_ANSWER_GENERATION = os.getenv(
    "SERVY_CUSTOMER_ANSWER_GENERATION", "disabled"
).lower().strip()
SERVY_CUSTOMER_GENERATION_URL = os.getenv(
    "SERVY_CUSTOMER_GENERATION_URL", "http://127.0.0.1:11434/api/chat"
)
SERVY_CUSTOMER_GENERATION_MODEL = os.getenv(
    "SERVY_CUSTOMER_GENERATION_MODEL", SERVY_OLLAMA_MODEL
)
try:
    SERVY_CUSTOMER_GENERATION_TIMEOUT = min(
        120.0, max(1.0, float(os.getenv("SERVY_CUSTOMER_GENERATION_TIMEOUT", "45")))
    )
except (TypeError, ValueError):
    SERVY_CUSTOMER_GENERATION_TIMEOUT = 45.0

try:
    SERVY_CUSTOMER_GENERATION_MAX_CLAIMS = min(
        6, max(0, int(os.getenv("SERVY_CUSTOMER_GENERATION_MAX_CLAIMS", "6")))
    )
except (TypeError, ValueError):
    SERVY_CUSTOMER_GENERATION_MAX_CLAIMS = 6
# Qwen GGUF through llama.cpp's OpenAI-compatible local server. Staff-mode RAG
# may opt into this separate generator; customer mode uses the source-span path above.
SERVY_LLAMA_CPP_URL = os.getenv(
    "SERVY_LLAMA_CPP_URL", "http://127.0.0.1:9931/v1/chat/completions"
)
SERVY_LLAMA_CPP_MODEL = os.getenv(
    "SERVY_LLAMA_CPP_MODEL", "gguf/Qwen3.5-2B-GGUF:Q4_K_M"
)
SERVY_LLAMA_CPP_TIMEOUT = float(os.getenv("SERVY_LLAMA_CPP_TIMEOUT", "90"))
SERVY_LLAMA_CPP_MAX_TOKENS = int(os.getenv("SERVY_LLAMA_CPP_MAX_TOKENS", "512"))
SERVY_LLM_REQUIRE_LLAMA_CPP = os.getenv("SERVY_LLM_REQUIRE_LLAMA_CPP", "0") == "1"
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
SERVY_ALLOW_REMOTE_LLM = os.getenv("SERVY_ALLOW_REMOTE_LLM", "0") == "1"

# Embedding model: the approved BGE-small profile is the only document and query
# embedding model. MiniLM has been retired; there is no MiniLM profile or fallback.
SERVY_USE_EXPERIMENTAL_RAG = os.getenv("SERVY_USE_EXPERIMENTAL_RAG", "0") == "1"
# Default to the evaluated BGE-small profile, using the separate company BGE
# collection for real tenants. Verify the local model and index on each host.
SERVY_SERVING_EMBEDDING_PROFILE = os.getenv("SERVY_SERVING_EMBEDDING_PROFILE", "bge-small").strip().lower()
if "minilm" in SERVY_SERVING_EMBEDDING_PROFILE:
    from django.core.exceptions import ImproperlyConfigured
    raise ImproperlyConfigured("MiniLM has been retired; SERVY_SERVING_EMBEDDING_PROFILE must be a BGE profile.")
# Retrieval mode actually served. It was previously never declared here, so
# SERVY_RETRIEVAL_MODE in the environment was ignored and serving silently
# reported "hybrid". An unknown value is a configuration error, not a default.
SERVY_RETRIEVAL_MODES = ("hybrid", "sparse_only", "dense_only")
SERVY_RETRIEVAL_MODE = os.getenv("SERVY_RETRIEVAL_MODE", "hybrid").strip().lower()
if SERVY_RETRIEVAL_MODE not in SERVY_RETRIEVAL_MODES:
    from django.core.exceptions import ImproperlyConfigured
    raise ImproperlyConfigured(
        "SERVY_RETRIEVAL_MODE must be one of " + ", ".join(SERVY_RETRIEVAL_MODES)
        + f"; got {SERVY_RETRIEVAL_MODE!r}."
    )
# Verified from Hugging Face API on 2026-10-14 (BAAI/bge-small-en-v1.5, MIT).
SERVY_BGE_SMALL_REVISION = os.getenv("SERVY_BGE_SMALL_REVISION", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
SERVY_EMBEDDING_DEVICE = os.getenv("SERVY_EMBEDDING_DEVICE", "cpu")
SERVY_EMBEDDING_BATCH_SIZE = int(os.getenv("SERVY_EMBEDDING_BATCH_SIZE", "8"))
# Experimental Qwen3-Embedding-0.6B dense retrieval: off by default.
# Requires the full local checkpoint, a fully verified separate 1024-D index,
# and SERVY_USE_EXPERIMENTAL_RAG=1 before any request can opt in.
SERVY_QWEN3_MODEL_PATH = os.getenv("SERVY_QWEN3_MODEL_PATH", "").strip()
SERVY_QWEN3_ENABLED = os.getenv("SERVY_QWEN3_ENABLED", "0") == "1"
SERVY_QWEN3_INDEX_VERIFIED = os.getenv("SERVY_QWEN3_INDEX_VERIFIED", "0") == "1"
SERVY_QWEN3_DENSE_GATE = float(os.getenv("SERVY_QWEN3_DENSE_GATE", "0.35"))
# Separate, experimental dense-first ablation; OFF until measured on independent
# questions. Preserves authorized sparse candidates as a backfill.
SERVY_QWEN3_DENSE_FIRST = os.getenv("SERVY_QWEN3_DENSE_FIRST", "0") == "1"

SERVY_RERANKER_ENABLED = os.getenv("SERVY_RERANKER_ENABLED", "0") == "1"
# Explicit strict deployment mode: block application startup if pinned V5 inference fails.
SERVY_RERANKER_REQUIRED = os.getenv("SERVY_RERANKER_REQUIRED", "0") == "1"
# Optional V5 reranker: local checkpoint only, never the Qwen3 embedding model.
SERVY_RERANKER_MODEL_PATH = os.getenv("SERVY_RERANKER_MODEL_PATH", "").strip()
SERVY_RERANKER_EXPECTED_SHA256 = os.getenv("SERVY_RERANKER_EXPECTED_SHA256", "").strip()
SERVY_RERANKER_DEVICE = os.getenv("SERVY_RERANKER_DEVICE", SERVY_EMBEDDING_DEVICE)
# The evaluated BGE+V5 dev configuration used a head pool of five candidates.
SERVY_RERANKER_POOL_SIZE = int(os.getenv("SERVY_RERANKER_POOL_SIZE", "5"))
# Experimental evidence + section-intent reranking. OFF by default; isolate for evaluation.
SERVY_EXPERIMENTAL_SECTION_AWARE_RERANK = os.getenv("SERVY_EXPERIMENTAL_SECTION_AWARE_RERANK", "0") == "1"
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
# Tickets, conversations, attachments and technician updates are official MySQL
# records; there is no SQLite ticket workspace.
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

# --- Access-code demo gateways (local development only) --------------------------------
# SERVY_DEMO_GATEWAY=customer|technician serves the official APIs to the existing
# access-code demos from the isolated servy_dev database. Unset for every other use.
SERVY_DEMO_GATEWAY = os.getenv("SERVY_DEMO_GATEWAY", "").strip().lower()
SERVY_DEMO_EXPECTED_DATABASE = os.getenv("SERVY_DEMO_EXPECTED_DATABASE", "servy_dev")
SERVY_DEMO_EXPECTED_USER = os.getenv("SERVY_DEMO_EXPECTED_USER", "servy_dev_app")
SERVY_DEMO_ACCESS_FILE = os.getenv("SERVY_DEMO_ACCESS_FILE", "")
SERVY_TECHNICIAN_ACCESS_FILE = os.getenv("SERVY_TECHNICIAN_ACCESS_FILE", "")
if SERVY_DEMO_GATEWAY:
    from django.core.exceptions import ImproperlyConfigured as _GatewayConfigError
    if SERVY_DEMO_GATEWAY not in ("customer", "technician"):
        raise _GatewayConfigError("SERVY_DEMO_GATEWAY must be 'customer' or 'technician'.")
    if globals().get("MYSQL_DATABASE") != "servy_dev" or not os.getenv("SERVY_MYSQL_ENV_FILE"):
        raise _GatewayConfigError("The demo gateways run only on servy_dev selected through SERVY_MYSQL_ENV_FILE.")
    _access = SERVY_DEMO_ACCESS_FILE if SERVY_DEMO_GATEWAY == "customer" else SERVY_TECHNICIAN_ACCESS_FILE
    if not _access or not Path(_access).is_file():
        raise _GatewayConfigError("The demo gateway needs its existing access-code file.")
    ROOT_URLCONF = f"servy_rag.demo_{SERVY_DEMO_GATEWAY}_urls"
    # The gateways run on the threaded development server, which serves every
    # request in a new thread. Persistent connections would outlive those threads
    # and exhaust MySQL (max_connections); close each request's connection instead.
    DATABASES["default"]["CONN_MAX_AGE"] = 0
