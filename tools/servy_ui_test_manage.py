"""Run a small allowlist of Django commands against the isolated UI test schema."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.servy_ui_test_isolation import PRIVATE, TEST_DB, TEST_USER, load_test_env, verify_test_connection


def main(argv):
    allowed = {"migrate", "showmigrations", "check", "check_mysql_ready",
               "check_rag_serving", "seed_ui_test", "runserver"}
    if not argv or argv[0] not in allowed:
        raise SystemExit("Only isolated setup and server commands are permitted.")
    if argv[0] == "runserver" and (len(argv) < 2 or argv[1] != "127.0.0.1:8001"):
        raise SystemExit("Isolated Django server must bind to 127.0.0.1:8001.")
    values = load_test_env()
    verify_test_connection()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["DJANGO_SETTINGS_MODULE"] = "servy_rag.settings"
    from django.conf import settings
    if (settings.DATABASES["default"]["ENGINE"] != "django.db.backends.mysql"
        or settings.DATABASES["default"]["NAME"] != TEST_DB
        or settings.DATABASES["default"]["USER"] != TEST_USER
        or settings.SERVY_CUSTOMER_DATA_MODE != "test_isolated"
        or not settings.DEBUG
        or settings.SERVY_ALLOW_REMOTE_LLM
        or Path(settings.MEDIA_ROOT).resolve() != (PRIVATE / "media").resolve()
        or Path(settings.CHROMA_PERSIST_DIR).resolve() != (PRIVATE / "chroma").resolve()
        or settings.SERVY_SERVING_EMBEDDING_PROFILE != "bge-small"
        or not settings.SERVY_RERANKER_ENABLED
        or settings.SERVY_RERANKER_POOL_SIZE != 5
        or values.get("SERVY_LLM_PROVIDER") != "extractive"):
        raise SystemExit("Isolated database, storage, local inference, or serving configuration failed closed.")
    from django.core.management import execute_from_command_line
    execute_from_command_line(["manage.py", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
