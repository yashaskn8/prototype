"""Provision and audit a localhost-only, synthetic Servy UI test schema.

The original company database, documents, Chroma index and model are read only.
Secrets and integrity manifests stay in an ACL-restricted local folder.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.servy_500_benchmark import private_folder, write_json, load_json

ORIGINAL_DB = "servy"
TEST_DB = "servy_ui_test_isolated"
TEST_USER = "servy_ui_test"
PRIVATE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations" / "servy_ui_test_isolated_20261031_01"
BEFORE = PRIVATE / "original_integrity_before.json"
AFTER = PRIVATE / "original_integrity_after.json"
SECRETS = PRIVATE / "ui_test_secrets.env"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError("Required original directory is unavailable.")
    return {str(path.relative_to(root)): file_sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def original_snapshot():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
    import django
    django.setup()
    from django.conf import settings
    from django.db import connection
    from django.contrib.auth.models import User
    from core.models import (Tenant, Customer, Asset, KnowledgeDocument, KnowledgeChunk,
                             CustomerInteraction, DiagnosticSession, ServiceCall)

    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE(), CURRENT_USER(), @@hostname")
        db_name, db_user, server = cursor.fetchone()
    if (db_name != ORIGINAL_DB or not db_user.startswith("servy_app@")
        or settings.SERVY_CUSTOMER_DATA_MODE != "verified_real_only"
        or Path(settings.CHROMA_PERSIST_DIR).resolve() != (ROOT / "chroma_db").resolve()
        or Path(settings.MEDIA_ROOT).resolve() != (ROOT / "media").resolve()):
        raise RuntimeError("Original-system integrity snapshot must use the unchanged company configuration.")
    counts = {model.__name__: model.objects.count() for model in
              (User, Tenant, Customer, Asset, KnowledgeDocument, KnowledgeChunk,
               CustomerInteraction, DiagnosticSession, ServiceCall)}
    media_root = Path(settings.MEDIA_ROOT).resolve()
    documents = {}
    for doc in KnowledgeDocument.objects.filter(tenant__slug="skydream").order_by("id"):
        record = {"content_sha256": hashlib.sha256((doc.content_text or "").encode()).hexdigest(),
                  "checksum_sha256": doc.checksum_sha256, "index_version": doc.index_version}
        if doc.file:
            path = Path(doc.file.path).resolve()
            if not path.is_relative_to(media_root) or not path.is_file():
                raise RuntimeError("A company document attachment is outside the original media root or missing.")
            record["attachment_sha256"] = file_sha(path)
        documents[str(doc.id)] = record
    reranker = Path(settings.SERVY_RERANKER_MODEL_PATH).resolve()
    if not reranker.is_dir():
        raise RuntimeError("Original V5 model folder is unavailable.")
    return {
        "database": {"name": db_name, "user": db_user, "server": server,
                     "counts": counts, "company_document_records": documents},
        "chroma_files": tree_manifest(settings.CHROMA_PERSIST_DIR),
        "reranker_files": tree_manifest(reranker),
        "configuration": {
            "mode": settings.SERVY_CUSTOMER_DATA_MODE,
            "embedding": settings.SERVY_SERVING_EMBEDDING_PROFILE,
            "retrieval": getattr(settings, "SERVY_RETRIEVAL_MODE", "hybrid"),
            "reranker_enabled": settings.SERVY_RERANKER_ENABLED,
            "reranker_pool": settings.SERVY_RERANKER_POOL_SIZE,
            "reranker_expected_sha256": settings.SERVY_RERANKER_EXPECTED_SHA256,
        },
    }


def snapshot(stage):
    if stage not in ("before", "after"):
        raise ValueError(stage)
    folder = private_folder(PRIVATE)
    target = BEFORE if stage == "before" else AFTER
    if target.exists():
        raise RuntimeError("Integrity snapshot already exists; it will not be overwritten.")
    value = original_snapshot()
    if stage == "after":
        if not BEFORE.is_file():
            raise RuntimeError("Original before snapshot is missing.")
        before = load_json(BEFORE)
        differences = [key for key in before if before[key] != value[key]]
        value["comparison"] = {"unchanged": not differences,
                               "changed_categories": differences}
    write_json(target, value)
    return {"stage": stage, "company_documents": len(value["database"]["company_document_records"]),
            "chroma_files": len(value["chroma_files"]), "model_files": len(value["reranker_files"]),
            "unchanged": value.get("comparison", {}).get("unchanged")}


def provision():
    if not BEFORE.is_file():
        raise RuntimeError("Original integrity baseline must be recorded first.")
    if SECRETS.exists():
        raise RuntimeError("Isolated secrets already exist; refusing to replace them.")
    from dotenv import dotenv_values
    import MySQLdb
    root_password = dotenv_values(ROOT / ".env.mysql_credentials").get("MYSQL_ROOT_PASSWORD")
    if not root_password:
        raise RuntimeError("Existing local administrator credential is unavailable.")
    db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root",
                         passwd=root_password, connect_timeout=5)
    try:
        cursor = db.cursor()
        cursor.execute("SELECT @@hostname, CURRENT_USER()")
        server, identity = cursor.fetchone()
        if server != load_json(BEFORE)["database"]["server"] or not identity.startswith("root@"):
            raise RuntimeError("Administrator is not connected to the baseline local MySQL instance.")
        cursor.execute("SHOW DATABASES LIKE %s", (TEST_DB,))
        if cursor.fetchone():
            raise RuntimeError("Proposed isolated schema already exists; refusing to overwrite it.")
        cursor.execute("SELECT COUNT(*) FROM mysql.user WHERE user=%s", (TEST_USER,))
        if cursor.fetchone()[0]:
            raise RuntimeError("Proposed isolated database account already exists; refusing to change it.")
        test_password = secrets.token_urlsafe(36)
        login_password = secrets.token_urlsafe(36)
        folder = private_folder(PRIVATE)
        payload = {
            "MYSQL_HOST": "127.0.0.1", "MYSQL_PORT": "3306", "MYSQL_DATABASE": TEST_DB,
            "MYSQL_USER": TEST_USER, "MYSQL_PASSWORD": test_password,
            "DJANGO_DEBUG": "1", "SERVY_CUSTOMER_DATA_MODE": "test_isolated",
            "SERVY_MEDIA_ROOT": str(folder / "media"),
            "SERVY_CHROMA_PERSIST_DIR": str(folder / "chroma"),
            "SERVY_SERVING_EMBEDDING_PROFILE": "bge-small",
            "SERVY_RETRIEVAL_MODE": "hybrid", "SERVY_RERANKER_ENABLED": "1",
            "SERVY_RERANKER_REQUIRED": "1", "SERVY_RERANKER_POOL_SIZE": "5",
            "SERVY_LLM_PROVIDER": "extractive", "SERVY_ALLOW_REMOTE_LLM": "0",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "SERVY_UI_TEST_LOGIN_PASSWORD": login_password,
        }
        with SECRETS.open("x", encoding="utf-8") as stream:
            for key, value in payload.items():
                stream.write(f"{key}={value}\n")
        (folder / "media").mkdir(exist_ok=False)
        (folder / "chroma").mkdir(exist_ok=False)
        cursor.execute(f"CREATE DATABASE `{TEST_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
        cursor.execute(f"CREATE USER '{TEST_USER}'@'127.0.0.1' IDENTIFIED BY %s", (test_password,))
        cursor.execute(f"GRANT ALL PRIVILEGES ON `{TEST_DB}`.* TO '{TEST_USER}'@'127.0.0.1'")
        db.commit()
    finally:
        db.close()
    verify_test_connection()
    return {"test_schema_created": True, "test_user_created": True,
            "credentials_private": True, "original_user_unchanged": True}


def load_test_env():
    from dotenv import dotenv_values
    if not SECRETS.is_file():
        raise RuntimeError("Private isolated configuration is missing.")
    values = dotenv_values(SECRETS)
    for key, value in values.items():
        if value is None:
            raise RuntimeError("Private isolated configuration is incomplete.")
        os.environ[key] = value
    return values


def verify_test_connection():
    from dotenv import dotenv_values
    import MySQLdb
    values = dotenv_values(SECRETS)
    db = MySQLdb.connect(host="127.0.0.1", port=3306, user=TEST_USER,
                         passwd=values["MYSQL_PASSWORD"], db=TEST_DB, connect_timeout=5)
    try:
        cursor = db.cursor()
        cursor.execute("SELECT DATABASE(), CURRENT_USER()")
        name, identity = cursor.fetchone()
        cursor.execute("SHOW GRANTS FOR CURRENT_USER")
        grants = [row[0] for row in cursor.fetchall()]
        if name != TEST_DB or not identity.startswith(f"{TEST_USER}@"):
            raise RuntimeError("Isolated connection identity does not match the expected account.")
        if not any(f"ON `{TEST_DB}`.*" in grant for grant in grants):
            raise RuntimeError("Isolated account lacks its test-schema grant.")
        for grant in grants:
            if " ON *.* " in grant and "GRANT USAGE" not in grant:
                raise RuntimeError("Isolated account has global privileges.")
            if " ON `" in grant and f"ON `{TEST_DB}`.*" not in grant:
                raise RuntimeError("Isolated account has access to another schema.")
        cursor.execute("SHOW DATABASES")
        visible = {row[0] for row in cursor.fetchall()}
        if ORIGINAL_DB in visible:
            raise RuntimeError("Original company schema is visible to isolated account.")
        return {"database": name, "restricted_user": identity,
                "original_schema_visible": False, "grant_statements": len(grants)}
    finally:
        db.close()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command == "before":
        result = snapshot("before")
    elif command == "after":
        result = snapshot("after")
    elif command == "provision":
        result = provision()
    elif command == "check":
        result = verify_test_connection()
    else:
        raise SystemExit("Usage: servy_ui_test_isolation.py before|provision|check|after")
    print(json.dumps(result))
