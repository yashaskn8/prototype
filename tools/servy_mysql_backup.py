"""MySQL-era backup, verification and restore for the Servy development stack.

One backup directory (outside the repository) contains:

  mysql/<database>.sql   consistent mysqldump (--single-transaction)
  chroma/                copy of the vector index directory
  media/                 copy of approved source documents and attachments
  manifest.json          SHA-256 of every file, row counts of key tables,
                         model identities, git commit, creation time

Secrets are never copied: no .env file is included, and the database password
is passed to mysqldump/mysql only through a temporary options file that is
deleted immediately. Restore refuses the production database, refuses to
overwrite its own source database, verifies every hash first, and requires the
target database name to be confirmed explicitly.

Usage
  backup  --env-file .env.mysql.servy_dev --chroma <dir> --media <dir> --out <backup-root>
  verify  --backup <dir>
  restore --backup <dir> --env-file .env.mysql.servy_dev_test --confirm-target test_servy_dev
          [--chroma-to <empty dir>] [--media-to <empty dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MYSQL_BIN = Path(os.getenv("SERVY_MYSQL_BIN", r"C:\mysql\mysql-8.0.41-winx64\bin"))
PROTECTED_DATABASES = {"servy", "test_servy"}
# Hard limits (seconds). subprocess.run kills the child process when one expires.
QUERY_TIMEOUT = int(os.getenv("SERVY_BACKUP_QUERY_TIMEOUT", "60"))
DUMP_TIMEOUT = int(os.getenv("SERVY_BACKUP_DUMP_TIMEOUT", "900"))
KEY_TABLES = ("core_tenant", "core_customer", "core_asset", "core_knowledgedocument", "core_knowledgechunk",
              "core_customerconversation", "core_customermessage", "core_servicecall")


def read_env(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    missing = [k for k in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD") if not values.get(k)]
    if missing:
        raise SystemExit(f"{path.name} is missing {', '.join(missing)}")
    return values


class _Options:
    """Temporary [client] options file so the password never appears on a command line."""

    def __init__(self, env):
        self.env = env

    def __enter__(self):
        handle, self.path = tempfile.mkstemp(prefix="servy-my-", suffix=".cnf")
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write("[client]\n")
            f.write(f"host={self.env['MYSQL_HOST']}\nport={self.env['MYSQL_PORT']}\n")
            f.write(f"user={self.env['MYSQL_USER']}\npassword={self.env['MYSQL_PASSWORD']}\n")
        return self.path

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file()}


def mysql_query(env, sql: str) -> list[list[str]]:
    with _Options(env) as cnf:
        out = subprocess.run([str(MYSQL_BIN / "mysql.exe"), f"--defaults-extra-file={cnf}", "-N", "-B",
                              env["MYSQL_DATABASE"], "-e", sql], capture_output=True, text=True, check=True,
                             timeout=QUERY_TIMEOUT).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def row_counts(env) -> dict:
    counts = {}
    for table in KEY_TABLES:
        try:
            counts[table] = int(mysql_query(env, f"SELECT COUNT(*) FROM `{table}`")[0][0])
        except subprocess.CalledProcessError:
            counts[table] = None
    return counts


def dump_row_counts(dump: Path) -> dict:
    """Rows per key table in the dump (one INSERT line per row)."""
    prefixes = {table: f"INSERT INTO `{table}` VALUES".encode() for table in KEY_TABLES}
    counts = dict.fromkeys(KEY_TABLES, 0)
    with dump.open("rb") as f:
        for line in f:
            if line.startswith(b"INSERT INTO `"):
                for table, prefix in prefixes.items():
                    if line.startswith(prefix):
                        counts[table] += 1
                        break
    return counts


def model_identities() -> dict:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    try:
        from core.ml.reranker import SERVY_V5_WEIGHTS_SHA256
    except Exception:
        SERVY_V5_WEIGHTS_SHA256 = None
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=QUERY_TIMEOUT).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit = ""
    return {"v5_weights_sha256": SERVY_V5_WEIGHTS_SHA256, "embedding_profile": "bge-small", "git_commit": commit or None}


def backup(args):
    env = read_env(Path(args.env_file))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(args.out) / f"servy_backup_{env['MYSQL_DATABASE']}_{stamp}"
    if Path(__file__).resolve().parents[1] in target.resolve().parents:
        raise SystemExit("Backups must be written outside the repository.")
    (target / "mysql").mkdir(parents=True)
    dump = target / "mysql" / f"{env['MYSQL_DATABASE']}.sql"
    with _Options(env) as cnf, dump.open("wb") as out:
        # One INSERT per row (--skip-extended-insert) so row counts can be read from
        # the dump itself: they then describe exactly the snapshot that is restored.
        subprocess.run([str(MYSQL_BIN / "mysqldump.exe"), f"--defaults-extra-file={cnf}", "--single-transaction",
                        "--routines", "--triggers", "--no-tablespaces", "--hex-blob", "--default-character-set=utf8mb4",
                        "--skip-extended-insert", env["MYSQL_DATABASE"]], stdout=out, check=True,
                       timeout=DUMP_TIMEOUT)
    for name, source in (("chroma", args.chroma), ("media", args.media)):
        if source:
            shutil.copytree(source, target / name)
    files = hash_tree(target)
    manifest = {"format": "servy-mysql-backup-v1", "created_utc": stamp, "source_database": env["MYSQL_DATABASE"],
                "row_counts": dump_row_counts(dump), "models": model_identities(), "files": files,
                "contains_secrets": False}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"backup": str(target), "files": len(files), "bytes": sum(f["bytes"] for f in files.values()),
                      "row_counts": manifest["row_counts"]}, indent=1))
    return target


def verify(args) -> dict:
    root = Path(args.backup)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    actual = hash_tree(root)
    actual.pop("manifest.json", None)
    bad = sorted(k for k in set(manifest["files"]) | set(actual) if manifest["files"].get(k) != actual.get(k))
    if bad:
        raise SystemExit(f"Backup verification FAILED for {len(bad)} file(s), e.g. {bad[:3]}")
    secrets = [k for k in actual if Path(k).name.startswith(".env")]
    if secrets:
        raise SystemExit("Backup unexpectedly contains environment files.")
    print(f"Backup verified: {len(actual)} files match their SHA-256.")
    return manifest


def restore(args):
    manifest = verify(args)
    env = read_env(Path(args.env_file))
    database = env["MYSQL_DATABASE"]
    if database in PROTECTED_DATABASES:
        raise SystemExit(f"Refusing to restore into protected database {database}.")
    if database == manifest["source_database"]:
        raise SystemExit("Refusing to restore over the database the backup was taken from.")
    if args.confirm_target != database:
        raise SystemExit(f"--confirm-target must equal the target database name ({database}).")
    dump = Path(args.backup) / "mysql" / f"{manifest['source_database']}.sql"
    with _Options(env) as cnf, dump.open("rb") as source:
        subprocess.run([str(MYSQL_BIN / "mysql.exe"), f"--defaults-extra-file={cnf}", "--default-character-set=utf8mb4",
                        database], stdin=source, check=True, timeout=DUMP_TIMEOUT)
    for name, destination in (("chroma", args.chroma_to), ("media", args.media_to)):
        if destination:
            dest = Path(destination)
            if dest.exists() and any(dest.iterdir()):
                raise SystemExit(f"{name} destination must be empty: {dest}")
            shutil.copytree(Path(args.backup) / name, dest, dirs_exist_ok=True)
    counts = row_counts(env)
    mismatched = {t: (manifest["row_counts"].get(t), counts.get(t)) for t in KEY_TABLES
                  if manifest["row_counts"].get(t) != counts.get(t)}
    print(json.dumps({"restored_into": database, "row_counts": counts, "row_count_mismatches": mismatched}, indent=1))
    if mismatched:
        raise SystemExit("Restore verification FAILED: row counts differ.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backup"); b.add_argument("--env-file", required=True); b.add_argument("--out", required=True)
    b.add_argument("--chroma"); b.add_argument("--media")
    v = sub.add_parser("verify"); v.add_argument("--backup", required=True)
    r = sub.add_parser("restore"); r.add_argument("--backup", required=True); r.add_argument("--env-file", required=True)
    r.add_argument("--confirm-target", required=True); r.add_argument("--chroma-to"); r.add_argument("--media-to")
    args = parser.parse_args()
    try:
        {"backup": backup, "verify": verify, "restore": restore}[args.command](args)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"{args.command} FAILED: {Path(str(exc.cmd[0])).name} exceeded {exc.timeout}s and was stopped.")


if __name__ == "__main__":
    main()
