"""
Phase 2 Comprehensive Backup Script for Servy RAG.
Creates a timestamped backup outside the Git repository (prototype).
Uses the SQLite Online Backup API for 100% consistent atomic snapshot.
Backs up SQLite DB, .env, chroma_db, media, ml_artifacts, ml_models, private_training_data, and migrations.
Calculates SHA-256 hashes and file sizes for every backed-up item.
Verifies integrity of the backup database.
"""
import os
import sys
import shutil
import sqlite3
import hashlib
import json
from datetime import datetime

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def backup_dir_recursive(src, dst, manifest):
    if not os.path.exists(src):
        return
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel_root) if rel_root != '.' else dst
        os.makedirs(target_root, exist_ok=True)
        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(target_root, f)
            shutil.copy2(src_file, dst_file)
            size = os.path.getsize(dst_file)
            file_hash = sha256_file(dst_file)
            rel_manifest_path = os.path.relpath(dst_file, dst)
            manifest[os.path.join(os.path.basename(src), rel_manifest_path).replace('\\', '/')] = {
                "size_bytes": size,
                "sha256": file_hash
            }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_base = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "backups_servy_migration"))
    backup_dir = os.path.join(backup_base, f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    print(f"=== INITIATING COMPREHENSIVE BACKUP ===")
    print(f"Target backup directory: {backup_dir}")

    manifest = {
        "timestamp": timestamp,
        "backup_directory": backup_dir,
        "items": {}
    }

    # 1. Consistent SQLite backup using sqlite3 backup API
    src_db = os.path.abspath('db.sqlite3')
    backup_db = os.path.join(backup_dir, 'db.sqlite3')
    print(f"\n[1/7] Backing up SQLite database via SQLite Online Backup API...")
    src_conn = sqlite3.connect(f'file:{src_db}?mode=ro', uri=True)
    dst_conn = sqlite3.connect(backup_db)
    with dst_conn:
        src_conn.backup(dst_conn, pages=100)
    dst_conn.close()
    src_conn.close()

    # Verify backup database
    verify_conn = sqlite3.connect(f'file:{backup_db}?mode=ro', uri=True)
    vcur = verify_conn.cursor()
    vcur.execute("PRAGMA integrity_check;")
    integrity = vcur.fetchall()
    vcur.execute("PRAGMA foreign_key_check;")
    fk_check = vcur.fetchall()
    vcur.execute("SELECT count(*) FROM sqlite_master WHERE type='table';")
    tbl_cnt = vcur.fetchone()[0]
    verify_conn.close()

    if integrity != [('ok',)]:
        raise RuntimeError(f"Backup SQLite failed integrity check: {integrity}")
    if fk_check != []:
        raise RuntimeError(f"Backup SQLite has foreign key violations: {fk_check}")

    backup_db_size = os.path.getsize(backup_db)
    backup_db_hash = sha256_file(backup_db)
    src_db_hash = sha256_file(src_db)
    print(f"  Backup DB Size: {backup_db_size} bytes")
    print(f"  Source DB Hash: {src_db_hash}")
    print(f"  Backup DB Hash: {backup_db_hash}")
    print(f"  Tables: {tbl_cnt}")
    print(f"  Integrity Check: {integrity[0][0]} - VERIFIED")

    manifest["items"]["db.sqlite3"] = {
        "size_bytes": backup_db_size,
        "sha256": backup_db_hash,
        "source_sha256": src_db_hash,
        "integrity": integrity[0][0]
    }

    # 2. Back up .env and .env.example
    print("\n[2/7] Backing up environment files...")
    for env_file in ['.env', '.env.example']:
        if os.path.exists(env_file):
            dst_env = os.path.join(backup_dir, env_file)
            shutil.copy2(env_file, dst_env)
            manifest["items"][env_file] = {
                "size_bytes": os.path.getsize(dst_env),
                "sha256": sha256_file(dst_env)
            }
            print(f"  Backed up {env_file}")

    # 3. Back up chroma_db
    print("\n[3/7] Backing up Chroma vector database...")
    chroma_dst = os.path.join(backup_dir, 'chroma_db')
    backup_dir_recursive('chroma_db', chroma_dst, manifest["items"])
    print(f"  Chroma DB backed up to {chroma_dst}")

    # 4. Back up media
    print("\n[4/7] Backing up media attachments...")
    media_dst = os.path.join(backup_dir, 'media')
    backup_dir_recursive('media', media_dst, manifest["items"])
    print(f"  Media attachments backed up to {media_dst}")

    # 5. Back up ml_artifacts & ml_models
    print("\n[5/7] Backing up ML artifacts and models...")
    for ml_dir in ['ml_artifacts', 'ml_models']:
        dst_dir = os.path.join(backup_dir, ml_dir)
        backup_dir_recursive(ml_dir, dst_dir, manifest["items"])
        print(f"  {ml_dir} backed up")

    # 6. Back up private_training_data
    print("\n[6/7] Backing up private training & evaluation data...")
    backup_dir_recursive('private_training_data', os.path.join(backup_dir, 'private_training_data'), manifest["items"])
    print(f"  private_training_data backed up")

    # 7. Dump migration state
    print("\n[7/7] Dumping migration metadata...")
    conn = sqlite3.connect(f'file:{backup_db}?mode=ro', uri=True)
    cur = conn.cursor()
    cur.execute("SELECT app, name, applied FROM django_migrations ORDER BY id;")
    migrations = [{"app": r[0], "name": r[1], "applied": r[2]} for r in cur.fetchall()]
    conn.close()

    migration_file = os.path.join(backup_dir, "django_migrations.json")
    with open(migration_file, 'w', encoding='utf-8') as f:
        json.dump(migrations, f, indent=2)
    manifest["items"]["django_migrations.json"] = {
        "size_bytes": os.path.getsize(migration_file),
        "sha256": sha256_file(migration_file)
    }

    # Save manifest
    manifest_path = os.path.join(backup_dir, "backup_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n=== BACKUP COMPLETED AND VERIFIED ===")
    print(f"Manifest written to: {manifest_path}")
    print(f"Total backup items tracked: {len(manifest['items'])}")
    return backup_dir

if __name__ == '__main__':
    main()
