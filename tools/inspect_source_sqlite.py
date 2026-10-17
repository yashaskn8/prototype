"""
Comprehensive SQLite Pre-Migration Inspection Script for Servy RAG.
Inspects tables, row counts, primary keys, foreign keys, indexes,
Chroma cross-references, migration history, and database integrity.
Runs strictly read-only.
"""
import os
import sys
import sqlite3
import json
import hashlib

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    db_path = os.path.abspath('db.sqlite3')
    if not os.path.exists(db_path):
        print(f"ERROR: SQLite database does not exist at {db_path}")
        sys.exit(1)

    file_size = os.path.getsize(db_path)
    file_hash = sha256_file(db_path)
    print(f"=== SQLITE DATABASE FILE ===")
    print(f"Path: {db_path}")
    print(f"Size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"SHA-256: {file_hash}")

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cur = conn.cursor()

    cur.execute("PRAGMA integrity_check;")
    integrity = cur.fetchall()
    print(f"Integrity check: {integrity}")

    cur.execute("PRAGMA foreign_key_check;")
    fk_violations = cur.fetchall()
    print(f"Foreign key violations: {fk_violations}")

    cur.execute("PRAGMA journal_mode;")
    journal_mode = cur.fetchone()[0]
    print(f"Journal mode: {journal_mode}")

    # Inspect tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    print(f"\nTotal application/framework tables: {len(tables)}")

    inventory = {}
    total_rows = 0

    for table in tables:
        cur.execute(f'SELECT COUNT(*) FROM "{table}";')
        count = cur.fetchone()[0]
        total_rows += count

        cur.execute(f'PRAGMA table_info("{table}");')
        columns = cur.fetchall()  # cid, name, type, notnull, dflt_value, pk

        cur.execute(f'PRAGMA foreign_key_list("{table}");')
        fks = cur.fetchall()

        cur.execute(f'PRAGMA index_list("{table}");')
        indexes = cur.fetchall()

        inventory[table] = {
            "row_count": count,
            "columns": [{"name": c[1], "type": c[2], "notnull": c[3], "pk": c[5]} for c in columns],
            "pks": [c[1] for c in columns if c[5] > 0],
            "fks": [{"table": fk[2], "from": fk[3], "to": fk[4]} for fk in fks],
            "indexes": [idx[1] for idx in indexes]
        }

    print(f"Total rows across all tables: {total_rows}\n")
    print(f"{'Table Name':<45} | {'Rows':<8} | {'PKs':<20} | {'FK Count':<8}")
    print("-" * 90)
    for t in sorted(tables):
        inv = inventory[t]
        pks_str = ",".join(inv["pks"]) or "None"
        print(f"{t:<45} | {inv['row_count']:<8} | {pks_str:<20} | {len(inv['fks']):<8}")

    # Inspect migration history
    if 'django_migrations' in tables:
        cur.execute("SELECT app, name, applied FROM django_migrations ORDER BY applied, id;")
        migrations = cur.fetchall()
        print(f"\nApplied Django Migrations ({len(migrations)} total):")
        for m in migrations:
            print(f"  [{m[0]}] {m[1]} (applied: {m[2]})")

    # Inspect Chroma cross-reference if chroma exists
    chroma_dir = os.path.abspath('chroma_db')
    print(f"\n=== CHROMA VECTOR DATABASE ===")
    print(f"Chroma path: {chroma_dir}")
    if os.path.exists(chroma_dir):
        print(f"Chroma dir exists. Checking contents...")
        for root, dirs, files in os.walk(chroma_dir):
            for f in files:
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, chroma_dir)
                print(f"  {rel} ({os.path.getsize(fp)} bytes)")
    else:
        print("Chroma dir does NOT exist.")

    # Inspect media attachments
    media_dir = os.path.abspath('media')
    print(f"\n=== MEDIA ATTACHMENTS ===")
    print(f"Media path: {media_dir}")
    if os.path.exists(media_dir):
        media_files = []
        for root, dirs, files in os.walk(media_dir):
            for f in files:
                fp = os.path.join(root, f)
                media_files.append((os.path.relpath(fp, media_dir), os.path.getsize(fp)))
        print(f"Total media files: {len(media_files)}")
        for mf, sz in media_files[:10]:
            print(f"  {mf} ({sz} bytes)")
        if len(media_files) > 10:
            print(f"  ... and {len(media_files)-10} more")
    else:
        print("Media dir does NOT exist.")

    # Inspect checkpoints / artifacts
    for d in ['ml_artifacts', 'ml_models', 'private_training_data', 'sample_data']:
        dp = os.path.abspath(d)
        if os.path.exists(dp):
            cnt = sum(len(files) for _, _, files in os.walk(dp))
            print(f"Directory {d}: {cnt} files found.")

    # Save JSON inventory
    with open('inventory_sqlite.json', 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
    print("Saved inventory to inventory_sqlite.json")

    conn.close()

if __name__ == '__main__':
    main()
