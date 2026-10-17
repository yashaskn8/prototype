"""
Phase 6: Comprehensive SQLite vs MySQL Verification Tool for Servy RAG.
Audits table inventory, row counts, PK sets, FK integrity, M2M mappings,
Null distributions, document hashes, knowledge chunks, user hashes,
JSON fields, and canonicalized record hashes.
"""
import os
import sys
import json
import hashlib
import sqlite3
import pymysql
from decimal import Decimal
from datetime import datetime, date

def load_mysql_credentials():
    creds = {}
    with open(".env.mysql_credentials", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
    return creds

def canonical_val(val):
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, datetime):
        s = str(val).replace('T', ' ')
        if '.' in s:
            base, frac = s.split('.', 1)
            frac = (frac + "000000")[:6]
            return f"{base}.{frac}"
        return f"{s}.000000"
    if isinstance(val, date):
        return str(val)
    if isinstance(val, Decimal):
        return f"{float(val):.6f}"
    if isinstance(val, float):
        return f"{val:.6f}"
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        s_strip = val.strip()
        # Handle stringified datetimes in SQLite
        if len(s_strip) >= 19 and s_strip[4] == '-' and s_strip[7] == '-' and (s_strip[10] in (' ', 'T')):
            s_norm = s_strip.replace('T', ' ')
            if '.' in s_norm:
                base, frac = s_norm.split('.', 1)
                frac = (frac + "000000")[:6]
                return f"{base}.{frac}"
            return f"{s_norm}.000000"
        # Try JSON normalization
        if (s_strip.startswith("{") and s_strip.endswith("}")) or (s_strip.startswith("[") and s_strip.endswith("]")):
            try:
                parsed = json.loads(s_strip)
                return json.dumps(parsed, sort_keys=True)
            except Exception:
                pass
        return s_strip
    return str(val)

def main():
    print("====================================================================")
    print("  SERVY RAG: DEEP SQLITE VS MYSQL DATA INTEGRITY AUDIT (PHASE 6)")
    print("====================================================================")

    creds = load_mysql_credentials()

    sq_conn = sqlite3.connect("file:db.sqlite3?mode=ro", uri=True)
    sq_cur = sq_conn.cursor()

    my_conn = pymysql.connect(
        host=creds.get("MYSQL_HOST", "127.0.0.1"),
        port=int(creds.get("MYSQL_PORT", 3306)),
        user=creds.get("MYSQL_USER", "servy_app"),
        password=creds.get("MYSQL_PASSWORD"),
        database=creds.get("MYSQL_DATABASE", "servy"),
        charset="utf8mb4"
    )
    my_cur = my_conn.cursor()

    with open("inventory_sqlite.json", "r", encoding="utf-8") as f:
        inv = json.load(f)

    results = {
        "tables_checked": 0,
        "row_count_mismatches": 0,
        "pk_set_mismatches": 0,
        "fk_violations": 0,
        "hash_mismatches": 0,
        "details": {}
    }

    # 1. Table inventory check
    my_cur.execute("SHOW TABLES;")
    my_tables = set(r[0] for r in my_cur.fetchall())
    sq_tables = set(inv.keys())

    missing_in_mysql = sq_tables - my_tables
    print(f"\n[1/10] Table Inventory Audit:")
    print(f"  SQLite tables: {len(sq_tables)}")
    print(f"  MySQL tables:  {len(my_tables)}")
    if missing_in_mysql:
        print(f"  FAILED: Missing in MySQL: {missing_in_mysql}")
        sys.exit(1)
    print(f"  PASSED: All {len(sq_tables)} tables present in MySQL.")

    # 2. Row count and PK set audit
    print(f"\n[2/10] Row Counts and Primary Key Sets Audit:")
    for table_name in sorted(sq_tables):
        sq_count = inv[table_name]["row_count"]
        my_cur.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
        my_count = my_cur.fetchone()[0]

        pk_col = inv[table_name]["pks"]
        pk_match = True

        if sq_count > 0 and pk_col:
            pk_name = pk_col[0]
            sq_cur.execute(f'SELECT "{pk_name}" FROM "{table_name}" ORDER BY "{pk_name}";')
            sq_pks = set(r[0] for r in sq_cur.fetchall())

            my_cur.execute(f"SELECT `{pk_name}` FROM `{table_name}` ORDER BY `{pk_name}`;")
            my_pks = set(r[0] for r in my_cur.fetchall())

            if sq_pks != my_pks:
                pk_match = False
                results["pk_set_mismatches"] += 1
                print(f"  FAIL PK: {table_name} PK sets differ!")

        if sq_count != my_count:
            results["row_count_mismatches"] += 1
            print(f"  FAIL COUNT: {table_name}: SQLite={sq_count}, MySQL={my_count}")

        results["tables_checked"] += 1

    if results["row_count_mismatches"] == 0 and results["pk_set_mismatches"] == 0:
        print(f"  PASSED: 100% of row counts and primary key sets match across all tables.")
    else:
        print(f"  FAILED row counts or PK sets!")
        sys.exit(1)

    # 3. Foreign key integrity audit in MySQL
    print(f"\n[3/10] Foreign Key Referential Integrity Audit:")
    fk_errors = 0
    for table_name, data in inv.items():
        for fk in data["fks"]:
            fk_col = fk["from"]
            ref_table = fk["table"]
            ref_col = fk["to"]
            if ref_table in my_tables:
                check_sql = f"""
                SELECT COUNT(*) FROM `{table_name}` t
                WHERE t.`{fk_col}` IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM `{ref_table}` r WHERE r.`{ref_col}` = t.`{fk_col}`
                  );
                """
                my_cur.execute(check_sql)
                orphans = my_cur.fetchone()[0]
                if orphans > 0:
                    print(f"  FAIL FK: {table_name}.{fk_col} -> {ref_table}.{ref_col} has {orphans} orphans!")
                    fk_errors += orphans
    results["fk_violations"] = fk_errors
    if fk_errors == 0:
        print(f"  PASSED: 0 foreign key violations detected across all relations.")

    # 4. Many-to-many relationship audit (core_project_assets)
    print(f"\n[4/10] Many-to-Many Associations Audit:")
    sq_cur.execute("SELECT id, project_id, asset_id FROM core_project_assets ORDER BY id;")
    sq_m2m = list(sq_cur.fetchall())
    my_cur.execute("SELECT id, project_id, asset_id FROM core_project_assets ORDER BY id;")
    my_m2m = list(my_cur.fetchall())
    if sq_m2m == my_m2m:
        print(f"  PASSED: core_project_assets ({len(sq_m2m)} associations) match 100% identically.")
    else:
        print(f"  FAIL M2M: core_project_assets mismatch!")
        sys.exit(1)

    # 5. User authentication and password hash audit
    print(f"\n[5/10] User Authentication & Password Hash Audit:")
    sq_cur.execute("SELECT id, username, password, is_active, is_staff, is_superuser FROM auth_user ORDER BY id;")
    sq_users = [tuple(canonical_val(x) for x in r) for r in sq_cur.fetchall()]
    my_cur.execute("SELECT id, username, password, is_active, is_staff, is_superuser FROM auth_user ORDER BY id;")
    my_users = [tuple(canonical_val(x) for x in r) for r in my_cur.fetchall()]
    if sq_users == my_users:
        print(f"  PASSED: All {len(sq_users)} users, password hashes, and flags match 100%.")
    else:
        print(f"  FAIL: User auth mismatch! Differences: {[(a, b) for a, b in zip(sq_users, my_users) if a != b]}")
        sys.exit(1)

    # 6. Knowledge Document & Chunk Audit
    print(f"\n[6/10] Knowledge Document & Chunk Provenance Audit:")
    sq_cur.execute("SELECT id, tenant_id, product_id, title, checksum_sha256 FROM core_knowledgedocument ORDER BY id;")
    sq_docs = [tuple(canonical_val(x) for x in r) for r in sq_cur.fetchall()]
    my_cur.execute("SELECT id, tenant_id, product_id, title, checksum_sha256 FROM core_knowledgedocument ORDER BY id;")
    my_docs = [tuple(canonical_val(x) for x in r) for r in my_cur.fetchall()]
    if sq_docs == my_docs:
        print(f"  PASSED: All {len(sq_docs)} documents (IDs, titles, tenant IDs, source SHA-256) match 100%.")
    else:
        print(f"  FAIL: Document metadata mismatch!")
        sys.exit(1)

    sq_cur.execute("SELECT id, document_id, tenant_id, chunk_index, length(text) FROM core_knowledgechunk ORDER BY id;")
    sq_chunks = list(sq_cur.fetchall())
    my_cur.execute("SELECT id, document_id, tenant_id, chunk_index, CHAR_LENGTH(text) FROM core_knowledgechunk ORDER BY id;")
    my_chunks = list(my_cur.fetchall())
    if sq_chunks == my_chunks:
        print(f"  PASSED: All {len(sq_chunks)} chunks (IDs, document_id mapping, character lengths) match 100%.")
    else:
        print(f"  FAIL: Chunk metadata mismatch!")
        sys.exit(1)

    # Deep chunk text SHA-256 comparison
    sq_cur.execute("SELECT id, text FROM core_knowledgechunk ORDER BY id;")
    sq_chunk_texts = {r[0]: hashlib.sha256(r[1].encode('utf-8')).hexdigest() for r in sq_cur.fetchall()}
    my_cur.execute("SELECT id, text FROM core_knowledgechunk ORDER BY id;")
    my_chunk_texts = {r[0]: hashlib.sha256(r[1].encode('utf-8')).hexdigest() for r in my_cur.fetchall()}
    if sq_chunk_texts == my_chunk_texts:
        print(f"  PASSED: 100% of chunk text content SHA-256 hashes match identically.")
    else:
        print(f"  FAIL: Chunk content hash mismatch!")
        sys.exit(1)

    # 7. Service Call & Resolution Index Audit
    print(f"\n[7/10] Service Calls & Resolution Index Audit:")
    sq_cur.execute("SELECT count(*), count(resolution_text), count(complaint_text) FROM core_servicecall;")
    sq_sc_stats = tuple(sq_cur.fetchone())
    my_cur.execute("SELECT count(*), count(resolution_text), count(complaint_text) FROM core_servicecall;")
    my_sc_stats = tuple(my_cur.fetchone())
    if sq_sc_stats == my_sc_stats:
        print(f"  PASSED: ServiceCall stats (total={sq_sc_stats[0]}, non-null resolutions={sq_sc_stats[1]}, complaints={sq_sc_stats[2]}) match.")
    else:
        print(f"  FAIL: ServiceCall stats mismatch!")
        sys.exit(1)

    sq_cur.execute("SELECT count(*), count(tenant_id), count(service_call_id) FROM core_serviceresolutionindex;")
    sq_sri = tuple(sq_cur.fetchone())
    my_cur.execute("SELECT count(*), count(tenant_id), count(service_call_id) FROM core_serviceresolutionindex;")
    my_sri = tuple(my_cur.fetchone())
    if sq_sri == my_sri:
        print(f"  PASSED: ServiceResolutionIndex ({sq_sri[0]} rows) match.")
    else:
        print(f"  FAIL: ServiceResolutionIndex mismatch!")
        sys.exit(1)

    # 8. Tenant & Customer Isolation Audit
    print(f"\n[8/10] Multi-Tenant Boundaries & Ownership Audit:")
    sq_cur.execute("SELECT id, name, slug FROM core_tenant ORDER BY id;")
    sq_tenants = list(sq_cur.fetchall())
    my_cur.execute("SELECT id, name, slug FROM core_tenant ORDER BY id;")
    my_tenants = list(my_cur.fetchall())
    if sq_tenants == my_tenants:
        print(f"  PASSED: All {len(sq_tenants)} tenants match identically.")
    else:
        print(f"  FAIL: Tenant mismatch!")
        sys.exit(1)

    sq_cur.execute("SELECT tenant_id, count(*) FROM core_asset GROUP BY tenant_id ORDER BY tenant_id;")
    sq_asset_tenant = list(sq_cur.fetchall())
    my_cur.execute("SELECT tenant_id, count(*) FROM core_asset GROUP BY tenant_id ORDER BY tenant_id;")
    my_asset_tenant = list(my_cur.fetchall())
    if sq_asset_tenant == my_asset_tenant:
        print(f"  PASSED: Asset-to-tenant distribution matches identically.")
    else:
        print(f"  FAIL: Asset tenant distribution mismatch!")
        sys.exit(1)

    # 9. JSON Field Structure and Content Audit
    print(f"\n[9/10] Structured JSON Fields Equivalence Audit:")
    sq_cur.execute("SELECT id, payload FROM core_diagnosticevent WHERE payload IS NOT NULL AND payload != '' ORDER BY id;")
    sq_payloads = sq_cur.fetchall()
    my_cur.execute("SELECT id, payload FROM core_diagnosticevent WHERE payload IS NOT NULL AND payload != '' ORDER BY id;")
    my_payloads = dict(my_cur.fetchall())
    json_match = True
    for event_id, payload_str in sq_payloads:
        my_val = my_payloads.get(event_id)
        p1 = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
        p2 = json.loads(my_val) if isinstance(my_val, str) else my_val
        if p1 != p2:
            json_match = False
            print(f"  FAIL JSON: Event {event_id} payload differs!")
            break
    if json_match:
        print(f"  PASSED: 100% of diagnostic event JSON payloads ({len(sq_payloads)} records) match semantically.")
    else:
        sys.exit(1)

    # 10. Canonical Record Hash Audit for High-Value Tables
    print(f"\n[10/10] Canonical Content Hash Audit for High-Value Tables:")
    high_value_tables = [
        "core_tenant", "core_customer", "core_product", "core_asset",
        "core_knowledgedocument", "core_knowledgechunk", "core_staffprofile",
        "core_diagnosticplaybook", "core_slapolicy"
    ]
    hash_failures = 0
    for tbl in high_value_tables:
        sq_cur.execute(f'PRAGMA table_info("{tbl}");')
        cols = [c[1] for c in sq_cur.fetchall()]
        cols_str_sq = ", ".join(f'"{c}"' for c in cols)
        cols_str_my = ", ".join(f'`{c}`' for c in cols)

        sq_cur.execute(f'SELECT {cols_str_sq} FROM "{tbl}" ORDER BY id;')
        sq_rows = sq_cur.fetchall()
        my_cur.execute(f"SELECT {cols_str_my} FROM `{tbl}` ORDER BY id;")
        my_rows = my_cur.fetchall()

        # Canonicalize each row
        sq_canon = [[canonical_val(val) for val in r] for r in sq_rows]
        my_canon = [[canonical_val(val) for val in r] for r in my_rows]

        row_diffs = []
        for idx, (r1, r2) in enumerate(zip(sq_canon, my_canon)):
            if r1 != r2:
                row_diffs.append((idx, r1, r2))

        if not row_diffs:
            sq_hash = hashlib.sha256(json.dumps(sq_canon, sort_keys=True).encode('utf-8')).hexdigest()
            print(f"  PASSED: {tbl:<25} 100% identical ({len(sq_rows)} rows). SHA-256: {sq_hash[:16]}...")
        else:
            print(f"  FAIL: {tbl:<25} {len(row_diffs)} rows differ! Sample diff: {row_diffs[0]}")
            hash_failures += len(row_diffs)

    results["hash_mismatches"] = hash_failures

    print("\n====================================================================")
    print("  VERIFICATION SUMMARY")
    print(f"  Tables Audited:          {results['tables_checked']}")
    print(f"  Row Count Mismatches:    {results['row_count_mismatches']}")
    print(f"  PK Set Mismatches:       {results['pk_set_mismatches']}")
    print(f"  FK Violations:           {results['fk_violations']}")
    print(f"  Hash Mismatches:         {results['hash_mismatches']}")
    if results['row_count_mismatches'] == 0 and results['fk_violations'] == 0 and results['hash_mismatches'] == 0:
        print("  STATUS: 100% DATA INTEGRITY VERIFIED - PERFECT MIGRATION")
    else:
        print("  STATUS: AUDIT FAILED - MISMATCHES DETECTED")
        sys.exit(1)
    print("====================================================================")

    sq_conn.close()
    my_conn.close()

if __name__ == '__main__':
    main()
