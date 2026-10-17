"""
Inspect framework-generated records in SQLite source vs MySQL destination.
"""
import sqlite3
import pymysql

# Load MySQL credentials
creds = {}
with open(".env.mysql_credentials", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()

sqlite_conn = sqlite3.connect("file:db.sqlite3?mode=ro", uri=True)
sq_cur = sqlite_conn.cursor()

mysql_conn = pymysql.connect(
    host="127.0.0.1", port=3306, user="servy_app", password=creds["MYSQL_PASSWORD"], database="servy"
)
my_cur = mysql_conn.cursor()

print("=== CONTENT TYPES ===")
sq_cur.execute("SELECT id, app_label, model FROM django_content_type ORDER BY id;")
sq_ct = sq_cur.fetchall()
my_cur.execute("SELECT id, app_label, model FROM django_content_type ORDER BY id;")
my_ct = my_cur.fetchall()

print(f"SQLite ContentTypes count: {len(sq_ct)}")
print(f"MySQL ContentTypes count: {len(my_ct)}")

sq_ct_map = {f"{r[1]}.{r[2]}": r[0] for r in sq_ct}
my_ct_map = {f"{r[1]}.{r[2]}": r[0] for r in my_ct}

ct_mismatches = []
for k in sq_ct_map:
    if k in my_ct_map:
        if sq_ct_map[k] != my_ct_map[k]:
            ct_mismatches.append((k, sq_ct_map[k], my_ct_map[k]))
    else:
        print(f"Missing in MySQL: {k}")

print(f"ContentType ID mismatches between SQLite and MySQL: {len(ct_mismatches)}")
for m in ct_mismatches[:5]:
    print(f"  {m[0]}: SQLite id={m[1]}, MySQL id={m[2]}")

print("\n=== PERMISSIONS ===")
sq_cur.execute("SELECT id, content_type_id, codename, name FROM auth_permission ORDER BY id;")
sq_perm = sq_cur.fetchall()
my_cur.execute("SELECT id, content_type_id, codename, name FROM auth_permission ORDER BY id;")
my_perm = my_cur.fetchall()
print(f"SQLite Permissions count: {len(sq_perm)}")
print(f"MySQL Permissions count: {len(my_perm)}")

sq_perm_map = {f"{r[1]}:{r[2]}": r[0] for r in sq_perm}
my_perm_map = {f"{r[1]}:{r[2]}": r[0] for r in my_perm}

perm_id_mismatches = []
for k in sq_perm_map:
    if k in my_perm_map:
        if sq_perm_map[k] != my_perm_map[k]:
            perm_id_mismatches.append((k, sq_perm_map[k], my_perm_map[k]))
print(f"Permission ID mismatches: {len(perm_id_mismatches)}")

# Check if any user_permissions or group_permissions exist in SQLite
sq_cur.execute("SELECT count(*) FROM auth_user_user_permissions;")
print(f"SQLite auth_user_user_permissions: {sq_cur.fetchone()[0]}")
sq_cur.execute("SELECT count(*) FROM auth_group_permissions;")
print(f"SQLite auth_group_permissions: {sq_cur.fetchone()[0]}")

sqlite_conn.close()
mysql_conn.close()
