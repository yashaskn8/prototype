"""
Verify MySQL connection, version, privileges, storage engine, charset and collation.
"""
import os
import pymysql

# Load credentials
creds = {}
with open(".env.mysql_credentials", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()

print("Connecting to MySQL using servy_app account...")
conn = pymysql.connect(
    host=creds.get("MYSQL_HOST", "127.0.0.1"),
    port=int(creds.get("MYSQL_PORT", 3306)),
    user=creds.get("MYSQL_USER", "servy_app"),
    password=creds.get("MYSQL_PASSWORD"),
    database=creds.get("MYSQL_DATABASE", "servy"),
    charset="utf8mb4"
)

with conn.cursor() as cur:
    cur.execute("SELECT VERSION();")
    print(f"MySQL Version: {cur.fetchone()[0]}")

    cur.execute("SELECT CURRENT_USER();")
    print(f"Current User: {cur.fetchone()[0]}")

    cur.execute("SELECT @@default_storage_engine, @@character_set_server, @@collation_server;")
    engine, charset, collation = cur.fetchone()
    print(f"Default Storage Engine: {engine}")
    print(f"Server Charset: {charset}")
    print(f"Server Collation: {collation}")

    cur.execute("SELECT @@sql_mode;")
    print(f"SQL Mode: {cur.fetchone()[0]}")

    cur.execute("SHOW TABLES;")
    tables = cur.fetchall()
    print(f"Tables in servy: {[t[0] for t in tables]}")

conn.close()
print("MySQL connection and privileges successfully verified!")
