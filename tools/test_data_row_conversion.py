"""
Pre-flight check of data row formats in SQLite.
"""
import sqlite3
import json

conn = sqlite3.connect("file:db.sqlite3?mode=ro", uri=True)
cur = conn.cursor()

with open("inventory_sqlite.json", "r", encoding="utf-8") as f:
    inv = json.load(f)

for table, data in sorted(inv.items()):
    if data["row_count"] > 0:
        cur.execute(f'SELECT * FROM "{table}" LIMIT 1;')
        row = cur.fetchone()
        col_names = [c["name"] for c in data["columns"]]
        types = [type(val).__name__ for val in row]
        # Check for any datetime or special strings
        dt_cols = [col_names[i] for i, val in enumerate(row) if isinstance(val, str) and (val.startswith("202") and ("-" in val or ":" in val))]
        if dt_cols:
            print(f"{table}: sample datetime cols: {dt_cols}")

conn.close()
