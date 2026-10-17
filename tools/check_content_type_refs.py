import sqlite3
import json

with open("inventory_sqlite.json", "r", encoding="utf-8") as f:
    inv = json.load(f)

ct_refs = []
for table, data in inv.items():
    for col in data["columns"]:
        if "content_type" in col["name"].lower():
            ct_refs.append((table, col["name"], data["row_count"]))
    for fk in data["fks"]:
        if fk["table"] == "django_content_type":
            ct_refs.append((table, fk["from"], data["row_count"]))

print("Tables referencing content_type:")
for r in sorted(set(ct_refs)):
    print(f"  {r[0]}.{r[1]} (row count in SQLite: {r[2]})")
