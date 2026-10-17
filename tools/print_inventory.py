import json

with open('inventory_sqlite.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total tables: {len(data)}")
total_rows = sum(v["row_count"] for v in data.values())
print(f"Total rows: {total_rows}\n")

for k, v in sorted(data.items()):
    pks = ",".join(v["pks"]) or "None"
    print(f"{k:<45} | Rows: {v['row_count']:<6} | PKs: {pks}")
