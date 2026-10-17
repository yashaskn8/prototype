"""
Topological sort of SQLite tables based on foreign key dependencies.
"""
import json

with open("inventory_sqlite.json", "r", encoding="utf-8") as f:
    inv = json.load(f)

# Build dependency graph
# If table A has an FK pointing to table B, B must be inserted before A (unless self-referential)
graph = {t: set() for t in inv}
self_referential = set()

for t, data in inv.items():
    for fk in data["fks"]:
        target_t = fk["table"]
        if target_t == t:
            self_referential.add(t)
        elif target_t in graph:
            graph[t].add(target_t)

# Topological sort
sorted_tables = []
visited = set()
visiting = set()

def dfs(node):
    if node in visiting:
        # Cycle detected, break cycle
        return
    if node not in visited:
        visiting.add(node)
        for dep in graph.get(node, []):
            dfs(dep)
        visiting.remove(node)
        visited.add(node)
        sorted_tables.append(node)

for t in sorted(graph.keys()):
    dfs(t)

print("Self-referential tables:", sorted(list(self_referential)))
print(f"\nTopological insertion order ({len(sorted_tables)} tables):")
for i, t in enumerate(sorted_tables, 1):
    rows = inv[t]["row_count"]
    deps = list(graph[t])
    print(f"{i:2d}. {t:<40} (Rows: {rows:5d}) [Depends on: {','.join(deps) if deps else 'None'}]")
