import os
import sqlite3
import json

chroma_sqlite = os.path.abspath('chroma_db/chroma.sqlite3')
if not os.path.exists(chroma_sqlite):
    print("No chroma.sqlite3 found")
    exit(0)

conn = sqlite3.connect(f'file:{chroma_sqlite}?mode=ro', uri=True)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Chroma SQLite tables:", [r[0] for r in cur.fetchall()])

# Check collections
cur.execute("SELECT id, name FROM collections;")
collections = cur.fetchall()
print(f"\nCollections ({len(collections)} total):")
for c in collections:
    print(f"  ID: {c[0]}, Name: {c[1]}")

# Check embeddings/records count in chroma
# Check schema of embeddings and embedding_metadata
cur.execute("PRAGMA table_info(embeddings);")
print("embeddings columns:", [r[1] for r in cur.fetchall()])
cur.execute("PRAGMA table_info(embedding_metadata);")
print("embedding_metadata columns:", [r[1] for r in cur.fetchall()])

cur.execute("SELECT segment_id, count(*) FROM embeddings GROUP BY segment_id;")
for r in cur.fetchall():
    print("  Segment count:", r)

cur.execute("SELECT DISTINCT key FROM embedding_metadata;")
print("Metadata keys:", [r[0] for r in cur.fetchall()])

cur.execute("SELECT id, string_value, int_value FROM embedding_metadata WHERE key IN ('chunk_id', 'document_id', 'doc_id') LIMIT 10;")
print("Sample metadata entries:", cur.fetchall())

conn.close()
