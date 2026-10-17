"""
Phase 10: Performance & Resource Benchmark Tool.
Compares SQLite and MySQL 8.0 latencies across:
- Connection time
- Document listing
- Asset-scoped queries
- Knowledge-chunk retrieval
- Service call queries
- Memory consumption on Windows
"""
import os
import sys
import time
import json
import sqlite3
import pymysql

def load_mysql_credentials():
    creds = {}
    with open(".env.mysql_credentials", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
    return creds

def time_it(func, iterations=10):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "min_ms": round(min(times), 3),
        "avg_ms": round(sum(times) / len(times), 3),
        "max_ms": round(max(times), 3)
    }

def main():
    print("====================================================================")
    print("  PHASE 10: SQLITE VS MYSQL 8.0 PERFORMANCE BENCHMARK")
    print("====================================================================")

    creds = load_mysql_credentials()
    sq_path = "db.sqlite3"

    # 1. Connection Latency
    def connect_sqlite():
        conn = sqlite3.connect(f"file:{sq_path}?mode=ro", uri=True)
        conn.close()

    def connect_mysql():
        conn = pymysql.connect(
            host=creds.get("MYSQL_HOST", "127.0.0.1"),
            port=int(creds.get("MYSQL_PORT", 3306)),
            user=creds.get("MYSQL_USER", "servy_app"),
            password=creds.get("MYSQL_PASSWORD"),
            database=creds.get("MYSQL_DATABASE", "servy")
        )
        conn.close()

    bench_connect_sq = time_it(connect_sqlite, 20)
    bench_connect_my = time_it(connect_mysql, 20)

    # Establish persistent connections for query benchmarking
    sq_conn = sqlite3.connect(f"file:{sq_path}?mode=ro", uri=True)
    my_conn = pymysql.connect(
        host=creds.get("MYSQL_HOST", "127.0.0.1"),
        port=int(creds.get("MYSQL_PORT", 3306)),
        user=creds.get("MYSQL_USER", "servy_app"),
        password=creds.get("MYSQL_PASSWORD"),
        database=creds.get("MYSQL_DATABASE", "servy")
    )

    # 2. Document Listing
    def doc_list_sq():
        cur = sq_conn.cursor()
        cur.execute("SELECT id, title, is_rag_enabled, tenant_id FROM core_knowledgedocument WHERE is_rag_enabled=1;")
        return cur.fetchall()

    def doc_list_my():
        cur = my_conn.cursor()
        cur.execute("SELECT id, title, is_rag_enabled, tenant_id FROM core_knowledgedocument WHERE is_rag_enabled=1;")
        return cur.fetchall()

    bench_doc_sq = time_it(doc_list_sq, 30)
    bench_doc_my = time_it(doc_list_my, 30)

    # 3. Asset-Scoped Queries
    def asset_query_sq():
        cur = sq_conn.cursor()
        cur.execute("SELECT id, name, asset_code, status FROM core_asset WHERE tenant_id=6;")
        return cur.fetchall()

    def asset_query_my():
        cur = my_conn.cursor()
        cur.execute("SELECT id, name, asset_code, status FROM core_asset WHERE tenant_id=6;")
        return cur.fetchall()

    bench_asset_sq = time_it(asset_query_sq, 30)
    bench_asset_my = time_it(asset_query_my, 30)

    # 4. Knowledge Chunk Retrieval
    def chunk_query_sq():
        cur = sq_conn.cursor()
        cur.execute("SELECT id, chunk_index, heading, length(text) FROM core_knowledgechunk WHERE document_id=174;")
        return cur.fetchall()

    def chunk_query_my():
        cur = my_conn.cursor()
        cur.execute("SELECT id, chunk_index, heading, char_length(text) FROM core_knowledgechunk WHERE document_id=174;")
        return cur.fetchall()

    bench_chunk_sq = time_it(chunk_query_sq, 30)
    bench_chunk_my = time_it(chunk_query_my, 30)

    # 5. Service Call Queries (Large table: 7706 rows)
    def sc_query_sq():
        cur = sq_conn.cursor()
        cur.execute("SELECT id, servy_id, status, priority FROM core_servicecall WHERE tenant_id=6 LIMIT 100;")
        return cur.fetchall()

    def sc_query_my():
        cur = my_conn.cursor()
        cur.execute("SELECT id, servy_id, status, priority FROM core_servicecall WHERE tenant_id=6 LIMIT 100;")
        return cur.fetchall()

    bench_sc_sq = time_it(sc_query_sq, 30)
    bench_sc_my = time_it(sc_query_my, 30)

    print(f"\n{'Operation':<35} | {'SQLite (avg)':<14} | {'MySQL 8.0 (avg)':<16} | {'Delta':<10}")
    print("-" * 80)
    def print_row(name, sq, my):
        delta = my['avg_ms'] - sq['avg_ms']
        sign = "+" if delta >= 0 else ""
        print(f"{name:<35} | {sq['avg_ms']:<8.3f} ms   | {my['avg_ms']:<10.3f} ms   | {sign}{delta:<6.3f} ms")

    print_row("TCP/DB Connection Latency", bench_connect_sq, bench_connect_my)
    print_row("Knowledge Document Listing", bench_doc_sq, bench_doc_my)
    print_row("Asset-Scoped Query", bench_asset_sq, bench_asset_my)
    print_row("Knowledge Chunk Retrieval", bench_chunk_sq, bench_chunk_my)
    print_row("Service Call Batch Query (100)", bench_sc_sq, bench_sc_my)

    results = {
        "connection": {"sqlite": bench_connect_sq, "mysql": bench_connect_my},
        "document_listing": {"sqlite": bench_doc_sq, "mysql": bench_doc_my},
        "asset_query": {"sqlite": bench_asset_sq, "mysql": bench_asset_my},
        "chunk_retrieval": {"sqlite": bench_chunk_sq, "mysql": bench_chunk_my},
        "service_calls": {"sqlite": bench_sc_sq, "mysql": bench_sc_my}
    }

    with open("performance_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sq_conn.close()
    my_conn.close()
    print("\nSaved benchmark results to performance_benchmark.json")

if __name__ == '__main__':
    main()
