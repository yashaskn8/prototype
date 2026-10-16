import os
import sys
import time
import tracemalloc
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
django.setup()

from core.ml.embeddings import get_profile, load_model, encode_queries
from core.models import Tenant, KnowledgeDocument
from core.services.retriever import retrieve

def run_benchmark():
    print("--- SERVY RAG CPU & RESOURCE BENCHMARK ---")
    tracemalloc.start()

    # 1. Model Loading
    t0 = time.perf_counter()
    profile = get_profile("bge-small")
    model = load_model(profile)
    load_time_s = time.perf_counter() - t0
    print(f"BGE-small cold load time: {load_time_s:.2f} s")

    # 2. Warm Query Latency
    queries = [
        "Spectra milk analyzer sample intake failure",
        "AeroLift safety gate interlock fault ERR-402",
        "Hydraulic oil pressure pump blockage",
        "Optical calibration unstable fat readings",
        "Centrifuge vibration damper replacement",
    ]
    # Warmup
    encode_queries(profile, model, [queries[0]])

    latencies = []
    for _ in range(5):
        for q in queries:
            t1 = time.perf_counter()
            encode_queries(profile, model, [q])
            latencies.append((time.perf_counter() - t1) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"Query encoding latency: p50 = {p50:.2f} ms, p95 = {p95:.2f} ms (N={len(latencies)})")

    # 3. Retrieval Latency
    tenant = Tenant.objects.first()
    if tenant:
        ret_lats = []
        for q in queries:
            t1 = time.perf_counter()
            retrieve(tenant=tenant, question=q, asset=None, top_k=5)
            ret_lats.append((time.perf_counter() - t1) * 1000)
        ret_lats.sort()
        print(f"End-to-end retrieve latency: p50 = {ret_lats[len(ret_lats)//2]:.2f} ms, p95 = {ret_lats[-1]:.2f} ms")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Traced memory: Current = {current / (1024*1024):.1f} MB, Peak = {peak / (1024*1024):.1f} MB")

if __name__ == "__main__":
    run_benchmark()
