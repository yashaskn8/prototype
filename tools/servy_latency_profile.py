"""Per-stage latency of the real conversational pipeline, from its own stream events.

Streams questions through the Customer gateway and records, per turn: time to the
first streamed event, time to the final answer, and the pipeline's own stage
durations (dense/BGE, sparse, rank fusion, V5 reranking, retrieval total, Qwen,
evidence validation). Never prints the access code.

Usage: .venv\\Scripts\\python.exe -B tools\\servy_latency_profile.py --access-file <file> [--label x]
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8002/demo-api/customer-support/conversations/"
QUESTIONS = [
    (1, "My split AC is not cooling."), (1, "Why is there ice on the coil?"),
    (1, "How often should a comprehensive service be performed?"),
    (2, "The VRF outdoor module shows a high-pressure alarm."), (2, "What causes a communication fault?"),
    (2, "What should be checked during routine inspection?"),
    (3, "The airflow from the ducts is weak."), (3, "What can cause water leakage?"),
    (3, "How often should the filters be inspected?"),
]
STAGES = ("dense_retrieval", "sparse_retrieval", "rank_fusion", "v5_reranking", "retrieval_pipeline",
          "customer_qwen_generation", "customer_evidence_validation", "customer_rag_execution")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-file", required=True)
    parser.add_argument("--label", default="profile")
    args = parser.parse_args()
    auth = {"Authorization": "Demo " + Path(args.access_file).read_text().strip()}
    stage_ms, first_ms, total_ms = defaultdict(list), [], []
    for asset, question in QUESTIONS:
        uid = requests.post(BASE, headers=auth, json={"asset_id": asset}, timeout=30).json()["conversation_uuid"]
        started = time.perf_counter()
        response = requests.post(f"{BASE}{uid}/messages-stream/", headers=auth, json={"message": question},
                                 stream=True, timeout=300)
        first, event = None, None
        for raw in response.iter_lines(decode_unicode=True):
            if first is None and raw:
                first = (time.perf_counter() - started) * 1000
            if raw.startswith("event:"):
                event = raw[6:].strip()
            elif raw.startswith("data:") and event == "log":
                data = json.loads(raw[5:])
                if data.get("stage") in STAGES and data.get("status") == "completed" and "duration_ms" in data:
                    stage_ms[data["stage"]].append(float(data["duration_ms"]))
        total_ms.append((time.perf_counter() - started) * 1000)
        first_ms.append(first or 0.0)
        time.sleep(3.2)                                   # stay under the 20/minute customer limit

    def stats(values):
        values = sorted(values)
        return {"n": len(values), "median_ms": round(statistics.median(values), 1),
                "p90_ms": round(values[max(0, int(len(values) * 0.9) - 1)], 1), "max_ms": round(values[-1], 1)}

    report = {"label": args.label, "turns": len(QUESTIONS),
              "time_to_first_event": stats(first_ms), "end_to_end": stats(total_ms),
              "stages": {k: stats(v) for k, v in stage_ms.items() if v}}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
