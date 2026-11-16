"""Measure real Customer-gateway RAG latency and clean up only this run's conversations.

Usage:
  .venv\Scripts\python.exe -B tools\servy_latency_profile.py --access-file <file> [--label current-demo]

The profile separates conversation creation, time to first SSE event, time to
final result, total message-stream time and server-reported pipeline stages.
Every conversation created by this run is soft-deleted in a finally path.
"""
import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8002/demo-api/customer-support/conversations/"
QUESTIONS = [
    (1, "My split AC is not cooling."),
    (1, "Why is there ice on the coil?"),
    (1, "How often should a comprehensive service be performed?"),
    (2, "The VRF outdoor module shows a high-pressure alarm."),
    (2, "What causes a communication fault?"),
    (2, "What should be checked during routine inspection?"),
    (3, "The airflow from the ducts is weak."),
    (3, "What can cause water leakage?"),
    (3, "How often should the filters be inspected?"),
]
STAGES = {
    "dense_retrieval",
    "sparse_retrieval",
    "rank_fusion",
    "v5_reranking",
    "retrieval_pipeline",
    "customer_qwen_generation",
    "customer_evidence_validation",
    "customer_rag_execution",
}
SUCCESSFUL_DELETE_STATUSES = {200, 202, 204, 404}


def summarize(values):
    """Return distribution stats without inventing values for missing stages."""
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    p90_index = max(0, math.ceil(0.90 * len(clean)) - 1)
    return {
        "n": len(clean),
        "median_ms": round(statistics.median(clean), 1),
        "p90_ms": round(clean[p90_index], 1),
        "max_ms": round(clean[-1], 1),
    }


def soft_delete_created_conversation(session, headers, conversation_uuid, cleaned_uuids):
    """Delete one profiler-owned UUID, accepting a retry after an earlier deletion."""
    if conversation_uuid in cleaned_uuids:
        return 204
    response = session.delete(
        f"{BASE}{conversation_uuid}/",
        headers=headers,
        timeout=(10, 30),
    )
    if response.status_code not in SUCCESSFUL_DELETE_STATUSES:
        response.raise_for_status()
        raise RuntimeError(f"Unexpected cleanup HTTP status {response.status_code}")
    cleaned_uuids.add(conversation_uuid)
    return response.status_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-file", required=True)
    parser.add_argument("--label", default="profile")
    args = parser.parse_args()

    access_code = Path(args.access_file).read_text(encoding="utf-8").strip()
    if not access_code:
        parser.error("The access file is empty.")
    headers = {"Authorization": "Demo " + access_code}
    stage_ms = defaultdict(list)
    turns = []
    created_uuids = []
    cleaned_uuids = set()
    cleanup_failures = {}
    turns_by_uuid = {}
    session = requests.Session()

    def cleanup_conversation(conversation_uuid):
        # The caller supplies only UUIDs captured from this run's create responses.
        result = soft_delete_created_conversation(
            session, headers, conversation_uuid, cleaned_uuids,
        )
        cleanup_failures.pop(conversation_uuid, None)
        return result

    try:
        for index, (asset_id, question) in enumerate(QUESTIONS, start=1):
            turn = {
                "index": index,
                "asset_id": asset_id,
                "question": question,
                "conversation_create_ms": None,
                "conversation_create_http_status": None,
                "message_stream_http_status": None,
                "time_to_first_event_ms": None,
                "time_to_final_result_ms": None,
                "message_stream_total_ms": None,
                "turn_total_ms": None,
                "result_received": False,
                "stage_durations_ms": {},
                "stage_statuses": {},
                "cleanup_http_status": None,
            }
            turn_started = time.perf_counter()
            conversation_uuid = None

            try:
                create_started = time.perf_counter()
                created_response = session.post(
                    BASE,
                    headers=headers,
                    json={"asset_id": asset_id},
                    timeout=(10, 30),
                )
                turn["conversation_create_http_status"] = created_response.status_code
                created_response.raise_for_status()
                created_payload = created_response.json()
                conversation_uuid = created_payload.get("conversation_uuid")
                if not conversation_uuid:
                    raise RuntimeError("Conversation creation returned no conversation_uuid.")
                created_uuids.append(conversation_uuid)
                turns_by_uuid[conversation_uuid] = turn
                turn["conversation_create_ms"] = round(
                    (time.perf_counter() - create_started) * 1000, 1
                )

                stream_started = time.perf_counter()
                current_event = ""
                with session.post(
                    f"{BASE}{conversation_uuid}/messages-stream/",
                    headers=headers,
                    json={"message": question},
                    stream=True,
                    timeout=(10, 300),
                ) as response:
                    turn["message_stream_http_status"] = response.status_code
                    response.raise_for_status()

                    for raw in response.iter_lines(decode_unicode=True):
                        if not raw:
                            continue
                        if raw.startswith("event:"):
                            current_event = raw[6:].strip()
                            if turn["time_to_first_event_ms"] is None:
                                turn["time_to_first_event_ms"] = round(
                                    (time.perf_counter() - stream_started) * 1000, 1
                                )
                        elif raw.startswith("data:"):
                            if turn["time_to_first_event_ms"] is None:
                                turn["time_to_first_event_ms"] = round(
                                    (time.perf_counter() - stream_started) * 1000, 1
                                )
                            payload = json.loads(raw[5:].strip())
                            if current_event == "log" and payload.get("type") == "stage":
                                stage = payload.get("stage")
                                status_value = payload.get("status")
                                duration = payload.get("duration_ms")
                                if stage in STAGES:
                                    if status_value:
                                        turn["stage_statuses"][stage] = status_value
                                    if duration is not None:
                                        duration_ms = float(duration)
                                        turn["stage_durations_ms"][stage] = duration_ms
                                        if status_value == "completed":
                                            stage_ms[stage].append(duration_ms)
                            elif current_event == "result":
                                turn["result_received"] = True
                                turn["time_to_final_result_ms"] = round(
                                    (time.perf_counter() - stream_started) * 1000, 1
                                )
                            elif current_event == "error":
                                detail = payload.get("detail")
                                turn["stream_error"] = (
                                    str(detail)[:240] if detail else "RAG stream emitted an error event."
                                )

                turn["message_stream_total_ms"] = round(
                    (time.perf_counter() - stream_started) * 1000, 1
                )
                if not turn["result_received"] and not turn.get("stream_error"):
                    turn["stream_error"] = "Stream ended before a final result event."

            except Exception as exc:
                # Keep the other asset/questions measurable after one failed turn.
                turn["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            finally:
                if conversation_uuid:
                    try:
                        turn["cleanup_http_status"] = cleanup_conversation(conversation_uuid)
                    except Exception as exc:
                        cleanup_failures[conversation_uuid] = (
                            f"{type(exc).__name__}: {str(exc)[:200]}"
                        )
                        turn["cleanup_error"] = cleanup_failures[conversation_uuid]
                turn["turn_total_ms"] = round(
                    (time.perf_counter() - turn_started) * 1000, 1
                )
                turns.append(turn)

            # Stay below the configured 20-requests/minute customer RAG limit.
            if index < len(QUESTIONS):
                time.sleep(3.2)

    finally:
        # Retry transient cleanup failures. Never enumerate or delete unrelated history.
        for conversation_uuid in created_uuids:
            if conversation_uuid in cleaned_uuids:
                continue
            try:
                cleanup_status = cleanup_conversation(conversation_uuid)
                turn = turns_by_uuid.get(conversation_uuid)
                if turn is not None:
                    turn["cleanup_http_status"] = cleanup_status
                    turn.pop("cleanup_error", None)
            except Exception as exc:
                cleanup_failures[conversation_uuid] = (
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )
        session.close()

    report = {
        "label": args.label,
        "requested_turns": len(QUESTIONS),
        "successful_results": sum(1 for turn in turns if turn["result_received"]),
        "failed_turns": sum(1 for turn in turns if turn.get("error") or turn.get("stream_error")),
        "cleanup": {
            "created_by_this_run": len(created_uuids),
            "soft_deleted_or_already_absent": len(cleaned_uuids),
            "cleanup_failures": len(cleanup_failures),
        },
        "timings": {
            "conversation_creation": summarize(
                turn["conversation_create_ms"] for turn in turns
            ),
            "time_to_first_sse_event": summarize(
                turn["time_to_first_event_ms"] for turn in turns
            ),
            "time_to_final_sse_result": summarize(
                turn["time_to_final_result_ms"] for turn in turns
            ),
            "message_stream_end_to_end": summarize(
                turn["message_stream_total_ms"] for turn in turns
            ),
            "whole_turn_including_creation_and_cleanup": summarize(
                turn["turn_total_ms"] for turn in turns
            ),
        },
        "stages": {
            stage: summarize(values)
            for stage, values in sorted(stage_ms.items())
            if values
        },
        "turns": turns,
        "cleanup_failure_types": sorted({
            value.split(":", 1)[0] for value in cleanup_failures.values()
        }),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
