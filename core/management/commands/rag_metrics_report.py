"""Summarize the per-request RAG metrics log (latency percentiles, outcomes, model rates).

  python manage.py rag_metrics_report [--log <file>] [--last N] [--json]
"""
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


def _pct(values, p):
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(p * len(ordered)) - 1)], 1) if ordered else None


def summarize(rows):
    totals = [r["total_ms"] for r in rows if isinstance(r.get("total_ms"), (int, float))]
    cited = [r for r in rows if r.get("status") in ("success", "degraded")]
    stage_names = sorted({k for r in rows for k in (r.get("stage_ms") or {})})
    return {
        "requests": len(rows),
        "latency_ms": {"p50": _pct(totals, .5), "p90": _pct(totals, .9), "p95": _pct(totals, .95),
                       "max": _pct(totals, 1.0)},
        "status": dict(Counter(r.get("status") for r in rows)),
        "response_mode": dict(Counter(r.get("response_mode") for r in rows)),
        "deadline_exceeded": sum(bool(r.get("deadline_exceeded")) for r in rows),
        "reranker_status": dict(Counter(r.get("reranker_status") for r in rows)),
        "generation_status": dict(Counter(r.get("generation_status") for r in rows)),
        "generated_share_of_cited": round(sum(bool(r.get("generation_applied")) for r in cited) / len(cited), 3)
        if cited else None,
        "stage_median_ms": {s: round(statistics.median([r["stage_ms"][s] for r in rows if s in (r.get("stage_ms") or {})]), 1)
                            for s in stage_names},
        "rss_mb_max": max((r.get("rss_mb") or 0 for r in rows), default=None),
    }


class Command(BaseCommand):
    help = "Summarize the bounded per-request RAG metrics log."

    def add_arguments(self, parser):
        parser.add_argument("--log", default=None)
        parser.add_argument("--last", type=int, default=0)
        parser.add_argument("--json", action="store_true")

    def handle(self, *args, **options):
        path = options["log"] or str(getattr(settings, "SERVY_RAG_METRICS_LOG", "") or "")
        if not path:
            raise CommandError("No metrics log configured (SERVY_RAG_METRICS_LOG or --log).")
        files = sorted(Path(path).parent.glob(Path(path).name + "*"), key=lambda p: p.stat().st_mtime)
        rows = []
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        rows.sort(key=lambda r: r.get("time_utc") or "")
        if options["last"]:
            rows = rows[-options["last"]:]
        report = summarize(rows)
        self.stdout.write(json.dumps(report, indent=None if options["json"] else 1))
