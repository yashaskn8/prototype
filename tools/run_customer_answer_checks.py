"""Reproducible synthetic-only customer answer regression runner.

Uses SQLite in memory and a new private Chroma directory. Does not use the
configured company database or indexes. No model download or external LLM.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["SERVY_DB_ENGINE"] = "sqlite3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SERVY_LLM_PROVIDER"] = "extractive"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
sys.argv = ["manage.py", "test"]  # existing fixture-provenance gate requires a real test invocation

from django.conf import settings
settings.DATABASES["default"]["TEST"]["NAME"] = ":memory:"
# Only synthetic users exist in this in-memory runner. Password validation and
# authentication stay enabled; avoid expensive production hashing per fixture.
settings.PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
settings.CHROMA_PERSIST_DIR = Path(tempfile.mkdtemp(prefix="answer_test_chroma_", dir=ROOT / "private_training_data"))
import django
django.setup()
from django.test.runner import DiscoverRunner


class Result(unittest.TextTestResult):
    latest = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Result.latest = self
        self.ledger = []
    def addFailure(self, test, err):
        self.record(test, err, "FAIL")
        super().addFailure(test, err)
    def addError(self, test, err):
        self.record(test, err, "ERROR")
        super().addError(test, err)
    def record(self, test, err, status):
        self.ledger.append({"test_id": test.id(), "attack_input": test.shortDescription() or test.id(),
            "expected_safe_behavior": "All isolated regression assertions pass.",
            "observed_behavior": "".join(traceback.format_exception(*err)),
            "root_cause": str(err[1]), "severity": "HIGH", "affected_component": test.__class__.__module__,
            "fix_implemented": "pending review", "regression_test_result": status})


class Runner(DiscoverRunner):
    def get_resultclass(self):
        return Result


if __name__ == "__main__":
    labels = ["core.tests_customer_answer", "core.tests_customer_answer_redteam", "core.tests_serving_truth",
              "core.tests_customer_mutations", "core.tests_real_company_isolation", "core.tests_safety_authority",
              "core.tests_customer_rag_endtoend", "core.tests_llm_answer_ranking",
              "core.tests_customer_acceptance_scenarios", "core.tests_diagnostics", "core.tests_diagnostics_hardening"]
    failures = Runner(verbosity=1, interactive=False).run_tests(labels)
    result = Result.latest
    report = {"kind": "isolated_backend_regression", "executed": result.testsRun,
              "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
              "failed": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "failures": result.ledger, "company_database_used": False, "company_indexes_used": False}
    target = ROOT / "private_training_data/customer_answer_validation"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"backend_{time.time_ns()}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "failures": len(report["failures"]), "report": str(path)}))
    raise SystemExit(bool(failures))
