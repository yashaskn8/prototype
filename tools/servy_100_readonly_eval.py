"""Run the frozen Servy 100 questions with a SELECT-only DB account and private Chroma copy.

Only the private evaluation directory is writable. No company API or paid model API is used.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.servy_500_benchmark import (ROOT as SERVY_ROOT, SERVING_FILES,
    activate_index_copy, assets_and_sources, file_manifest, load_json,
    operational_counts, prepare_index_copy, private_folder, read_only_database,
    run_one, serial, serving, sha, write_json)
from tools.servy_500_report import evidence_check

BASE = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
BANK = BASE / "servy100_docgrounded_20261030_01" / "servy_100_document_grounded_questions.json"
PREVIOUS = BASE / "servy500_20261030_02"
OUTPUT = BASE / "servy100_live_20261030_01"
SECRET = BASE / "servy_eval_credentials" / "servy_eval_ro.dpapi"
EXPECTED = {168: (59, 14), 170: (59, 14), 171: (60, 13),
            172: (60, 13), 174: (61, 14)}


def configure_process():
    if not SECRET.is_file():
        raise RuntimeError("DPAPI-protected evaluation credential is missing.")
    cipher = SECRET.read_text(encoding="ascii")
    command = (
        "$cipher=[Console]::In.ReadToEnd().Trim(); "
        "$secure=ConvertTo-SecureString $cipher; "
        "$ptr=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure); "
        "try {[Console]::Out.Write([Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr))} "
        "finally {[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)}"
    )
    shell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
    if not shell:
        raise RuntimeError("PowerShell is required to decrypt the evaluation credential.")
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                            input=cipher, text=True, capture_output=True, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if not result.stdout:
        raise RuntimeError("Could not decrypt the evaluation credential.")
    os.environ.update({
        "MYSQL_USER": "servy_eval_ro", "MYSQL_PASSWORD": result.stdout,
        "MYSQL_HOST": "127.0.0.1", "MYSQL_DATABASE": "servy",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "SERVY_ALLOW_REMOTE_LLM": "0", "ANONYMIZED_TELEMETRY": "FALSE",
        "PYTHONDONTWRITEBYTECODE": "1",
        "DJANGO_SETTINGS_MODULE": "servy_rag.settings",
    })
    import django
    django.setup()


def verify_readonly_identity():
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE(), CURRENT_USER(), CURRENT_ROLE(), @@datadir, @@bind_address")
        database, user, role, datadir, bind = cursor.fetchone()
        cursor.execute("SHOW GRANTS FOR CURRENT_USER()")
        grants = [row[0].replace("`", "").upper() for row in cursor.fetchall()]
    if (database != "servy" or user != "servy_eval_ro@127.0.0.1" or role != "NONE"
        or bind != "127.0.0.1" or Path(datadir).resolve() != Path("C:/mysql/data").resolve()):
        raise RuntimeError("Evaluation connection is not the expected SELECT-only local account.")
    if (len(grants) != 2 or not any(g.startswith("GRANT USAGE ON *.* TO ") for g in grants)
        or not any(g.startswith("GRANT SELECT ON SERVY.* TO ") for g in grants)
        or any("GRANT OPTION" in g for g in grants)):
        raise RuntimeError("Effective grants or active roles exceed SELECT on servy.")
    return {"database": database, "account": user, "active_role": role,
            "grants": ["USAGE on *.*", "SELECT on servy.*"]}


def validate_bank():
    bank = load_json(BANK)
    rows = bank["questions"]
    if len(rows) != 100 or len({row["id"] for row in rows}) != 100:
        raise RuntimeError("The frozen bank must contain exactly 100 unique IDs.")
    counts = Counter(row["source_document_id"] for row in rows)
    if counts != {doc_id: 20 for doc_id in EXPECTED}:
        raise RuntimeError("The frozen five-document distribution changed.")
    for row in rows:
        if ((row["asset_id"], row["customer_id"]) != EXPECTED[row["source_document_id"]]
            or len(row["question"]) != row["question_char_count"]
            or hashlib.sha256(row["question"].encode()).hexdigest() != row["question_sha256"]):
            raise RuntimeError("Question text, hash, or asset binding changed.")
    return rows


def document_hashes(tenant):
    from django.conf import settings
    from core.models import KnowledgeDocument
    from core.services.customer_isolation import is_verified_real_document
    media = Path(settings.MEDIA_ROOT).resolve()
    output = {}
    for doc_id, (asset_id, customer_id) in EXPECTED.items():
        doc = KnowledgeDocument.objects.select_related(
            "tenant", "asset", "customer", "product", "category", "domain"
        ).get(pk=doc_id, tenant=tenant)
        if (doc.asset_id != asset_id or doc.customer_id not in (None, customer_id)
            or not doc.file or not doc.checksum_sha256 or not is_verified_real_document(doc)
            or doc.is_confidential or doc.disable_sharing or not doc.is_rag_enabled):
            raise RuntimeError("Selected original document is no longer eligible.")
        path = Path(doc.file.path).resolve()
        if not path.is_relative_to(media) or not path.is_file():
            raise RuntimeError("Selected document is not a local company attachment.")
        digest = sha(path)
        if digest.lower() != doc.checksum_sha256.lower():
            raise RuntimeError("Original company attachment checksum changed.")
        output[str(doc_id)] = digest
    return output


def verify_source_scope(rows):
    tenant, assets, coverage, sources = assets_and_sources(rows)
    if set(assets) != {59, 60, 61}:
        raise RuntimeError("Selected asset/customer scope changed.")
    for row in rows:
        asset = assets[row["asset_id"]]
        if asset.customer_id != row["customer_id"]:
            raise RuntimeError("Question customer does not own asset.")
        allowed = {(c["document_id"], c["chunk_id"]): c
                   for c in sources[str(asset.id)]}
        for reference in row["evidence_reference"]:
            passage = allowed.get((reference["document_id"], reference["chunk_id"]))
            if (not passage or passage["sha256"] != reference["chunk_sha256"]
                or passage["title"] != row["source_document_name"]):
                raise RuntimeError("Question evidence no longer matches an authorized passage.")
    return tenant, assets, coverage, sources


def verify_local_serving():
    from django.conf import settings
    from core.ml.reranker import require_verified_v5_reranker
    from core.ml.embeddings import get_profile, load_model
    if (settings.SERVY_SERVING_EMBEDDING_PROFILE != "bge-small"
        or str(getattr(settings, "SERVY_RETRIEVAL_MODE", "hybrid")).lower() != "hybrid"
        or not settings.SERVY_RERANKER_ENABLED or settings.SERVY_RERANKER_POOL_SIZE != 5):
        raise RuntimeError("Expected BGE-small, hybrid, V5 pool-5 serving configuration.")
    url = urlparse(settings.SERVY_OLLAMA_URL)
    if (settings.SERVY_ALLOW_REMOTE_LLM or url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or settings.SERVY_LLM_PROVIDER not in {"auto", "extractive", "ollama"}):
        raise RuntimeError("Answer generation is not restricted to approved local engines.")
    model = load_model(get_profile("company-bge-small"), allow_download=False)
    if model.get_embedding_dimension() != 384:
        raise RuntimeError("Local BGE-small model is not ready.")
    require_verified_v5_reranker()
    return serving()


def prepare(folder, rows):
    from django.conf import settings
    frozen_path = folder / "run_freeze.json"
    prior_manifest = load_json(PREVIOUS / "production_index_manifest.json")
    original_index = Path(settings.CHROMA_PERSIST_DIR).resolve()
    if file_manifest(original_index) != prior_manifest:
        raise RuntimeError("Original Chroma index differs from the frozen prior manifest.")
    with read_only_database():
        tenant, assets, coverage, sources = verify_source_scope(rows)
        documents = document_hashes(tenant)
        counts = operational_counts()
    if frozen_path.exists():
        frozen = load_json(frozen_path)
        if (frozen["bank_sha256"] != sha(BANK)
            or frozen["sources"] != sources or frozen["coverage"] != coverage
            or frozen["document_sha256"] != documents
            or frozen["original_index_manifest"] != prior_manifest
            or frozen["serving_code_sha256"] != {name: sha(SERVY_ROOT / name) for name in SERVING_FILES}):
            raise RuntimeError("A frozen input, source, document, or serving file changed.")
        activate_index_copy(folder)
        if verify_local_serving() != frozen["serving"]:
            raise RuntimeError("Serving configuration changed since preparation.")
        return frozen, assets, sources
    if (folder / "chroma_index_copy").exists():
        raise RuntimeError("Unfrozen evaluation index copy exists; refusing to use it.")
    source, target = prepare_index_copy(folder)
    if source != original_index or file_manifest(target) != prior_manifest:
        raise RuntimeError("Private index copy does not match original frozen index.")
    activate_index_copy(folder)
    config = verify_local_serving()
    frozen = {"bank_sha256": sha(BANK), "coverage": coverage, "sources": sources,
              "document_sha256": documents, "original_index_manifest": prior_manifest,
              "serving_code_sha256": {name: sha(SERVY_ROOT / name) for name in SERVING_FILES},
              "serving": config, "operational_counts_before": counts,
              "database_account": "servy_eval_ro@127.0.0.1", "paid_api_budget_inr": 0}
    write_json(frozen_path, frozen)
    return frozen, assets, sources


def run(folder, rows, limit):
    frozen, assets, sources = prepare(folder, rows)
    source_lookup = {asset_id: {(c["document_id"], c["chunk_id"]): c["text"]
                                for c in chunks} for asset_id, chunks in sources.items()}
    target = folder / "servy_100_actual_responses.json"
    results = load_json(target) if target.exists() else []
    done = {r["id"] for r in results}
    expected = {r["id"] for r in rows}
    if len(done) != len(results) or not done.issubset(expected):
        raise RuntimeError("Saved responses have duplicate or unknown IDs.")
    inflight = folder / "inflight.json"
    if inflight.exists():
        intent = load_json(inflight)
        if intent["id"] not in done:
            raise RuntimeError("An interrupted request may have run; inspect privately before resuming.")
        inflight.unlink()
    called = 0
    with read_only_database():
        if operational_counts() != frozen["operational_counts_before"]:
            raise RuntimeError("Operational record counts changed since preparation.")
        for row in rows:
            if row["id"] in done or called >= limit:
                continue
            write_json(inflight, {"id": row["id"], "started_at_epoch": time.time()})
            result = {**row, **run_one(row, assets[row["asset_id"]])}
            if result["classification"] == "NEEDS_HUMAN_REVIEW":
                result["classification"] = "ERROR"
                result["classification_reason"] += " Response shape cannot establish partial or complete behavior."
            check = evidence_check(result, source_lookup)
            result["citation_support_assessment"] = check
            result["answer_quality_assessment"] = {
                "factual_correctness": "NEEDS_HUMAN_REVIEW",
                "question_understanding": "NEEDS_HUMAN_REVIEW",
                "appropriate_refusal": "NEEDS_HUMAN_REVIEW",
                "expected_behavior_is_unapproved": True,
            }
            results.append(result)
            write_json(target, results)
            inflight.unlink()
            done.add(row["id"])
            called += 1
            print(json.dumps({"completed": len(done), "id": row["id"],
                              "classification": result["classification"]}), flush=True)
            if operational_counts() != frozen["operational_counts_before"]:
                raise RuntimeError("Operational database record count changed during evaluation.")
    from django.conf import settings
    if file_manifest(Path(frozen_index_path(settings, folder))) != frozen["original_index_manifest"]:
        raise RuntimeError("Original Chroma index changed during evaluation.")
    print(json.dumps({"selected": 100, "completed": len(done), "calls_this_invocation": called,
                      "counts": dict(Counter(r["classification"] for r in results))}), flush=True)


def frozen_index_path(settings, folder):
    # activate_index_copy changes settings only in this process. The prior freeze
    # records the original path, so derive it from its parent in the old freeze.
    prior = load_json(PREVIOUS / "freeze.json")
    return prior["production_index_path"]


def report(folder, rows):
    frozen = load_json(folder / "run_freeze.json")
    results = load_json(folder / "servy_100_actual_responses.json")
    if len(results) != 100 or {r["id"] for r in results} != {r["id"] for r in rows}:
        raise RuntimeError("All 100 unique saved responses are required before reporting.")
    counts = Counter(r["classification"] for r in results)
    categories = {}
    for key in ("source_document_id", "asset_id", "difficulty", "writing_style", "intent"):
        groups = defaultdict(Counter)
        for row in results:
            groups[str(row[key])][row["classification"]] += 1
        categories[key] = {k: dict(v) for k, v in groups.items()}
    checks = Counter((r.get("citation_support_assessment") or {}).get("status") for r in results)
    summary = {"total_questions": 100, "actually_executed": 100,
               "classification_counts": {k: counts[k] for k in
                   ("ANSWERED", "REFUSED", "PARTIAL", "CLARIFICATION", "ERROR", "NOT_RUN_UNAUTHORIZED")},
               "behavior_denominator": 100, "by_category": categories,
               "mechanical_citation_checks": dict(checks),
               "machine_checkable_exact_source_quotes": checks["SOURCE_QUOTES_EXACT_SEMANTICS_NOT_VERIFIED"],
               "verified_factual_accuracy": None, "verified_correct_refusals": None,
               "verified_incorrect_refusals": None, "expert_review_status": "PENDING_REVIEW",
               "paid_inference_api_calls": 0, "paid_api_cost_inr": 0,
               "read_only_account": frozen["database_account"],
               "limitations": ["Direct service call plus unsaved customer formatter, not HTTP customer login.",
                               "Exact source quotes do not prove semantic answer correctness.",
                               "Source files identify themselves as sample references, not exact manufacturer manuals."]}
    write_json(folder / "servy_100_evaluation_summary.json", summary)
    lines = ["# Servy 100 adversarial RAG evaluation", "",
             "All 100 questions were submitted through the local Servy service path with an unsaved customer interaction.",
             "The SELECT-only database account and private Chroma copy protected originals. Paid inference API cost: ₹0.", "",
             "## Behavioral outcomes", "", "| Outcome | Count |", "|---|---:|"]
    lines += [f"| {k} | {counts[k]} |" for k in summary["classification_counts"]]
    lines += ["", "## Evidence and accuracy limits", "",
              f"Exact source-quote checks: {checks['SOURCE_QUOTES_EXACT_SEMANTICS_NOT_VERIFIED']}/100 total questions.",
              "Citation ID and verbatim quote checks are mechanical; technical relevance, completeness, and safe customer action need expert review.",
              "No independently verified factual accuracy or refusal-correctness percentage is claimed.", "",
              "## Question-by-question results", "",
              "| Question ID | Document | Asset | Outcome | Source check | Engine | Citations |",
              "|---|---:|---:|---|---|---|---:|"]
    for row in results:
        raw, formatted = row.get("raw") or {}, row.get("formatted") or {}
        check = row.get("citation_support_assessment") or {}
        lines.append(f"| {row['id']} | {row['source_document_id']} | {row['asset_id']} | "
                     f"{row['classification']} | {check.get('status','NOT_VERIFIED')} | "
                     f"{raw.get('engine','none')} | {len(formatted.get('citations') or [])} |")
    failed = [r["id"] for r in results if (r.get("citation_support_assessment") or {}).get("status") == "CITATION_OR_SOURCE_CHECK_FAILED"]
    lines += ["", "## Citation issues", "", ", ".join(failed) if failed else "No mechanical citation mismatch found.",
              "", "## Unsupported answers and refusals", "",
              "Published claims without exact eligible source support are listed as citation issues above. Semantic unsupported claims and inappropriate refusals remain PENDING_REVIEW.",
              "", "## Findings and next steps", "",
              "Review the private full responses for relevance to each question and for completeness of all four answer points.",
              "Review refusals for questions with eligible original passages before changing retrieval or grounding code.",
              "Do not treat these generic sample reference attachments as exact manufacturer specifications.", ""]
    (folder / "servy_100_adversarial_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": summary["classification_counts"], "citation_checks": dict(checks)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "report"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.limit <= 100:
        raise SystemExit("--limit must be between 0 and 100")
    configure_process()
    identity = verify_readonly_identity()
    rows = validate_bank()
    folder = private_folder(OUTPUT)
    if args.phase == "report":
        report(folder, rows)
    elif args.phase == "prepare":
        prepare(folder, rows)
        print(json.dumps({"prepared": True, "readonly_identity": identity,
                          "question_count": len(rows), "private_index_copy": True,
                          "paid_api_budget_inr": 0}))
    else:
        run(folder, rows, args.limit)


if __name__ == "__main__":
    main()
