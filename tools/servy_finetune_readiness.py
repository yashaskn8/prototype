"""Fail-closed Servy fine-tuning dataset preflight. NO TRAINING OR MODEL DOWNLOADS.

Usage:
 python tools/servy_finetune_readiness.py --train path/train.jsonl \
   --dev path/dev.jsonl --corpus path/corpus.jsonl \
   --stage2 path/EXACT_USED_27.csv --out-dir ml_artifacts/ft_readiness

All results are *structural*. Technical correctness, equipment suitability,
PDF extraction fidelity and positive/negative labels require source-expert review.
The tool never modifies input files or production systems.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def load_rows(path, sheet="AI Triage - 50"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required private file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for n, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if text.strip():
                try:
                    row = json.loads(text)
                    if not isinstance(row, dict):
                        raise ValueError("JSON object expected")
                    rows.append(row)
                except (ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}:{n}: invalid JSONL object") from error
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    elif path.suffix.lower() == ".xlsx":
        try:
            from openpyxl import load_workbook
        except ImportError as error:
            raise RuntimeError("Install openpyxl to read XLSX, or export a CSV") from error
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            if sheet not in workbook.sheetnames:
                raise ValueError(f"Missing workbook sheet {sheet!r}")
            iterator = workbook[sheet].values
            columns = [str(v or "").strip() for v in next(iterator)]
            rows = [dict(zip(columns, values)) for values in iterator
                    if any(v is not None for v in values)]
        finally:
            workbook.close()
    else:
        raise ValueError(f"Unsupported file extension for {path}")
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def normal(text):
    return " ".join(str(text or "").casefold().split())


def canonical_question(text):
    """Normalize case, whitespace and punctuation (not paraphrases)."""
    return " ".join(re.findall(r"\w+", normal(text)))


def _ids(row, plural, singular):
    raw = row.get(plural)
    if raw is None:
        raw = row.get(singular)
    if raw is None or raw == "":
        return []
    if not isinstance(raw, list):
        raw = [raw]
    return [str(i).strip() for i in raw if i is not None and str(i).strip()]


def _first(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def build_corpus(rows):
    by_id, by_text = {}, defaultdict(list)
    for index, row in enumerate(rows, 1):
        cid = _first(row, "chunk_id", "passage_id")
        doc = _first(row, "document_id", "source_document_id")
        tenant = _first(row, "tenant_id")
        passage = _first(row, "passage", "text")
        if not (cid and doc and tenant and passage):
            raise ValueError(f"Corpus row {index} missing id/document/tenant/passage")
        if cid in by_id:
            raise ValueError(f"Duplicated corpus chunk ID: {cid}")
        entry = {
            "chunk_id": cid, "document": (tenant, doc),
            "tenant_id": tenant, "passage": passage,
            "passage_key": normal(passage),
        }
        by_id[cid] = entry
        by_text[entry["passage_key"]].append(entry)
    return by_id, by_text


def inspect_split(rows, name, by_id, by_text, errors, warnings):
    resolved = []
    seen_ids = set()
    for index, row in enumerate(rows, 1):
        qid = _first(row, "query_id") or f"{name}:{index}"
        question = _first(row, "query_text", "query")
        query_key = canonical_question(question)
        if not query_key:
            errors.append(f"{name}/{qid}: empty question")
            continue
        if qid in seen_ids:
            errors.append(f"{name}/{qid}: duplicated query_id")
        seen_ids.add(qid)
        tenant = _first(row, "tenant_id")
        pos = _ids(row, "positive_chunk_ids", "positive_chunk_id")
        if not pos:
            pos = _ids(row, "chunk_ids", "chunk_id")
        positive_text = _first(row, "positive_text", "positive")
        positive = []
        if pos:
            for cid in pos:
                if cid not in by_id:
                    errors.append(f"{name}/{qid}: unknown positive chunk {cid}")
                elif tenant and by_id[cid]["tenant_id"] != tenant:
                    errors.append(f"{name}/{qid}: positive chunk {cid} cross-tenant")
                else:
                    positive.append(by_id[cid])
            if positive_text and positive and normal(positive_text) not in {
                c["passage_key"] for c in positive
            }:
                errors.append(f"{name}/{qid}: positive passage text differs from indexed chunk")
        elif positive_text:
            matches = by_text.get(normal(positive_text), [])
            if tenant:
                matches = [c for c in matches if c["tenant_id"] == tenant]
            if len(matches) != 1:
                errors.append(f"{name}/{qid}: untraceable/ambiguous positive passage text ({len(matches)} matches)")
            else:
                positive = matches
        else:
            errors.append(f"{name}/{qid}: positive passage unresolvable")
        if not positive:
            continue
        if not tenant:
            tenants = {c["tenant_id"] for c in positive}
            if len(tenants) != 1:
                errors.append(f"{name}/{qid}: ambiguous tenant identity")
            else:
                tenant = next(iter(tenants))
        elif any(c["tenant_id"] != tenant for c in positive):
            errors.append(f"{name}/{qid}: mixed-tenant positive passages")

        negatives = []
        n_ids = _ids(row, "negative_chunk_ids", "negative_chunk_id")
        if n_ids:
            for cid in n_ids:
                if cid not in by_id:
                    errors.append(f"{name}/{qid}: unknown negative chunk {cid}")
                elif by_id[cid]["tenant_id"] != tenant:
                    errors.append(f"{name}/{qid}: negative chunk {cid} cross-tenant")
                else:
                    negatives.append(by_id[cid])
        negative_text = _first(row, "candidate_negative", "negative_text", "negative")
        if negative_text:
            matches = [c for c in by_text.get(normal(negative_text), [])
                       if c["tenant_id"] == tenant]
            if len(matches) != 1:
                errors.append(f"{name}/{qid}: untraceable/ambiguous negative passage text ({len(matches)} matches)")
            else:
                if n_ids and matches[0]["chunk_id"] not in n_ids:
                    errors.append(f"{name}/{qid}: negative passage text does not match cited negative ID")
                else:
                    negatives.append(matches[0])
        negatives = list({x["chunk_id"]: x for x in negatives}.values())
        group = _first(row, "source_group")
        if group.startswith("doc-"):
            # Weak heuristic metadata must not override canonical corpus document ID.
            pass
        if group and len({c["document"] for c in positive}) > 1:
            warnings.append(f"{name}/{qid}: one source_group spans multiple positive source documents")

        pos_ids = {c["chunk_id"] for c in positive}
        pos_texts = {c["passage_key"] for c in positive}
        for neg in negatives:
            if neg["chunk_id"] in pos_ids or neg["passage_key"] in pos_texts:
                errors.append(f"{name}/{qid}: positive passage incorrectly labeled as negative")
            if neg["document"] in {x["document"] for x in positive}:
                warnings.append(f"{name}/{qid}: same-document hard negative requires expert verification")
            overlap = set(neg["passage_key"].split()) & set(next(iter(pos_texts)).split())
            denominator = max(1, len(set(next(iter(pos_texts)).split())))
            if len(overlap) / denominator > .8:
                warnings.append(f"{name}/{qid}: negative shares >80% of positive vocabulary")
        approval = normal(row.get("expert_signoff") or row.get("review_status") or "")
        if name == "stage2" and approval not in ("approved", "verified", "yes"):
            warnings.append(f"{name}/{qid}: expert approval not established; research only")
        resolved.append({
            "qid": qid, "question": question, "query_key": query_key,
            "tenant": tenant, "group": group,
            "pos_docs": {x["document"] for x in positive},
            "neg_docs": {x["document"] for x in negatives},
            "pos_ids": pos_ids,
            "neg_ids": {x["chunk_id"] for x in negatives},
            "pos_texts": pos_texts,
            "neg_texts": {x["passage_key"] for x in negatives},
            "verified": approval in ("approved", "verified", "yes"),
        })
    return resolved


def pair_collisions(a, b, a_name, b_name, failures, observations):
    """Cases belonging to different splits must be group-independent."""
    aq = {r["query_key"] for r in a}
    ap = {p for r in a for p in r["pos_texts"]}
    ad = {d for r in a for d in r["pos_docs"] | r["neg_docs"]}
    ag = {r["group"].casefold().strip() for r in a if r["group"]}
    for row in b:
        prefix = f"{a_name}↔{b_name}/{row['qid']}"
        if row["query_key"] in aq:
            failures.append(f"{prefix}: duplicate normalized question")
        if row["pos_texts"] & ap:
            failures.append(f"{prefix}: exact positive passage collision")
        if row["pos_docs"] & ad:
            failures.append(f"{prefix}: shared training positive/negative source document")
        if row["pos_ids"] & {p for r in a for p in r["neg_ids"]}:
            failures.append(f"{prefix}: validation positive was used as training negative")
        if row["group"] and row["group"].casefold().strip() in ag:
            failures.append(f"{prefix}: source_group collision")
        # Stage2 negative passages from validation/holdout are also contamination.
        if row["neg_docs"] & ad:
            observations.append(f"{prefix}: negative source document also appears in other split")


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assess(train, dev, corpus, *, stage2=None, holdout=None):
    failures, warnings = [], []
    by_id, by_text = build_corpus(corpus)
    training = inspect_split(train, "train", by_id, by_text, failures, warnings)
    validation = inspect_split(dev, "dev", by_id, by_text, failures, warnings)
    if not training or not validation:
        failures.append("Both train and development need resolvable examples")
    pair_collisions(training, validation, "train", "dev", failures, warnings)

    # Detect contradictory relevance labels for the same query.
    by_question = defaultdict(lambda: {"pos": set(), "neg": set()})
    for item in training + validation:
        by_question[item["query_key"]]["pos"].update(item["pos_texts"])
        by_question[item["query_key"]]["neg"].update(item["neg_texts"])
    for query_key, labels in by_question.items():
        if labels["pos"] & labels["neg"]:
            failures.append(f"Contradictory positive/negative labels for query: {query_key[:90]}")
    stage2_rows = []
    if stage2 is not None:
        stage2_rows = inspect_split(stage2, "stage2", by_id, by_text, failures, warnings)
        # Stage2 may repeat the Stage1 TRAIN positives intentionally.
        pair_collisions(stage2_rows, validation, "stage2", "dev", failures, warnings)
        train_docs = {d for r in training for d in r["pos_docs"]}
        for row in stage2_rows:
            if not row["pos_docs"].issubset(train_docs):
                failures.append(f"stage2/{row['qid']}: positive source is not in Stage1 training documents")
        for row in stage2_rows:
            for train_row in training:
                if (row["query_key"] == train_row["query_key"]
                        and row["neg_texts"] & train_row["pos_texts"]):
                    failures.append(f"stage2/{row['qid']}: hard negative equals a known train positive")
        if len(stage2_rows) < 8:
            warnings.append("Very few Stage2 triplets; domain adaptation prone to overfitting")
        if not all(row["verified"] for row in stage2_rows):
            warnings.append("Stage2 labels not all expert-approved; no verified training claim")
    if holdout is not None:
        # Do NOT score holdout accuracy in this tool.
        unseen = []
        for idx, row in enumerate(holdout, 1):
            if not (row.get("positive_chunk_ids") or row.get("positive_chunk_id")
                    or row.get("chunk_id")):
                question = _first(row, "query_text", "query")
                unseen.append({
                    "qid": _first(row, "query_id") or f"holdout:{idx}",
                    "query_key": canonical_question(question),
                    "pos_docs": set(), "neg_docs": set(), "pos_texts": set(),
                    "pos_ids": set(), "neg_ids": set(),
                    "group": _first(row, "source_group"),
                })
            else:
                # Some independent holdout chunks may not be in a tenant-only export.
                # Do not silently treat missing gold as safe: report incomplete audit.
                missing = [cid for cid in _ids(row, "positive_chunk_ids", "positive_chunk_id")
                           if cid not in by_id]
                if missing:
                    failures.append(
                        f"holdout/{_first(row, 'query_id')}: gold ID not in exported corpus; "
                        "cannot certify guide-level holdout isolation")
                    continue
                unseen.extend(inspect_split([row], "holdout", by_id, by_text, failures, warnings))
        pair_collisions(training + stage2_rows, unseen, "train+stage2", "holdout",
                        failures, warnings)

    def coverage(rows):
        return {
            "rows": len(rows),
            "unique_questions": len({r["query_key"] for r in rows}),
            "unique_positive_passages": len({p for r in rows for p in r["pos_texts"]}),
            "unique_positive_documents": len({p for r in rows for p in r["pos_docs"]}),
            "negative_passages": sum(len(r["neg_ids"]) for r in rows),
            "expert_approved": sum(r["verified"] for r in rows),
        }

    return {
        "status": "BLOCKED" if failures else "RESEARCH_PREFLIGHT_PASSED",
        "production_eligible": False,
        "evidence_quality": "UNVERIFIED_UNTIL_AUTHORIZED_EXPERT_REVIEW",
        "training_loss_guarantee": "NONE; aim for generalization, not zero loss",
        "training": coverage(training),
        "development": coverage(validation),
        "stage2": coverage(stage2_rows) if stage2 is not None else "NOT_PROVIDED",
        "corpus_chunks": len(by_id),
        "blocking_findings": sorted(set(failures)),
        "review_findings": sorted(set(warnings)),
        "limitations": [
            "Only exact source IDs, passages, source_group and normalized questions are checked",
            "Near paraphrases, duplicate guides with changed identity and misleading technical labels require expert adjudication",
            "Corpus candidate filtering is tenant-only; not the full live authorization policy",
            "This is NOT a model training run, holdout score or readiness proof for production",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--stage2", help="Exact used Stage2 27 triplets (not unfiltered 50-row review)")
    parser.add_argument("--holdout", help="Optional independent holdout for overlap checks ONLY")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    paths = {k: getattr(args, k) for k in ("train", "dev", "corpus", "stage2", "holdout")}
    rows = {k: load_rows(v) if v else None for k, v in paths.items()}
    report = assess(rows["train"], rows["dev"], rows["corpus"],
                    stage2=rows["stage2"], holdout=rows["holdout"])
    report["input_sha256"] = {k: _sha256(v) for k, v in paths.items() if v}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "readiness_report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved at {target}")
    if report["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
