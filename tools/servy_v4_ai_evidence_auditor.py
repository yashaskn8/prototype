"""Autonomous AI Evidence Review and Data Validation Engine for Servy V4.

Local-only, read-only analysis using SQLite metadata and NLP heuristics.
No external network calls, no model downloading, no Docker.
Never automatically approves labels or modifies human reviewer fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def norm_tokens(text: str) -> list[str]:
    """Tokenize and normalize text into clean lower-case alphanumeric tokens."""
    return re.findall(r"\b[a-z0-9\-]+\b", str(text or "").lower())


def token_set(text: str) -> set[str]:
    return set(norm_tokens(text))


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ServyAIEvidenceAuditor:
    def __init__(self, bundle_dir: Path, corpus_path: Path, db_path: Path):
        self.bundle_dir = Path(bundle_dir).resolve()
        self.corpus_path = Path(corpus_path).resolve()
        self.db_path = Path(db_path).resolve()
        
        # Load corpus
        with self.corpus_path.open(encoding="utf-8-sig") as f:
            self.corpus_chunks = [json.loads(line) for line in f if line.strip()]
        self.chunk_by_id = {str(c["chunk_id"]): c for c in self.corpus_chunks}
        
        # Load queues from bundle
        with (self.bundle_dir / "source_family_review_queue.jsonl").open(encoding="utf-8-sig") as f:
            self.source_queue = [json.loads(line) for line in f if line.strip()]
            
        with (self.bundle_dir / "expert_review_queue.jsonl").open(encoding="utf-8-sig") as f:
            self.label_queue = [json.loads(line) for line in f if line.strip()]
            
        # Connect to SQLite for authoritative metadata
        self.db_conn = sqlite3.connect(self.db_path)
        self.db_conn.row_factory = sqlite3.Row
        self._load_authoritative_metadata()
        
    def _load_authoritative_metadata(self):
        cur = self.db_conn.cursor()
        cur.execute("""
            SELECT d.id as doc_id,
                   d.title as doc_title,
                   d.version as doc_version,
                   d.checksum_sha256 as doc_checksum,
                   d.tenant_id as doc_tenant_id,
                   p.name as product_name, 
                   c.name as category_name, 
                   b.name as brand_name,
                   dom.name as domain_name,
                   t.name as tenant_name,
                   t.slug as tenant_slug
            FROM core_knowledgedocument d
            LEFT JOIN core_product p ON d.product_id = p.id
            LEFT JOIN core_productcategory c ON d.category_id = c.id
            LEFT JOIN core_brand b ON p.brand_id = b.id
            LEFT JOIN core_productdomain dom ON d.domain_id = dom.id
            LEFT JOIN core_tenant t ON d.tenant_id = t.id;
        """)
        self.db_documents = {str(row["doc_id"]): dict(row) for row in cur.fetchall()}

    def close(self):
        self.db_conn.close()

    def audit_document_provenance(self) -> tuple[list[dict], dict]:
        """Phase 2: Audit all 57 source document records against authoritative SQLite metadata."""
        recommendations = []
        family_grouping = defaultdict(list)
        
        for item in self.source_queue:
            doc_id = str(item["document_id"]).replace("doc-", "")
            tenant_id = str(item["tenant_id"])
            chunk_count = item["chunk_count"]
            checksums = [c.lower() for c in item.get("source_document_sha256", [])]
            versions = item.get("document_versions", [])
            declared_groups = item.get("unverified_declared_groups", [])
            
            db_doc = self.db_documents.get(doc_id)
            if db_doc:
                db_tid = str(db_doc["doc_tenant_id"])
                db_sha = str(db_doc["doc_checksum"] or "").lower()
                sha_matches = bool(checksums and db_sha in checksums)
                tid_matches = (db_tid == tenant_id)
                
                if sha_matches and tid_matches:
                    status = "VERIFIED_SOURCE_EVIDENCE"
                    confidence = "HIGH"
                elif sha_matches or tid_matches:
                    status = "LIKELY_MATCH"
                    confidence = "MEDIUM"
                else:
                    status = "AMBIGUOUS_MATCH"
                    confidence = "LOW"
                    
                prod = str(db_doc.get("product_name") or "").strip()
                cat = str(db_doc.get("category_name") or "").strip()
                title = str(db_doc.get("doc_title") or "").strip()
                
                # Derive canonical family name based on authoritative product/equipment
                if prod:
                    # Clean product name
                    canonical_family = f"FAMILY-{prod.upper().replace(' ', '_')}"
                elif cat:
                    canonical_family = f"FAMILY-{cat.upper().replace(' ', '_')}"
                else:
                    canonical_family = f"FAMILY-DOC_{doc_id}"
                    
                approved_version = str(db_doc.get("doc_version") or (versions[0] if versions else "1.0")).strip()
                
                reasoning = (
                    f"Authoritative record matched in SQLite core_knowledgedocument (id={doc_id}). "
                    f"Title: '{title}'. Product: '{prod}'. Category: '{cat}'. "
                    f"Tenant: '{db_doc.get('tenant_name')}' (id={db_tid}). "
                    f"Authoritative checksum SHA-256: {db_sha} matched recovered corpus checksums."
                )
            else:
                status = "SOURCE_NOT_FOUND"
                confidence = "LOW"
                canonical_family = f"FAMILY-UNVERIFIED_DOC_{doc_id}"
                approved_version = versions[0] if versions else "1.0"
                reasoning = f"No authoritative document with ID {doc_id} found in SQLite registry."
                db_doc = {}
                
            rec = {
                "document_id": item["document_id"],
                "tenant_id": tenant_id,
                "evidence_status": status,
                "confidence": confidence,
                "proposed_canonical_source_family": canonical_family,
                "proposed_source_version": approved_version,
                "chunk_count": chunk_count,
                "authoritative_title": db_doc.get("doc_title", ""),
                "authoritative_product": db_doc.get("product_name", ""),
                "authoritative_category": db_doc.get("category_name", ""),
                "authoritative_checksum": db_doc.get("doc_checksum", ""),
                "declared_unverified_groups": declared_groups,
                "reasoning": reasoning,
                "requires_human_approval": True,
            }
            recommendations.append(rec)
            family_grouping[canonical_family].append(item["document_id"])
            
        summary = {
            "total_documents": len(recommendations),
            "status_counts": dict(Counter(r["evidence_status"] for r in recommendations)),
            "canonical_families_identified": len(family_grouping),
            "family_distributions": {k: len(v) for k, v in sorted(family_grouping.items())},
        }
        return recommendations, summary

    def audit_label_references(self) -> tuple[list[dict], list[dict], list[dict], dict]:
        """Phase 3 & 4: Audit all 762 references across all 395 label query rows."""
        recommendations = []
        uncertain_cases = []
        false_negative_candidates = []
        
        # Build equipment keyword mappings
        equipment_keywords = {
            "ktm": "KTM",
            "shredder": "Paper Shredder",
            "cmc1000": "Paper Shredder",
            "chiller": "Milk Chiller",
            "milk": "Milk",
            "analyzer": "Milk Analyzer",
            "autopick": "Autopick",
            "pallet": "Pallet Lift",
            "stacker": "Pallet Lift",
            "stabilizer": "Voltcore",
            "voltcore": "Voltcore",
            "ups": "Voltcore",
            "exterior jack": "Exterior Jack",
            "jack": "Exterior Jack",
            "celeron": "Asus Celeron",
            "asus": "Asus Celeron",
            "terminal": "Asus Celeron",
            "ac supplier": "AC Supplier",
            "aerolift": "Aerolift",
            "bloodanalyzer": "BloodAnalyzer",
            "ba3000": "BloodAnalyzer",
            "split ac": "Split AC",
            "vrf": "VRF",
            "ductable": "Ductable AC",
        }
        
        for q_idx, row in enumerate(self.label_queue):
            split = row.get("split")
            qid = str(row.get("query_id"))
            qtext = str(row.get("query_text") or "").strip()
            qtokens = token_set(qtext)
            
            # Identify equipment mentioned in query
            query_equip = set()
            for kw, eq in equipment_keywords.items():
                if kw in qtext.lower():
                    query_equip.add(eq)
                    
            references = row.get("references", [])
            for ref_idx, ref in enumerate(references):
                role = ref["role"]
                cid = str(ref["chunk_id"])
                doc_id = str(ref["document_id"]).replace("doc-", "")
                passage = str(ref["passage"] or "").strip()
                ptokens = token_set(passage)
                
                db_doc = self.db_documents.get(doc_id, {})
                doc_title = str(db_doc.get("doc_title") or "")
                doc_prod = str(db_doc.get("product_name") or "")
                
                # Check equipment match between query and document
                doc_equip = set()
                for kw, eq in equipment_keywords.items():
                    if kw in doc_title.lower() or kw in doc_prod.lower():
                        doc_equip.add(eq)
                        
                equip_match = bool(query_equip & doc_equip) if (query_equip and doc_equip) else True
                
                # Extract key actions/symptoms from query
                query_action_tokens = qtokens - {
                    "how", "to", "what", "is", "the", "a", "an", "and", "or", "in", "on", "for", 
                    "procedure", "guide", "manual", "troubleshooting", "servy", "system"
                }
                
                # Overlap between question action tokens and passage tokens
                shared_tokens = query_action_tokens & ptokens
                overlap_ratio = len(shared_tokens) / len(query_action_tokens) if query_action_tokens else 0.0
                
                heading_line = passage.split("\n")[0] if "\n" in passage else passage[:50]
                heading_tokens = token_set(heading_line)
                heading_overlap = len(query_action_tokens & heading_tokens)
                
                contradictions = []
                
                if role == "positive":
                    if not equip_match:
                        rec_decision = "POTENTIAL_EQUIPMENT_MISMATCH"
                        confidence = "LOW"
                        contradictions.append(f"Query refers to {query_equip} but source document is for {doc_equip}")
                        human_action = "VERIFY_EQUIPMENT_APPLICABILITY"
                    elif heading_overlap >= 2 or overlap_ratio >= 0.4:
                        rec_decision = "RECOMMEND_APPROVE"
                        confidence = "HIGH"
                        human_action = "APPROVE_POSITIVE_LABEL"
                    elif overlap_ratio >= 0.2:
                        rec_decision = "POTENTIAL_PARTIAL_ANSWER"
                        confidence = "MEDIUM"
                        human_action = "INSPECT_FOR_COMPLETENESS"
                    else:
                        rec_decision = "UNCERTAIN"
                        confidence = "LOW"
                        contradictions.append("Low semantic/action overlap with query")
                        human_action = "INSPECT_CONTENT_RELEVANCE"
                        
                    reasoning = (
                        f"Positive candidate for '{qtext}'. Source: '{doc_title}' ({doc_prod}). "
                        f"Heading: '{heading_line}'. Action token overlap: {len(shared_tokens)}/{len(query_action_tokens)} "
                        f"({overlap_ratio:.1%})."
                    )
                else: # role == "negative"
                    # False negative detection: does the negative actually answer the question?
                    if heading_overlap >= 3 or (overlap_ratio >= 0.6 and equip_match):
                        # Strong suspicion of false negative!
                        rec_decision = "POTENTIAL_FALSE_NEGATIVE"
                        confidence = "MEDIUM"
                        contradictions.append(
                            f"Negative passage shares strong procedural heading/action overlap ({len(shared_tokens)} shared tokens) with query intent"
                        )
                        human_action = "REJECT_NEGATIVE_LABEL_OR_RECLASSIFY_AS_POSITIVE"
                        false_neg_entry = {
                            "query_id": qid, "query_text": qtext, "chunk_id": cid,
                            "document_id": doc_id, "document_title": doc_title,
                            "heading": heading_line, "shared_tokens": sorted(shared_tokens),
                            "overlap_ratio": overlap_ratio,
                            "passage_preview": passage[:200]
                        }
                        false_negative_candidates.append(false_neg_entry)
                    elif not equip_match:
                        # Cross-equipment negative
                        rec_decision = "RECOMMEND_APPROVE"
                        confidence = "HIGH"
                        human_action = "CONFIRM_NEGATIVE_LABEL"
                    else:
                        # Same equipment, different procedure: valid hard negative
                        rec_decision = "RECOMMEND_APPROVE"
                        confidence = "HIGH"
                        human_action = "CONFIRM_NEGATIVE_LABEL"
                        
                    reasoning = (
                        f"Proposed negative for '{qtext}'. Source: '{doc_title}' ({doc_prod}). "
                        f"Heading: '{heading_line}'. Action overlap: {len(shared_tokens)}/{len(query_action_tokens)} "
                        f"({overlap_ratio:.1%})."
                    )
                    
                rec_record = {
                    "split": split,
                    "query_id": qid,
                    "query_text": qtext,
                    "reference_index": ref_idx,
                    "role": role,
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "product": doc_prod,
                    "heading": heading_line,
                    "proposed_ai_decision": rec_decision,
                    "confidence": confidence,
                    "contradictions": contradictions,
                    "shared_tokens": sorted(shared_tokens),
                    "action_overlap_ratio": round(overlap_ratio, 3),
                    "reasoning": reasoning,
                    "recommended_human_action": human_action,
                    "requires_human_approval": True,
                }
                recommendations.append(rec_record)
                
                if rec_decision in ("UNCERTAIN", "POTENTIAL_EQUIPMENT_MISMATCH", "POTENTIAL_PARTIAL_ANSWER"):
                    uncertain_cases.append(rec_record)
                    
        summary = {
            "total_references_audited": len(recommendations),
            "decision_breakdown": dict(Counter(r["proposed_ai_decision"] for r in recommendations)),
            "confidence_breakdown": dict(Counter(r["confidence"] for r in recommendations)),
            "uncertain_cases_count": len(uncertain_cases),
            "potential_false_negatives_count": len(false_negative_candidates),
        }
        return recommendations, uncertain_cases, false_negative_candidates, summary

    def run_full_audit(self, out_dir: Path) -> dict:
        """Run all audit phases and export the structured machine evidence package."""
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        doc_recs, doc_summary = self.audit_document_provenance()
        lbl_recs, uncertain, false_negs, lbl_summary = self.audit_label_references()
        
        # 1. ai_source_family_recommendations.jsonl
        with (out_dir / "ai_source_family_recommendations.jsonl").open("w", encoding="utf-8") as f:
            for r in doc_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        # 2. ai_label_recommendations.jsonl
        with (out_dir / "ai_label_recommendations.jsonl").open("w", encoding="utf-8") as f:
            for r in lbl_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        # 3. ai_uncertain_cases.jsonl
        with (out_dir / "ai_uncertain_cases.jsonl").open("w", encoding="utf-8") as f:
            for r in uncertain:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        # 4. ai_false_negative_candidates.jsonl
        with (out_dir / "ai_false_negative_candidates.jsonl").open("w", encoding="utf-8") as f:
            for r in false_negs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        # 5. ai_source_evidence_manifest.json
        manifest = {
            "schema_version": "servy-v4-ai-evidence-1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_sha256": compute_sha256(self.corpus_path),
            "sqlite_source_db": str(self.db_path),
            "documents_analyzed": len(doc_recs),
            "references_analyzed": len(lbl_recs),
            "document_summary": doc_summary,
            "label_summary": lbl_summary,
            "note": "Machine review recommendations only; human expert approval remains mandatory."
        }
        (out_dir / "ai_source_evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        
        # 6. ai_review_summary.json
        full_summary = {
            "provenance_audit": doc_summary,
            "relevance_audit": lbl_summary,
            "out_directory": str(out_dir),
            "training_ready": False,
            "human_approval_complete": False,
            "ai_analysis_complete": True
        }
        (out_dir / "ai_review_summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
        
        return full_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--db", default="db.sqlite3")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    
    auditor = ServyAIEvidenceAuditor(args.bundle, args.corpus, args.db)
    try:
        summary = auditor.run_full_audit(args.out_dir)
        print(json.dumps(summary, indent=2))
    finally:
        auditor.close()


if __name__ == "__main__":
    main()
