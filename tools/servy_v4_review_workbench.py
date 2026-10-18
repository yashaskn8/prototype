"""Offline Servy V4 human-review workbench; never self-approves data.

Launch from the project root (Windows):
  python tools/servy_v4_review_workbench.py --bundle private_training_data/recovery_v4/review_bundle_... --corpus private_training_data/recovery_v4/servy_corpus_recovered_....jsonl --out-dir private_training_data/recovery_v4/adjudication_...

The application binds no sockets, runs no model, and never touches the DB.
It copies the original proposal and queues into a NEW private bundle, resumes
explicit reviewer decisions there, and never modifies the original exports.
A human must compare each source family with the real source documents and
assess each positive/negative reference. Passing this GUI is not authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

NAMES = (
    "train_unapproved_candidate.jsonl",
    "dev_unapproved_candidate.jsonl",
    "source_family_review_queue.jsonl",
    "expert_review_queue.jsonl",
)
MANIFEST = "review_workbench_state.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Non-object JSONL row at {path}:{i}")
                rows.append(row)
    if not rows:
        raise ValueError(f"No JSONL rows: {path}")
    return rows


def safe_write(path, rows):
    """Complete JSONL replacement via a private temporary file and atomic move."""
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    if temporary.exists():
        raise FileExistsError(f"Incomplete prior write; preserve and inspect {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        # Preserve a failed .writing file for recovery instead of deleting it.
        pass


def validate_destination(path, source):
    path, source = Path(path).resolve(), Path(source).resolve()
    if "private_training_data" not in {piece.casefold() for piece in path.parts}:
        raise ValueError("Output MUST reside under private_training_data, not a public Git folder")
    if path == source or path in source.parents or source in path.parents:
        raise ValueError("Review output must be a NEW sibling of the original bundle")
    return path


class ReviewSession:
    def __init__(self, output, corpus, ai_evidence_dir=None):
        self.output = Path(output).resolve()
        self.corpus = Path(corpus).resolve()
        self.state = json.loads((self.output / MANIFEST).read_text(encoding="utf-8"))
        if digest(self.corpus) != self.state["corpus_sha256"]:
            raise ValueError("Original corpus changed since this review session was created")
        self.sources = read_jsonl(self.output / NAMES[2])
        self.labels = read_jsonl(self.output / NAMES[3])
        for name in NAMES[:2]:
            if digest(self.output / name) != self.state["candidate_sha256"][name]:
                raise ValueError("Candidate dataset changed; no review decisions were written")
        self.corpus_rows = read_jsonl(self.corpus)
        self.by_document = defaultdict(list)
        self.by_chunk = {}
        for record in self.corpus_rows:
            doc = (str(record.get("tenant_id")), str(record.get("document_id")))
            cid = str(record.get("chunk_id"))
            if cid in self.by_chunk:
                raise ValueError("Duplicate corpus chunk IDs")
            self.by_document[doc].append(record)
            self.by_chunk[cid] = record
        source_keys = [(str(x.get("tenant_id")), str(x.get("document_id"))) for x in self.sources]
        if len(source_keys) != len(set(source_keys)) or set(source_keys) != set(self.by_document):
            raise ValueError("Source review queue does not match original corpus documents")
        keys = [(x.get("split"), str(x.get("query_id"))) for x in self.labels]
        if len(keys) != len(set(keys)):
            raise ValueError("Ambiguous duplicate label review keys")
        for row in self.labels:
            for ref in row.get("references", []):
                chunk = self.by_chunk.get(str(ref.get("chunk_id")))
                if (not chunk or str(chunk.get("passage")) != str(ref.get("passage"))
                        or str(chunk.get("document_id")) != str(ref.get("document_id"))):
                    raise ValueError("Reviewed passage is not identical to recovered corpus")

        # Load AI Evidence Recommendations if available
        self.ai_sources = {}
        self.ai_labels = {}
        evidence_path = Path(ai_evidence_dir) if ai_evidence_dir else None
        if not evidence_path or not evidence_path.is_dir():
            candidate_dirs = sorted(self.output.parent.glob("ai_review_evidence_*"))
            if candidate_dirs:
                evidence_path = candidate_dirs[-1]
        if evidence_path and evidence_path.is_dir():
            s_file = evidence_path / "ai_source_family_recommendations.jsonl"
            l_file = evidence_path / "ai_label_recommendations.jsonl"
            if s_file.is_file():
                for line in s_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        self.ai_sources[str(r["document_id"])] = r
            if l_file.is_file():
                for line in l_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        self.ai_labels[(str(r["query_id"]), r["reference_index"])] = r

    @classmethod
    def create(cls, bundle, corpus, output, ai_evidence_dir=None):
        bundle, corpus = Path(bundle).resolve(), Path(corpus).resolve()
        output = validate_destination(output, bundle)
        if not corpus.is_file():
            raise FileNotFoundError(corpus)
        for name in NAMES:
            if not (bundle / name).is_file():
                raise FileNotFoundError(bundle / name)
        if output.exists():
            raise FileExistsError("Refusing to overwrite an existing review directory")
        # Do not start until input files and JSONL formats are checked.
        for name in NAMES:
            read_jsonl(bundle / name)
        state = {
            "schema": "servy-v4-offline-review-1",
            "original_bundle": str(bundle),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_sha256": digest(corpus),
            "candidate_sha256": {n: digest(bundle / n) for n in NAMES[:2]},
            "source_queue_original_sha256": digest(bundle / NAMES[2]),
            "label_queue_original_sha256": digest(bundle / NAMES[3]),
            "note": "Only explicit domain reviewer decisions may alter the copied review queues",
        }
        output.mkdir(parents=True, exist_ok=False)
        for name in NAMES:
            shutil.copyfile(bundle / name, output / name)
        report = bundle / "readiness_report.json"
        if report.exists():
            shutil.copyfile(report, output / report.name)
        (output / MANIFEST).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return cls(output, corpus, ai_evidence_dir=ai_evidence_dir)

    def review_source(self, index, *, decision, reviewer, evidence, family="", version=""):
        if decision not in ("approved", "rejected"):
            raise ValueError("Decision must be explicitly approved or rejected")
        reviewer, evidence = reviewer.strip(), evidence.strip()
        if not reviewer or not evidence:
            raise ValueError("Authorized reviewer identity and actual supporting evidence required")
        item = self.sources[index]
        if decision == "approved":
            family, version = family.strip(), version.strip()
            if not family or not version:
                raise ValueError("Verified canonical family and original source version required")
            original_versions = item.get("document_versions") or []
            if set(map(str, original_versions)) != {version}:
                raise ValueError("Approved version differs from the original source version")
            checksums = item.get("source_document_sha256") or []
            if len(checksums) != 1 or not re.fullmatch(r"[0-9a-f]{64}", str(checksums[0]).lower()):
                raise ValueError("Cannot approve source without a unique verified source checksum")
            item["verified_canonical_source_family"] = family
            item["approved_source_version"] = version
        else:
            item["verified_canonical_source_family"] = None
            item["approved_source_version"] = None
        item["decision"] = decision
        item["reviewer_id"] = reviewer
        item["evidence_reference"] = evidence
        safe_write(self.output / NAMES[2], self.sources)

    def review_reference(self, index, ref_index, *, decision, reviewer, evidence):
        if decision not in ("approved", "rejected"):
            raise ValueError("Decision must be explicitly approved or rejected")
        reviewer, evidence = reviewer.strip(), evidence.strip()
        if not reviewer or not evidence:
            raise ValueError("Authorized reviewer identity and actual supporting evidence required")
        row = self.labels[index]
        ref = row["references"][ref_index]
        # Each candidate is a separate affirmative human decision; never bulk approve.
        ref["decision"] = decision
        ref["reviewer_id"] = reviewer
        ref["review_evidence_reference"] = evidence
        for role, key in (("positive", "positive_decision"), ("negative", "negative_decision")):
            decisions = [r.get("decision", "PENDING") for r in row["references"] if r["role"] == role]
            row[key] = ("NOT_APPLICABLE" if not decisions else
                        "rejected" if "rejected" in decisions else
                        "approved" if all(x == "approved" for x in decisions) else "PENDING")
        if all(x.get("decision") == "approved" for x in row["references"]):
            row["reviewer_id"] = reviewer
            row["review_evidence_reference"] = evidence
        else:
            row["reviewer_id"] = None
            row["review_evidence_reference"] = None
        safe_write(self.output / NAMES[3], self.labels)

    def summary(self):
        refs = [r for row in self.labels for r in row.get("references", [])]
        source_counts = Counter(x.get("decision") for x in self.sources)
        label_counts = Counter(x.get("decision") for x in refs)
        rows_completed = sum(all(r.get("decision") == "approved" for r in row.get("references", []))
                             for row in self.labels)
        return {
            "source_documents": len(self.sources),
            "sources_approved": source_counts["approved"],
            "sources_rejected": source_counts["rejected"],
            "sources_pending": len(self.sources) - source_counts["approved"] - source_counts["rejected"],
            "label_rows": len(self.labels),
            "label_rows_fully_approved": rows_completed,
            "label_references": len(refs),
            "references_approved": label_counts["approved"],
            "references_rejected": label_counts["rejected"],
            "references_pending": len(refs) - label_counts["approved"] - label_counts["rejected"],
            "training_ready": False,
            "finalizer_and_external_authorization_still_required": True,
        }


def gui(session):
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    root = tk.Tk()
    root.title("Servy V4 — Offline Authorized Expert Review")
    root.geometry("1090x780")
    header = ttk.Frame(root, padding=10)
    header.pack(fill="x")
    ttk.Label(header, text="LOCAL ONLY • Review real sources. Never approve without evidence.",
              font=("Segoe UI", 12, "bold")).pack(anchor="w")
    counts = tk.StringVar()
    ttk.Label(header, textvariable=counts).pack(anchor="w")
    reviewer = tk.StringVar()
    ttk.Label(header, text="Reviewer ID (must match external authorization)").pack(anchor="w")
    ttk.Entry(header, textvariable=reviewer, width=50).pack(anchor="w")
    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True, padx=10, pady=10)

    # Both queues retain original source bytes and source IDs; only reviewer
    # attestation fields are editable through the buttons below.
    sources_tab = ttk.Frame(tabs)
    tabs.add(sources_tab, text="1. Source families")
    source_nav = ttk.Frame(sources_tab)
    source_nav.pack(fill="x")
    # Pin action controls above the expanding document viewer, so Windows
    # display scaling cannot push Approve/Reject beyond the bottom edge.
    source_actions = ttk.Frame(sources_tab)
    source_actions.pack(fill="x", pady=4)
    si = [0]
    source_id = tk.StringVar()
    ttk.Label(source_nav, textvariable=source_id).pack(side="left", padx=4)
    source_text = scrolledtext.ScrolledText(sources_tab, height=8, wrap="word")
    family = tk.StringVar()
    family_version = tk.StringVar()
    family_evidence = tk.StringVar()
    for title, variable in (("Canonical source family", family),
                            ("Approved original version", family_version),
                            ("Evidence reference to verified original document", family_evidence)):
        ttk.Label(sources_tab, text=title).pack(anchor="w")
        ttk.Entry(sources_tab, textvariable=variable, width=105).pack(fill="x")
    # Keep provenance input fields above the expanding passage viewer.
    source_text.pack(fill="both", expand=True)

    labels_tab = ttk.Frame(tabs)
    tabs.add(labels_tab, text="2. Positive / negative labels")
    label_nav = ttk.Frame(labels_tab)
    label_nav.pack(fill="x")
    label_actions = ttk.Frame(labels_tab)
    label_actions.pack(fill="x", pady=4)
    li = [0]
    label_id = tk.StringVar()
    ttk.Label(label_nav, textvariable=label_id).pack(side="left", padx=4)
    reference_list = tk.Listbox(labels_tab, height=5, exportselection=False)
    reference_list.pack(fill="x")
    label_text = scrolledtext.ScrolledText(labels_tab, wrap="word", height=8)
    label_evidence = tk.StringVar()
    ttk.Label(labels_tab, text="Evidence reference / technical justification for THIS passage").pack(anchor="w")
    ttk.Entry(labels_tab, textvariable=label_evidence, width=105).pack(fill="x")
    label_text.pack(fill="both", expand=True)

    def refresh_stats():
        stats = session.summary()
        counts.set(f"Sources approved {stats['sources_approved']}/{stats['source_documents']} | "
                   f"Label passages approved {stats['references_approved']}/{stats['label_references']} | "
                   f"Pending {stats['references_pending']} | Rejected {stats['references_rejected']}")

    def set_readonly(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def render_source():
        item = session.sources[si[0]]
        identity = (str(item['tenant_id']), str(item['document_id']))
        passages = session.by_document[identity]
        source_id.set(f"Document {si[0]+1}/{len(session.sources)} | tenant {identity[0]} | document {identity[1]} | "
                      f"status {item.get('decision')}")
        
        # Pre-fill from AI recommendation if empty and not yet approved
        ai_s = session.ai_sources.get(str(item['document_id']))
        if not item.get("verified_canonical_source_family") and ai_s:
            family.set(ai_s.get("proposed_canonical_source_family") or "")
        else:
            family.set(item.get("verified_canonical_source_family") or "")
            
        versions = item.get("document_versions") or []
        # Original version can contain an index-version suffix required by
        # the finalizer. The registry's plain content version is not enough.
        original_ver = str(versions[0]) if len(versions) == 1 else ""
        family_version.set(item.get("approved_source_version") or original_ver)
        # AI text is NOT reviewed evidence. Human reviewer must supply it.
        family_evidence.set(item.get("evidence_reference") or "")
        
        ai_header = ""
        if ai_s:
            ai_header = (
                "=== AI SOURCE EVIDENCE AUDIT ===\n"
                f"Status: {ai_s.get('evidence_status')} | Confidence: {ai_s.get('confidence')}\n"
                f"Authoritative Title: {ai_s.get('authoritative_title')}\n"
                f"Product: {ai_s.get('authoritative_product')} | Category: {ai_s.get('authoritative_category')}\n"
                f"Canonical Family: {ai_s.get('proposed_canonical_source_family')}\n"
                f"Checksum: {ai_s.get('authoritative_checksum')}\n"
                f"Reasoning: {ai_s.get('reasoning')}\n\n"
            )
            
        set_readonly(source_text,
                     ai_header +
                     "Document metadata:\n" + json.dumps({k:v for k,v in item.items()
                     if k not in ('evidence_reference',)}, indent=2) +
                     "\n\nRECOVERED PASSAGES — VERIFY AGAINST ORIGINAL SOURCE DOCUMENT:\n" +
                     "\n\n".join(f"Chunk {x['chunk_id']}\n{x['passage']}" for x in passages))
        refresh_stats()

    def show_reference(_event=None):
        row = session.labels[li[0]]
        selection = reference_list.curselection()
        if not selection:
            return
        ref = row["references"][selection[0]]
        ai_l = session.ai_labels.get((str(row.get("query_id")), selection[0]))
        
        # Never use machine-generated recommendations as a reviewer's own evidence.
        label_evidence.set(ref.get("review_evidence_reference") or "")
        
        ai_ref_section = ""
        if ai_l:
            contra = "; ".join(ai_l.get("contradictions", [])) or "None detected"
            ai_ref_section = (
                "=== AI RELEVANCE AUDIT & EVIDENCE ===\n"
                f"AI Recommendation: {ai_l.get('proposed_ai_decision')} | Confidence: {ai_l.get('confidence')}\n"
                f"Action Token Overlap: {ai_l.get('action_overlap_ratio', 0):.1%} | Shared: {', '.join(ai_l.get('shared_tokens', []))}\n"
                f"Source Section: {ai_l.get('heading')}\n"
                f"Technical Reasoning: {ai_l.get('reasoning')}\n"
                f"Contradictions: {contra}\n"
                f"Recommended Action: {ai_l.get('recommended_human_action')}\n\n"
            )
            
        set_readonly(label_text, 
                     ai_ref_section +
                     f"Question: {row.get('query_text')}\n\n"
                     f"Role: {ref.get('role')} | chunk: {ref.get('chunk_id')} | "
                     f"document: {ref.get('document_id')}\n\n"
                     f"DECISION: {ref.get('decision')}\n\n"
                     f"SOURCE PASSAGE:\n{ref.get('passage')}\n\n"
                     "Verify this exact technical relevance against the original document. "
                     "Negative means demonstrably nonrelevant to the question, "
                     "not merely a lower-ranked answer.")

    def render_label():
        row = session.labels[li[0]]
        label_id.set(f"Question {li[0]+1}/{len(session.labels)} | "
                     f"{row.get('split')} / {row.get('query_id')} | {row.get('query_text')}")
        reference_list.delete(0, "end")
        for idx, ref in enumerate(row["references"]):
            ai_l = session.ai_labels.get((str(row.get("query_id")), idx))
            tag = ""
            if ai_l:
                rec_dec = ai_l.get("proposed_ai_decision", "")
                if "FALSE_NEGATIVE" in rec_dec:
                    tag = " [! FALSE NEGATIVE ? !]"
                elif "UNCERTAIN" in rec_dec:
                    tag = " [? UNCERTAIN ?]"
                elif "PARTIAL" in rec_dec:
                    tag = " [PARTIAL]"
                else:
                    tag = " [OK]"
            reference_list.insert("end", f"{ref.get('role'):8s} | chunk {ref.get('chunk_id'):<4} | "
                                  f"{ref.get('decision'):<8s}{tag}")
        if row["references"]:
            reference_list.selection_set(0)
            show_reference()
        refresh_stats()

    def source_move(delta=1, pending=False):
        if pending:
            found = next((i for i,x in enumerate(session.sources)
                          if x.get('decision') not in ('approved','rejected') and i > si[0]), None)
            if found is None:
                found = next((i for i,x in enumerate(session.sources)
                              if x.get('decision') not in ('approved','rejected')), None)
            if found is not None:
                si[0] = found
        else:
            si[0] = (si[0]+delta) % len(session.sources)
        render_source()

    def label_move(delta=1, pending=False, high_risk=False):
        if pending:
            def needs_review(row):
                return any(r.get('decision') not in ('approved','rejected') for r in row['references'])
            found = next((i for i,x in enumerate(session.labels) if needs_review(x) and i > li[0]), None)
            if found is None:
                found = next((i for i,x in enumerate(session.labels) if needs_review(x)), None)
            if found is not None:
                li[0] = found
        elif high_risk:
            def is_high_risk(row):
                qid = str(row.get("query_id"))
                return any(
                    "FALSE_NEGATIVE" in session.ai_labels.get((qid, idx), {}).get("proposed_ai_decision", "")
                    or "UNCERTAIN" in session.ai_labels.get((qid, idx), {}).get("proposed_ai_decision", "")
                    for idx, _ in enumerate(row.get("references", []))
                )
            found = next((i for i,x in enumerate(session.labels) if is_high_risk(x) and i > li[0]), None)
            if found is None:
                found = next((i for i,x in enumerate(session.labels) if is_high_risk(x)), None)
            if found is not None:
                li[0] = found
        else:
            li[0] = (li[0]+delta) % len(session.labels)
        render_label()

    def source_decision(decision):
        if not messagebox.askyesno("Confirm actual expert decision",
                    f"Record {decision.upper()} for source document {session.sources[si[0]]['document_id']}?\n"
                    "You must have verified the original source document and evidence."):
            return
        try:
            session.review_source(si[0], decision=decision, reviewer=reviewer.get(),
                                  evidence=family_evidence.get(), family=family.get(),
                                  version=family_version.get())
            source_move(pending=True)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Blocked review save", str(exc))

    def label_decision(decision):
        selection = reference_list.curselection()
        if not selection:
            messagebox.showerror("No selected passage", "Select one positive or negative passage")
            return
        if not messagebox.askyesno("Confirm actual expert decision",
                    f"Record {decision.upper()} for selected {session.labels[li[0]]['references'][selection[0]]['role']} passage?\n"
                    "You must have examined the evidence. This is not an automated assessment."):
            return
        try:
            session.review_reference(li[0], selection[0], decision=decision,
                                     reviewer=reviewer.get(), evidence=label_evidence.get())
            render_label()
            nxt = selection[0]+1
            if nxt < len(session.labels[li[0]]["references"]):
                reference_list.selection_clear(0, "end")
                reference_list.selection_set(nxt)
                show_reference()
            else:
                label_move(pending=True)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Blocked review save", str(exc))

    for caption, callback in (("Previous",lambda:source_move(-1)),("Next",lambda:source_move(1)),
                              ("Next pending",lambda:source_move(pending=True))):
        ttk.Button(source_nav,text=caption,command=callback).pack(side="right",padx=4)
    ttk.Button(source_actions, text="Record APPROVED source", command=lambda:source_decision('approved')).pack(side="left",padx=5)
    ttk.Button(source_actions, text="Record REJECTED source", command=lambda:source_decision('rejected')).pack(side="left",padx=5)
    for caption, callback in (("Previous",lambda:label_move(-1)),("Next",lambda:label_move(1)),
                              ("Next high-risk",lambda:label_move(high_risk=True)),
                              ("Next pending",lambda:label_move(pending=True))):
        ttk.Button(label_nav,text=caption,command=callback).pack(side="right",padx=4)
    reference_list.bind("<<ListboxSelect>>", show_reference)
    ttk.Button(label_actions, text="Record APPROVED passage", command=lambda:label_decision('approved')).pack(side="left",padx=5)
    ttk.Button(label_actions, text="Record REJECTED passage", command=lambda:label_decision('rejected')).pack(side="left",padx=5)
    render_source()
    render_label()
    root.mainloop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, help="Original recovery review bundle")
    p.add_argument("--corpus", required=True, help="Original recovered corpus JSONL")
    p.add_argument("--out-dir", required=True, help="Explicit new private review output directory")
    p.add_argument("--ai-evidence", help="Path to AI evidence recommendations directory")
    p.add_argument("--status-only", action="store_true", help="Print counts; no GUI or new decisions")
    a = p.parse_args()
    output = validate_destination(a.out_dir, a.bundle)
    if output.exists():
        if not (output / MANIFEST).is_file():
            p.error("Existing directory is not an initialized V4 review session")
        session = ReviewSession(output, a.corpus, ai_evidence_dir=a.ai_evidence)
    else:
        if a.status_only:
            p.error("Review session does not exist")
        session = ReviewSession.create(a.bundle, a.corpus, output, ai_evidence_dir=a.ai_evidence)
    print(json.dumps({"review_directory": str(output), "progress": session.summary()}, indent=2))
    if not a.status_only:
        gui(session)
        print(json.dumps({"review_directory": str(output), "progress": session.summary()}, indent=2))


if __name__ == "__main__":
    main()