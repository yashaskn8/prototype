# Real-data demo Chroma integrity recovery

Company data, the original Chroma index, and frozen manifests must not be rewritten.
The current source differs from the 2026-10-31 frozen hashes in exactly three
files, but identical current logical collections do not prove frozen equivalence.

Run the read-only diagnostic in the authorized Windows demo environment:

    .\tools\Start-ServyRealDemo.ps1 -ReviewRequest

The deterministic JSON result binds the original frozen before.json and
production_index_manifest.json to current complete production/private index
manifests and exact changed filenames. It refuses missing/added files or
changes outside the three known SQLite/HNSW storage filenames.

Recovery options:

1. Prefer a separately verified full 122-file backup exactly matching the
   original SHA-256 manifest; preserve existing files and request owner approval
   before restoring.
2. Without a trusted original copy, require an independent technical reviewer
   to verify the actual document scope, collection records, provenance,
   embeddings, and authorized asset isolation against a trusted reference.
   Current index versus current private copy alone is not sufficient evidence.

An independent reviewer signs a JSON review object containing the exact request
from the diagnostic, nonempty reviewer ID, detailed reason, UTC approval time,
and this literal attestation:

    I independently verified source authorization, collection records, embeddings, provenance, and the scoped index against an accepted reference.

The reviewer signs these canonical UTF-8 bytes with an independently held
Ed25519 private key:

    json.dumps(review, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")

The approved artifact is JSON with keys 'review' and 'signature' (base64).
Store it as approved_chroma_drift_v2.json in the already protected private
demo evaluation folder, outside Git. Supply the independently controlled
Ed25519 public PEM path via SERVY_REAL_DEMO_REVIEW_PUBLIC_KEY.

Without a valid signature, trustworthy review or an original verified backup,
the startup gate remains blocked. Never self-sign to bypass the gate.

After legitimate recovery, run:

    .\tools\Start-ServyRealDemo.ps1 -Restart

Once verified, the demo runs Chroma from a disposable authenticated clone.
Normal reads cannot alter the original company index or the reviewed snapshot.
The benchmark's original immutable index check is unchanged.

## Alternative: a fresh, isolated source-verified index

The frozen Chroma image is still byte-drifted and MUST NOT be silently
approved as equivalent to the original. When original company document
hashes, DB record-level evidence, BGE model, and V5 weights match the
separately frozen sources, the demonstration can instead use a **new**
temporary index generated from the authorized source passages.

This is a separate, explicitly selected serving mode and **not** a
restoration of the old frozen index. It keeps the original Chroma index,
old private copy, frozen manifests, and company MySQL records unchanged.

Run from the repository root:

    .\tools\Start-ServyRealDemo.ps1 -Restart -RebuildFromVerifiedDocs

In this mode, the backend:
- verifies the unchanged frozen MySQL/document records and approved assets;
- reauthorizes each customer-visible chunk and excludes injection content;
- uses the local BGE-small checkpoint offline to recompute dense embeddings;
- creates a new Chroma collection in a disposable local private directory;
- rereads IDs, provenance metadata, passage text, and vectors to verify the result;
- serves the usual asset-isolated hybrid RAG, V5 reranking, and signed citations;
- keeps the operational database SELECT-only and creates no real service calls.

If any check fails, it must stop with an error. The original frozen Chroma
byte-integrity gate is not weakened; the changed index is not used.

This path is intended to restore a functional **source-grounded demo**,
not to assert byte equivalence or independently certified recovery of the
previous index. A future strict evaluation against the original frozen Chroma
snapshot still requires trusted original bytes or independent review.
