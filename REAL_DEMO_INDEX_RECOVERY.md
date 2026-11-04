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
