"""Fail-closed, independently signed review of local Chroma byte drift.

No change to the company's Chroma index, frozen manifests or database is made.
Only a reviewer's Ed25519 signature over the exact two index manifests can
authorize the *demo* to use the reviewed private copy. Other evaluations keep
their original strict frozen-manifest requirements.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

SCHEMA = "servy.real-demo.chroma-index-review.v1"
# Only these observed Chroma storage-level changes are in scope. Any new
# deletion, addition, or changed file requires another engineering audit.
REVIEWABLE_NAMES = frozenset({"chroma.sqlite3", "data_level0.bin", "length.bin"})
ATTESTATION = (
    "I independently verified source authorization, collection records, "
    "embeddings, provenance, and the scoped index against an accepted reference."
)
_RUNTIME_SNAPSHOT = None


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def changed_files(expected, observed):
    if not isinstance(expected, dict) or not expected or not isinstance(observed, dict):
        raise RuntimeError("Missing immutable Chroma index manifests.")
    if set(expected) != set(observed):
        raise RuntimeError("Chroma file inventory changed; explicit source recovery is required.")
    changed = sorted(key for key in expected if expected[key] != observed[key])
    if changed and any(Path(name).name not in REVIEWABLE_NAMES for name in changed):
        raise RuntimeError("Chroma drift includes files outside the reviewed storage-level set.")
    return changed


def review_request(private, *, expected, production, private_copy):
    """Return the exact, deterministic review request without changing any file."""
    private = Path(private)
    changed = changed_files(expected, production)
    if not changed:
        raise RuntimeError("No production index drift requires a new review.")
    if not isinstance(private_copy, dict) or set(private_copy) != set(expected):
        raise RuntimeError("Private Chroma snapshot has missing or additional files.")
    frozen = private / "before.json"
    manifest = private / "production_index_manifest.json"
    if not frozen.is_file() or not manifest.is_file():
        raise RuntimeError("The original frozen integrity evidence is missing.")
    from tools.servy_500_benchmark import load_json
    if load_json(frozen).get("index") != expected or load_json(manifest) != expected:
        raise RuntimeError("Frozen Chroma manifests disagree; manual investigation required.")
    return {
        "schema": SCHEMA,
        "frozen_before_sha256": sha256_file(frozen),
        "frozen_manifest_sha256": sha256_file(manifest),
        "original_index_manifest_sha256": hashlib.sha256(canonical_bytes(expected)).hexdigest(),
        "current_production_manifest_sha256": hashlib.sha256(canonical_bytes(production)).hexdigest(),
        "current_private_copy_manifest_sha256": hashlib.sha256(canonical_bytes(private_copy)).hexdigest(),
        "changed_paths": changed,
        "file_count": len(expected),
    }


def verify_signed_approval(private, request):
    """Verify an external reviewer's key and signature; never self-approve."""
    private = Path(private)
    approval_path = private / "approved_chroma_drift_v2.json"
    public_path = os.environ.get("SERVY_REAL_DEMO_REVIEW_PUBLIC_KEY", "").strip()
    if not approval_path.is_file() or not public_path:
        raise RuntimeError(
            "Chroma bytes changed. A reviewed, Ed25519-signed index approval is "
            "required; no original index or frozen baseline was modified."
        )
    try:
        key_file = Path(public_path).resolve(strict=True)
        if not key_file.is_file() or not key_file.is_absolute():
            raise ValueError("Invalid reviewer public key")
        key = serialization.load_pem_public_key(key_file.read_bytes())
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("Reviewer key must be Ed25519")
        approved = json.loads(approval_path.read_text(encoding="utf-8"))
        review = approved["review"]
        signature = base64.b64decode(approved["signature"], validate=True)
        if (not isinstance(review, dict) or review.get("request") != request
            or review.get("attestation") != ATTESTATION
            or not isinstance(review.get("reviewer"), str)
            or len(review["reviewer"].strip()) < 3
            or not isinstance(review.get("reason"), str)
            or len(review["reason"].strip()) < 20
            or not isinstance(review.get("approved_utc"), str)
            or not review["approved_utc"].endswith("Z")):
            raise ValueError("Invalid, incomplete or stale index approval")
        key.verify(signature, canonical_bytes(review))
    except (ValueError, KeyError, OSError, TypeError, InvalidSignature) as exc:
        raise RuntimeError("Chroma review signature, reviewer identity, or exact snapshot binding is invalid.") from exc
    return True


def activate_runtime_snapshot(source, expected_manifest, scratch):
    """Clone an authenticated source into disposable local space for Chroma.

    Chroma can rewrite index bytes on reads; only this short-lived working copy
    can be changed. Neither production nor the reviewed private copy is opened.
    """
    from django.conf import settings
    from core.services import chroma_store
    from tools.servy_500_benchmark import file_manifest
    global _RUNTIME_SNAPSHOT
    if chroma_store._chroma_client is not None:
        raise RuntimeError("Chroma was opened before immutable snapshot verification.")
    source, scratch = Path(source).resolve(strict=True), Path(scratch).resolve(strict=True)
    if not source.is_dir() or not scratch.is_dir() or source == scratch or scratch.is_relative_to(source):
        raise RuntimeError("Unsafe Chroma runtime snapshot paths.")
    if file_manifest(source) != expected_manifest:
        raise RuntimeError("Approved source index changed before runtime copy.")
    instance = tempfile.TemporaryDirectory(prefix="servy_chroma_runtime_", dir=str(scratch), ignore_cleanup_errors=True)
    target = Path(instance.name) / "index"
    try:
        shutil.copytree(source, target)
        if file_manifest(source) != expected_manifest or file_manifest(target) != expected_manifest:
            raise RuntimeError("Chroma index changed during runtime cloning; no queries run.")
    except BaseException:
        instance.cleanup()
        raise
    _RUNTIME_SNAPSHOT = instance  # keep it alive until process exit
    settings.CHROMA_PERSIST_DIR = target  # demo process only
    return target
