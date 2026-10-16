"""Freeze externally signed private cases after source and role reconciliation."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.ml.op23_review_dataset import DatasetValidationError, freeze_cases
from core.ml.op23_source_admission import admit_capture, canonical_bytes, sha256

parser = argparse.ArgumentParser()
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--admission", type=Path, required=True)
parser.add_argument("--reviewed", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--role", choices=("TRAIN", "DEVELOPMENT", "HISTORICAL_DIAGNOSTIC", "INDEPENDENT_FINAL_TEST"), required=True)
args = parser.parse_args()
private = (ROOT / "private_training_data").resolve()
for path in (args.capture, args.admission, args.reviewed, args.output):
    if not path.resolve().is_relative_to(private):
        raise DatasetValidationError("Evaluation inputs and output must remain private")
baseline = json.loads((private / "op23_review" / "sealed_prior_sources_v2.json").read_text(encoding="utf-8"))
op22 = private / "op22_review" / "prior_source_exclusion.json"
if sha256(op22.read_bytes()) != baseline["op22_exclusion_sha256"]:
    raise DatasetValidationError("Sealed historical exclusion manifest changed")
admission = json.loads(args.admission.read_text(encoding="utf-8"))
recomputed = admit_capture(args.capture, baseline, expected_org=admission["source_organisation"], tenant_id=admission["tenant_id"])
if canonical_bytes(recomputed) != canonical_bytes(admission):
    raise DatasetValidationError("Admission manifest differs from authenticated source files")
capture_manifest = json.loads((args.capture / "manifest.json").read_text(encoding="utf-8"))
if sha256((args.capture / "complaints.json").read_bytes()) != capture_manifest["complaints_sha256"]:
    raise DatasetValidationError("Complaint capture changed")
complaints = json.loads((args.capture / "complaints.json").read_text(encoding="utf-8"))
cases = [json.loads(line) for line in args.reviewed.read_text(encoding="utf-8").splitlines() if line.strip()]
trust_json = os.environ.get("SERVY_REVIEW_TRUST_ROOT_JSON")
if not trust_json:
    raise DatasetValidationError("No externally provisioned reviewer trust root; cases remain PENDING_REVIEW")
trusted_keys = json.loads(trust_json)
existing = []
for manifest_file in private.rglob("*.jsonl.manifest.json"):
    dataset_file = manifest_file.with_suffix("").with_suffix("")
    metadata = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not dataset_file.exists() or sha256(dataset_file.read_bytes()) != metadata.get("dataset_sha256"):
        raise DatasetValidationError("Prior frozen dataset or manifest has changed")
    existing.extend(json.loads(line) for line in dataset_file.read_text(encoding="utf-8").splitlines() if line.strip())
result = freeze_cases(cases, args.output, admission=admission, baseline=baseline, complaints=complaints,
                      trusted_keys=trusted_keys, dataset_role=args.role, existing_role_cases=existing)
print(json.dumps(result, indent=2))
