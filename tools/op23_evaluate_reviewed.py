"""Evaluate signed frozen cases; require a locked configuration for final test."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.ml.op23_evaluation import evaluate_reviewed
from core.ml.op23_review_authority import ApprovalError, verify_configuration_lock
from core.ml.op23_review_dataset import DatasetValidationError, validate_case
from core.ml.op23_source_admission import admit_capture, canonical_bytes, sha256

parser = argparse.ArgumentParser()
for name in ("dataset", "admission", "capture", "output", "configuration"):
    parser.add_argument(f"--{name}", type=Path, required=True)
parser.add_argument("--predictions", type=Path)
parser.add_argument("--generate-minilm", action="store_true")
parser.add_argument("--predictions-output", type=Path)
parser.add_argument("--config-lock", type=Path)
args = parser.parse_args()
if args.generate_minilm == bool(args.predictions) or (args.predictions_output and not args.generate_minilm):
    raise DatasetValidationError("Choose exactly one prediction source; output path requires MiniLM generation")
private = (ROOT / "private_training_data").resolve()
for path in (args.dataset, args.admission, args.capture, args.predictions, args.output,
             args.configuration, args.config_lock, args.predictions_output):
    if path and not path.resolve().is_relative_to(private):
        raise DatasetValidationError("Evaluation files must remain private")
manifest = json.loads(args.dataset.with_suffix(args.dataset.suffix + ".manifest.json").read_text(encoding="utf-8"))
if sha256(args.dataset.read_bytes()) != manifest["dataset_sha256"]:
    raise DatasetValidationError("Frozen dataset content changed")
baseline = json.loads((private / "op23_review" / "sealed_prior_sources_v2.json").read_text(encoding="utf-8"))
admission = json.loads(args.admission.read_text(encoding="utf-8"))
recomputed = admit_capture(args.capture, baseline, expected_org=admission["source_organisation"], tenant_id=admission["tenant_id"])
if canonical_bytes(recomputed) != canonical_bytes(admission) or sha256(canonical_bytes(admission)) != manifest["admission_sha256"]:
    raise DatasetValidationError("Source admission changed after freeze")
if sha256(canonical_bytes(baseline)) != manifest["baseline_sha256"]:
    raise DatasetValidationError("Historical source baseline changed after freeze")
trust_json = os.environ.get("SERVY_REVIEW_TRUST_ROOT_JSON")
if not trust_json:
    raise DatasetValidationError("Externally provisioned reviewer trust root is required")
trusted = json.loads(trust_json)
if manifest["dataset_role"] == "INDEPENDENT_FINAL_TEST":
    if not args.config_lock:
        raise DatasetValidationError("Final test remains sealed until retrieval configuration is frozen")
    lock = json.loads(args.config_lock.read_text(encoding="utf-8"))
    try:
        verify_configuration_lock(lock, trusted, expected_org=admission["source_organisation"],
                                  tenant_id=admission["tenant_id"],
                                  configuration_sha256=sha256(args.configuration.read_bytes()))
    except ApprovalError as exc:
        raise DatasetValidationError(str(exc)) from exc
complaints = json.loads((args.capture / "complaints.json").read_text(encoding="utf-8"))
cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
for case in cases:
    validate_case(case, admission=admission, baseline=baseline, complaints=complaints,
                  trusted_keys=trusted, dataset_role=manifest["dataset_role"])
if args.generate_minilm:
    configuration = json.loads(args.configuration.read_text(encoding="utf-8"))
    if configuration.get("retriever") != "op23-private-minilm-passage-v1":
        raise DatasetValidationError("Configuration does not select the private MiniLM baseline")
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
    django.setup()
    from core.ml.embeddings import get_profile, load_model
    from core.ml.op23_private_retrieval import predict_minilm_passages
    predictions = predict_minilm_passages(cases, admission, load_model(get_profile("minilm")))
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("x", encoding="utf-8") as output:
            for prediction in predictions:
                output.write(json.dumps(prediction, sort_keys=True) + "\n")
else:
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
report = {"dataset_sha256": manifest["dataset_sha256"], "configuration_sha256": sha256(args.configuration.read_bytes()),
          "metrics_by_role": evaluate_reviewed(cases, predictions)}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as target:
    json.dump(report, target, indent=2)
print(json.dumps(report, indent=2))
