"""Write a versioned private admission audit from a read-only source capture."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.ml.op23_source_admission import admit_capture, sha256

parser = argparse.ArgumentParser()
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expected-org", default="273")
parser.add_argument("--tenant-id", type=int, default=9)
args = parser.parse_args()
private = (ROOT / "private_training_data").resolve()
for path in (args.capture, args.output):
    if not path.resolve().is_relative_to(private):
        raise ValueError("Capture and admission outputs must remain private")
baseline_path = private / "op23_review" / "sealed_prior_sources_v2.json"
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
op22 = private / "op22_review" / "prior_source_exclusion.json"
if sha256(op22.read_bytes()) != baseline["op22_exclusion_sha256"]:
    raise ValueError("Historical source-exclusion manifest changed")
result = admit_capture(args.capture, baseline, expected_org=args.expected_org, tenant_id=args.tenant_id)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as output:
    json.dump(result, output, indent=2)
statuses = {}
for item in result["sources"]:
    key = item.get("independence_status", item.get("status"))
    statuses[key] = statuses.get(key, 0) + 1
print(json.dumps({"admission_manifest": str(args.output), "source_count": len(result["sources"]),
                  "statuses": statuses}, indent=2))
