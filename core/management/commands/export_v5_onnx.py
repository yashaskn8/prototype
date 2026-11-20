"""Export the SHA-pinned V5 checkpoint to ONNX (fp32) for faster CPU scoring.

The export is accepted only if it reproduces the PyTorch scores: identical
ranking order on every probe set and max |logit difference| <= 1e-3. The
manifest records the source weights SHA-256 and the SHA-256 of every exported
file; serving refuses an ONNX directory whose files do not match it.

Measured on the development laptop (20 frozen real pools, 96 pairs):
ONNX Runtime median 1.23 s/pool vs PyTorch 1.76 s; ranking identical 20/20;
max logit difference 5.1e-5; committed memory about 1.2 GB lower.

  python manage.py export_v5_onnx --out <directory outside the repository>
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

PROBES = [
    ("The split AC unit is not cooling.", ["Dirty filter or coil reduces cooling capacity.",
                                           "Check the remote control batteries.",
                                           "Annual service includes cleaning the condenser coil."]),
    ("What causes a high-pressure alarm?", ["Blocked condenser airflow raises discharge pressure.",
                                           "The indoor fan speed can be set to low, medium or high.",
                                           "A communication fault stops the outdoor module."]),
]
MAX_ABS_DIFF = 1e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


class Command(BaseCommand):
    help = "Export the verified V5 reranker to ONNX fp32 with a parity-checked manifest."

    def add_arguments(self, parser):
        parser.add_argument("--out", required=True)

    def handle(self, *args, **options):
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer


        from core.ml.heavy_job import HeavyJobBusy, acquire, release
        try:
            acquire("export_v5_onnx")
        except HeavyJobBusy as exc:
            raise CommandError(str(exc)) from None
        try:
            self._export(options, np, onnx, ort, torch, AutoModelForSequenceClassification, AutoTokenizer)
        finally:
            release()

    def _export(self, options, np, onnx, ort, torch, AutoModelForSequenceClassification, AutoTokenizer):
        from core.ml.reranker import SERVY_V5_WEIGHTS_SHA256, _local_checkpoint, _sha256_file
        source = _local_checkpoint()
        if source is None or _sha256_file(source / "model.safetensors") != SERVY_V5_WEIGHTS_SHA256:
            raise CommandError("The configured V5 checkpoint does not match the pinned SHA-256.")
        out = Path(options["out"]).resolve()
        repo = Path(__file__).resolve().parents[3]
        if repo in out.parents or out == repo:
            raise CommandError("Write the export outside the repository.")

        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(source), local_files_only=True).eval()
        work = Path(tempfile.mkdtemp(prefix="servy-v5-onnx-"))
        try:
            sample = tokenizer([["q", "p"]], padding=True, truncation=True, max_length=256, return_tensors="pt")
            raw = work / "raw.onnx"
            with torch.inference_mode():
                torch.onnx.export(model, (sample["input_ids"], sample["attention_mask"]), str(raw), dynamo=False,
                                  input_names=["input_ids", "attention_mask"], output_names=["logits"],
                                  dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                                "logits": {0: "b"}},
                                  opset_version=17, do_constant_folding=True)
            staged = work / "export"
            staged.mkdir()
            onnx.save(onnx.load(str(raw), load_external_data=True), str(staged / "model.onnx"),
                      save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx.data")
            tokenizer.save_pretrained(str(staged))

            session = ort.InferenceSession(str(staged / "model.onnx"), providers=["CPUExecutionProvider"])
            worst = 0.0
            for query, passages in PROBES:
                enc = tokenizer([[query, p] for p in passages], padding=True, truncation=True, max_length=256,
                                return_tensors="pt")
                with torch.inference_mode():
                    expected = model(**enc).logits[:, 0].numpy()
                got = session.run(["logits"], {"input_ids": enc["input_ids"].numpy().astype(np.int64),
                                               "attention_mask": enc["attention_mask"].numpy().astype(np.int64)})[0][:, 0]
                worst = max(worst, float(np.max(np.abs(expected - got))))
                if list(np.argsort(-expected)) != list(np.argsort(-got)):
                    raise CommandError("ONNX ranking differs from PyTorch; export rejected.")
            if worst > MAX_ABS_DIFF:
                raise CommandError(f"ONNX logits differ by {worst:.2e} (> {MAX_ABS_DIFF}); export rejected.")
            del session

            manifest = {
                "format": "servy-v5-onnx-v1", "precision": "fp32", "opset": 17, "max_length": 256,
                "source_weights_sha256": SERVY_V5_WEIGHTS_SHA256, "parity_max_abs_logit_diff": worst,
                "files": {p.name: _sha256(p) for p in sorted(staged.iterdir()) if p.is_file()},
            }
            (staged / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            if out.exists():
                shutil.rmtree(out)
            shutil.move(str(staged), str(out))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.stdout.write(f"V5 ONNX export verified (max |diff| {worst:.1e}); {len(manifest['files'])} files in {out}")
