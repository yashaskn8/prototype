"""V5 ONNX runtime selection: a verified ONNX export is used; anything unverified falls back to PyTorch."""
import hashlib
import json
import tempfile
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase, override_settings

from core.ml import reranker


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class V5RuntimeSelectionTests(SimpleTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.ckpt = root / "v5"
        self.ckpt.mkdir()
        (self.ckpt / "model.safetensors").write_bytes(b"synthetic weights")
        (self.ckpt / "config.json").write_text("{}")
        self.weights_sha = _sha(self.ckpt / "model.safetensors")
        self.onnx = root / "onnx"
        self.onnx.mkdir()
        for name, data in (("model.onnx", b"graph"), ("model.onnx.data", b"tensors"), ("tokenizer.json", b"{}")):
            (self.onnx / name).write_bytes(data)
        self.write_manifest()
        reranker._cross_encoder = None
        reranker._loaded_config = None
        self.addCleanup(setattr, reranker, "_cross_encoder", None)
        self.addCleanup(setattr, reranker, "_loaded_config", None)

    def write_manifest(self, **changes):
        manifest = {"format": "servy-v5-onnx-v1", "precision": "fp32", "source_weights_sha256": self.weights_sha,
                    "files": {p.name: _sha(p) for p in self.onnx.iterdir() if p.name != "manifest.json"}}
        manifest.update(changes)
        (self.onnx / "manifest.json").write_text(json.dumps(manifest))

    def load(self, backend="onnx"):
        with override_settings(SERVY_RERANKER_MODEL_PATH=str(self.ckpt), SERVY_RERANKER_EXPECTED_SHA256=self.weights_sha,
                               SERVY_RERANKER_BACKEND=backend, SERVY_RERANKER_ONNX_DIR=str(self.onnx)), \
                mock.patch.object(reranker, "OnnxV5", side_effect=lambda d, t: ("onnx", d)) as onnx_cls, \
                mock.patch("sentence_transformers.CrossEncoder", return_value="torch-model") as torch_cls:
            model = reranker.get_reranker()
        return model, onnx_cls, torch_cls

    def test_verified_export_runs_on_onnx(self):
        model, onnx_cls, torch_cls = self.load()
        self.assertEqual(model[0], "onnx")
        torch_cls.assert_not_called()

    def test_tampered_export_file_falls_back_to_pytorch(self):
        (self.onnx / "model.onnx.data").write_bytes(b"tampered")
        model, onnx_cls, torch_cls = self.load()
        self.assertEqual(model, "torch-model")
        onnx_cls.assert_not_called()

    def test_export_from_other_weights_falls_back_to_pytorch(self):
        self.write_manifest(source_weights_sha256="0" * 64)
        model, onnx_cls, _ = self.load()
        self.assertEqual(model, "torch-model")
        onnx_cls.assert_not_called()

    def test_torch_backend_is_the_rollback(self):
        model, onnx_cls, _ = self.load(backend="torch")
        self.assertEqual(model, "torch-model")
        onnx_cls.assert_not_called()

    def test_tampered_checkpoint_loads_nothing(self):
        (self.ckpt / "model.safetensors").write_bytes(b"different weights")
        model, onnx_cls, torch_cls = self.load()
        self.assertIsNone(model)
        onnx_cls.assert_not_called()
        torch_cls.assert_not_called()
