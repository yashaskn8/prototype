"""Synthetic-only tests for the atomic, offline V5 checkpoint installer."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from tools.install_servy_v5_checkpoint import install_checkpoint


class InstallV5CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "src" / "best_model"
        self.source.mkdir(parents=True)
        self.weights = b"synthetic-only-v5-checkpoint-weights"
        (self.source / "model.safetensors").write_bytes(self.weights)
        (self.source / "config.json").write_text("{}", encoding="utf-8")
        (self.source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        self.expected = hashlib.sha256(self.weights).hexdigest()
        self.target = self.root / "installed" / "best_model"

    def test_verified_copy_and_preserves_source(self):
        output = install_checkpoint(self.source, self.target, self.expected)
        self.assertEqual(output, self.target)
        self.assertEqual((self.target / "model.safetensors").read_bytes(), self.weights)
        self.assertEqual((self.target / "tokenizer_config.json").read_text(), "{}")
        self.assertTrue((self.source / "model.safetensors").exists())
        self.assertEqual(list(self.target.parent.glob(".v5-install-*")), [])

    def test_wrong_weights_cannot_publish_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            install_checkpoint(self.source, self.target, "0" * 64)
        self.assertFalse(self.target.exists())

    def test_existing_install_is_not_overwritten(self):
        self.target.mkdir(parents=True)
        (self.target / "sentinel").write_text("existing")
        with self.assertRaises(FileExistsError):
            install_checkpoint(self.source, self.target, self.expected)
        self.assertEqual((self.target / "sentinel").read_text(), "existing")

    def test_refuses_source_inside_target(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            install_checkpoint(self.source, self.source / "nested", self.expected)

    def test_rejects_missing_config(self):
        (self.source / "config.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing config.json"):
            install_checkpoint(self.source, self.target, self.expected)

    @unittest.skipUnless(os.name == "nt", "directory junctions exist only on Windows")
    def test_rejects_directory_junction_without_symlink_privilege(self):
        # A junction needs no SeCreateSymbolicLinkPrivilege and Path.is_symlink()
        # reports it as False; the installer must still refuse it.
        import _winapi
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("outside the checkpoint", encoding="utf-8")
        junction = self.source / "linked_dir"
        _winapi.CreateJunction(str(outside), str(junction))
        self.assertFalse(junction.is_symlink())          # the gap being guarded
        with self.assertRaisesRegex(ValueError, "symlinks"):
            install_checkpoint(self.source, self.target, self.expected)
        self.assertFalse(self.target.exists())

    @unittest.skipUnless(os.name == "nt", "directory junctions exist only on Windows")
    def test_rejects_junction_as_source(self):
        import _winapi
        junction = self.root / "junction_source"
        _winapi.CreateJunction(str(self.source), str(junction))
        with self.assertRaisesRegex(ValueError, "real directory"):
            install_checkpoint(junction, self.target, self.expected)

    def test_rejects_symlink(self):
        link = self.source / "linked.bin"
        try:
            link.symlink_to(self.source / "config.json")
        except (NotImplementedError, OSError):
            self.skipTest("Symlink creation unavailable")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            install_checkpoint(self.source, self.target, self.expected)


if __name__ == "__main__":
    unittest.main()
