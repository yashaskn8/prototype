"""Backup safeguards: local temporary files only, no network."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from servy_checkpoint_backup import backup_checkpoint, is_temporary_colab_path


class TestCheckpointBackup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_directory_roundtrip_verified(self):
        source = self.root / "source"
        source.mkdir()
        (source / "config.json").write_text('{"model": "test"}')
        (source / "weights").mkdir()
        (source / "weights" / "model.bin").write_bytes(b"\0\1\2")
        output = self.root / "approved" / "checkpoint"
        output.parent.mkdir()
        result = backup_checkpoint(source, output)
        self.assertEqual(result["status"], "VERIFIED_BACKUP")
        self.assertEqual(result["files"], 2)
        self.assertEqual((output / "weights/model.bin").read_bytes(), b"\0\1\2")

    def test_zip_file_roundtrip_verified(self):
        path = self.root / "weights.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("model.bin", b"hello")
        dest = self.root / "approved" / "archive.zip"
        dest.parent.mkdir()
        result = backup_checkpoint(path, dest)
        self.assertEqual(result["files"], 1)
        self.assertEqual(dest.read_bytes(), path.read_bytes())
        with zipfile.ZipFile(dest) as z:
            self.assertEqual(z.read("model.bin"), b"hello")

    def test_refuses_overwrite(self):
        folder = self.root / "source"
        folder.mkdir()
        (folder / "weight").write_bytes(b"x")
        target = self.root / "store"
        target.mkdir()
        (target / "existing").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            backup_checkpoint(folder, target)
        self.assertEqual((target / "existing").read_bytes(), b"keep")

    def test_nested_destination_rejected(self):
        source = self.root / "source"
        source.mkdir()
        (source / "file").write_bytes(b"123")
        with self.assertRaisesRegex(ValueError, "nested"):
            backup_checkpoint(source, source / "nested")
        self.assertFalse((source / "nested").exists())

    def test_ephemeral_colab_paths_rejected(self):
        self.assertTrue(is_temporary_colab_path("/content/checkpoints"))
        self.assertFalse(is_temporary_colab_path("/content/drive/MyDrive/checkpoints"))
        source = self.root / "weights.zip"
        source.write_bytes(b"test")
        with self.assertRaisesRegex(ValueError, "ephemeral"):
            backup_checkpoint(source, "/content/disposable.zip")

    def test_source_symlink_rejected(self):
        source = self.root / "source"
        source.mkdir()
        (source / "weights").write_bytes(b"test")
        try:
            (source / "weights_link").symlink_to(source / "weights")
        except (OSError, NotImplementedError):
            self.skipTest("symlink unavailable")
        destination = self.root / "store"
        with self.assertRaisesRegex(ValueError, "symlink"):
            backup_checkpoint(source, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
