"""Safety rules of the MySQL backup/restore tool (no database or MySQL client needed)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import servy_mysql_backup as tool

ENV = "MYSQL_HOST=127.0.0.1\nMYSQL_PORT=3306\nMYSQL_DATABASE={db}\nMYSQL_USER=u\nMYSQL_PASSWORD=p\n"


class BackupToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.backup = self.root / "servy_backup_servy_dev"
        (self.backup / "mysql").mkdir(parents=True)
        dump = self.backup / "mysql" / "servy_dev.sql"
        dump.write_text("CREATE TABLE `core_asset` (id int);\n"
                        "INSERT INTO `core_asset` VALUES (1);\nINSERT INTO `core_asset` VALUES (2);\n"
                        "INSERT INTO `core_knowledgechunk` VALUES (7,'text with ),( inside');\n", encoding="utf-8")
        files = tool.hash_tree(self.backup)
        manifest = {"source_database": "servy_dev", "row_counts": tool.dump_row_counts(dump), "files": files}
        (self.backup / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def env_file(self, db):
        path = self.root / f"{db}.env"
        path.write_text(ENV.format(db=db), encoding="utf-8")
        return str(path)

    def test_row_counts_come_from_the_dump_snapshot(self):
        counts = tool.dump_row_counts(self.backup / "mysql" / "servy_dev.sql")
        self.assertEqual(counts["core_asset"], 2)
        self.assertEqual(counts["core_knowledgechunk"], 1)
        self.assertEqual(counts["core_servicecall"], 0)

    def test_verify_accepts_an_intact_backup_and_rejects_tampering(self):
        tool.verify(SimpleNamespace(backup=str(self.backup)))
        (self.backup / "mysql" / "servy_dev.sql").write_text("DROP DATABASE servy_dev;", encoding="utf-8")
        with self.assertRaises(SystemExit):
            tool.verify(SimpleNamespace(backup=str(self.backup)))

    def test_verify_rejects_an_added_environment_file(self):
        (self.backup / ".env").write_text("SECRET=1", encoding="utf-8")
        with self.assertRaises(SystemExit):
            tool.verify(SimpleNamespace(backup=str(self.backup)))

    def restore(self, db, confirm):
        args = SimpleNamespace(backup=str(self.backup), env_file=self.env_file(db), confirm_target=confirm,
                               chroma_to=None, media_to=None)
        with mock.patch.object(tool.subprocess, "run") as run, \
                mock.patch.object(tool, "row_counts", return_value=tool.dump_row_counts(
                    self.backup / "mysql" / "servy_dev.sql")):
            tool.restore(args)
        return run

    def test_restore_refuses_production_source_and_unconfirmed_targets(self):
        for db, confirm in (("servy", "servy"), ("servy_dev", "servy_dev"), ("test_servy_dev", "other")):
            with self.subTest(db=db):
                with mock.patch.object(tool.subprocess, "run") as run, self.assertRaises(SystemExit):
                    tool.restore(SimpleNamespace(backup=str(self.backup), env_file=self.env_file(db),
                                                 confirm_target=confirm, chroma_to=None, media_to=None))
                run.assert_not_called()

    def test_confirmed_isolated_restore_runs_the_mysql_client(self):
        run = self.restore("test_servy_dev", "test_servy_dev")
        command = run.call_args.args[0]
        self.assertTrue(command[0].endswith("mysql.exe"))
        self.assertEqual(command[-1], "test_servy_dev")
        self.assertFalse(any("password" in part.lower() and "=" in part and "extra-file" not in part
                             for part in command))

    def test_password_is_never_on_the_command_line(self):
        with tool._Options({"MYSQL_HOST": "h", "MYSQL_PORT": "1", "MYSQL_USER": "u", "MYSQL_PASSWORD": "pw-123"}) as cnf:
            self.assertIn("pw-123", Path(cnf).read_text(encoding="utf-8"))
        self.assertFalse(Path(cnf).exists())

    def test_backups_inside_the_repository_are_refused(self):
        repo = Path(tool.__file__).resolve().parents[1]
        args = SimpleNamespace(env_file=self.env_file("servy_dev"), out=str(repo / "backups"), chroma=None, media=None)
        with mock.patch.object(tool.subprocess, "run") as run, self.assertRaises(SystemExit):
            tool.backup(args)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
