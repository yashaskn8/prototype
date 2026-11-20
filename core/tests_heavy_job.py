"""Machine-wide lock for memory-heavy model jobs."""
import os
import tempfile
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

from core.ml import heavy_job


class HeavyJobLockTests(SimpleTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(heavy_job, "LOCK", Path(tmp.name) / "job.lock")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_second_live_job_is_refused_immediately(self):
        heavy_job.LOCK.write_text("424242 servy gateway", encoding="utf-8")
        with mock.patch.object(heavy_job, "_alive", return_value=True):
            with self.assertRaisesRegex(heavy_job.HeavyJobBusy, "servy gateway, PID 424242"):
                heavy_job.acquire("export_v5_onnx")
        self.assertEqual(heavy_job.LOCK.read_text(encoding="utf-8"), "424242 servy gateway")

    def test_stale_lock_from_a_dead_process_is_replaced(self):
        heavy_job.LOCK.write_text("424242 crashed job", encoding="utf-8")
        with mock.patch.object(heavy_job, "_alive", return_value=False):
            with heavy_job.heavy_job("export_v5_onnx"):
                self.assertTrue(heavy_job.LOCK.read_text(encoding="utf-8").startswith(str(os.getpid())))
        self.assertFalse(heavy_job.LOCK.exists())

    def test_release_only_removes_our_own_lock(self):
        heavy_job.LOCK.write_text("424242 other job", encoding="utf-8")
        heavy_job.release()
        self.assertTrue(heavy_job.LOCK.exists())

    def test_lock_is_released_after_a_failure(self):
        with self.assertRaises(ValueError):
            with heavy_job.heavy_job("check"):
                raise ValueError("model failed")
        self.assertFalse(heavy_job.LOCK.exists())
