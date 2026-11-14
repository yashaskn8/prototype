"""ServiceCall attachments: limits, types, storage, isolation and authorization."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from django.test import SimpleTestCase, TestCase, override_settings

from core.models import Customer, ServiceCall, ServiceCallAttachment, Site, Tenant
from core.services.call_attachments import (
    MAX_ATTACHMENT_BYTES, AttachmentError, authorized_attachment, store_attachments, validate_attachments)

ROOT = Path(__file__).resolve().parent.parent
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PDF = b"%PDF-1.4\n%synthetic\n"


def png(i=0):
    return {"name": f"photo-{i}.png", "content_type": "image/png", "content": PNG + bytes([i])}


class AttachmentValidationTests(SimpleTestCase):
    def test_zero_one_and_ten_attachments_are_accepted(self):
        self.assertEqual(validate_attachments([]), [])
        self.assertEqual(len(validate_attachments([png()])), 1)
        self.assertEqual(len(validate_attachments([png(i) for i in range(10)])), 10)

    def test_eleven_attachments_are_rejected(self):
        with self.assertRaisesRegex(AttachmentError, "at most 10"):
            validate_attachments([png(i) for i in range(11)])

    def test_allowed_types(self):
        items = validate_attachments([
            png(), {"name": "report.pdf", "content_type": "application/pdf", "content": PDF},
            {"name": "notes.txt", "content_type": "text/plain", "content": "Unit is noisy.".encode()},
            {"name": "a.jpg", "content_type": "image/jpeg", "content": b"\xff\xd8\xff\xe0" + b"0" * 10},
            {"name": "a.gif", "content_type": "image/gif", "content": b"GIF89a" + b"0" * 10},
            {"name": "a.webp", "content_type": "image/webp", "content": b"RIFF\x00\x00\x00\x00WEBP" + b"0" * 4},
        ])
        self.assertEqual(len(items), 6)

    def test_invalid_types_extensions_and_spoofed_content_are_rejected(self):
        cases = (
            {"name": "run.exe", "content_type": "application/x-msdownload", "content": b"MZ\x90\x00"},
            {"name": "page.html", "content_type": "text/html", "content": b"<script>alert(1)</script>"},
            {"name": "photo.pdf", "content_type": "image/png", "content": PNG},          # extension mismatch
            {"name": "fake.png", "content_type": "image/png", "content": b"MZ\x90\x00not a png"},
            {"name": "fake.pdf", "content_type": "application/pdf", "content": b"not a pdf"},
            {"name": "bin.txt", "content_type": "text/plain", "content": b"\xff\xfe\x00\x00"},
            {"name": "empty.png", "content_type": "image/png", "content": b""},
        )
        for item in cases:
            with self.subTest(name=item["name"]):
                with self.assertRaises(AttachmentError):
                    validate_attachments([item])

    def test_size_limits(self):
        big = {"name": "big.pdf", "content_type": "application/pdf", "content": PDF + b"0" * MAX_ATTACHMENT_BYTES}
        with self.assertRaisesRegex(AttachmentError, "larger than 25 MB"):
            validate_attachments([big])
        half = {"name": "half.pdf", "content_type": "application/pdf",
                "content": PDF + b"0" * (MAX_ATTACHMENT_BYTES // 2 + 10)}
        with self.assertRaisesRegex(AttachmentError, "25 MB total"):
            validate_attachments([half, dict(half, name="half2.pdf")])

    def test_names_are_reduced_to_a_safe_basename(self):
        (name, _t, _c), = validate_attachments([dict(png(), name="..\\..\\etc/passwd/photo.png")])
        self.assertEqual(name, "photo.png")


class AttachmentStorageTests(TestCase):
    def setUp(self):
        self.media = tempfile.TemporaryDirectory()
        self.addCleanup(self.media.cleanup)
        override = override_settings(MEDIA_ROOT=self.media.name)
        override.enable()
        self.addCleanup(override.disable)
        self.tenant = Tenant.objects.create(name="Synthetic", slug="synthetic-attach")
        self.other_tenant = Tenant.objects.create(name="Other", slug="synthetic-attach-other")
        self.customer = Customer.objects.create(tenant=self.tenant, name="C1")
        self.other_customer = Customer.objects.create(tenant=self.tenant, name="C2")
        site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="S1")
        self.call = ServiceCall.objects.create(tenant=self.tenant, servy_id=90001, customer=self.customer,
                                               site=site, complaint_text="not cooling")
        other_site = Site.objects.create(tenant=self.tenant, customer=self.other_customer, name="S2")
        self.other_call = ServiceCall.objects.create(tenant=self.tenant, servy_id=90002,
                                                     customer=self.other_customer, site=other_site)

    def test_files_and_metadata_persist_and_reload(self):
        items = validate_attachments([png(i) for i in range(10)])
        saved = store_attachments(self.call, items)
        self.assertEqual(ServiceCallAttachment.objects.filter(service_call=self.call).count(), 10)
        reloaded = ServiceCallAttachment.objects.get(pk=saved[3].pk)   # fresh read, as after a restart
        self.assertTrue(Path(reloaded.file.path).is_file())
        self.assertTrue(str(Path(reloaded.file.path).resolve()).startswith(str(Path(self.media.name).resolve())))
        with reloaded.file.open("rb") as handle:
            self.assertEqual(handle.read(), PNG + bytes([3]))
        self.assertEqual((reloaded.original_name, reloaded.content_type, reloaded.size),
                         ("photo-3.png", "image/png", len(PNG) + 1))

    def test_zero_attachments_store_nothing(self):
        self.assertEqual(store_attachments(self.call, validate_attachments([])), [])
        self.assertEqual(ServiceCallAttachment.objects.count(), 0)

    def test_lookup_only_through_the_callers_authorized_calls(self):
        att = store_attachments(self.call, validate_attachments([png()]))[0]
        mine = ServiceCall.objects.filter(tenant=self.tenant, customer=self.customer)
        theirs = ServiceCall.objects.filter(tenant=self.tenant, customer=self.other_customer)
        other_tenant = ServiceCall.objects.filter(tenant=self.other_tenant)
        self.assertEqual(authorized_attachment(mine, self.call.id, att.id), att)
        self.assertIsNone(authorized_attachment(theirs, self.call.id, att.id))
        self.assertIsNone(authorized_attachment(other_tenant, self.call.id, att.id))
        self.assertIsNone(authorized_attachment(mine, self.other_call.id, att.id))   # wrong call
        self.assertIsNone(authorized_attachment(mine, self.call.id, "../1"))

    def test_cross_tenant_attachment_is_rejected(self):
        from django.core.exceptions import ValidationError
        with self.assertRaises(ValidationError):
            ServiceCallAttachment.objects.create(tenant=self.other_tenant, service_call=self.call,
                                                 original_name="x.png", content_type="image/png", size=1, sha256="0")


class DevMediaIsolationTests(SimpleTestCase):
    def test_servy_dev_refuses_the_repository_media_folder(self):
        env = {k: v for k, v in os.environ.items() if k not in ("SERVY_MEDIA_ROOT",)}
        env.update(DJANGO_SETTINGS_MODULE="servy_rag.settings", SERVY_DB_ENGINE="mysql",
                   SERVY_MYSQL_ENV_FILE=".env.mysql.servy_dev", DJANGO_DEBUG="1")
        if not (ROOT / ".env.mysql.servy_dev").is_file():
            self.skipTest("servy_dev is not provisioned on this machine")
        result = subprocess.run([sys.executable, "-c", "import django; django.setup()"],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SERVY_MEDIA_ROOT", result.stderr)
