import json
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.test import APIClient
import requests

from core.models import Asset, Customer, Product, ProductCategory, ProductDomain, Site, Tenant, TenantMembership
from core.services.starlly_api import (
    StarllyApiClient,
    StarllyApiError,
    StarllyAuthenticationError,
    StarllyConfigurationError,
    StarllyConnectionError,
    StarllyTimeoutError,
    mask_secret,
    validate_base_url,
    validate_api_key,
)

SAMPLE_STARLLY_ASSETS = [
    {
        "id": 332,
        "name": "Split AC Unit 2T",
        "code": "SKY-AST-00001",
        "description": "AC-2T",
        "brand_id": 345,
        "product_id": 443,
        "model_number": "SAC-2024",
        "extra_details": None,
        "org_id": 273,
        "created_at": "2026-05-23T07:05:24.237648Z",
        "created_by_id": 1035,
        "updated_at": "2026-07-19T08:53:36.097420Z",
        "updated_by_id": 1035,
        "is_active": True,
        "attachments": [
            {
                "id": 9,
                "name": "03_Troubleshooting.txt",
                "file_url": "https://example.com/03_Troubleshooting.txt",
                "created_at": "2026-10-14T04:39:21Z",
            }
        ],
    },
    {
        "id": 333,
        "name": "Industrial Chiller 50T",
        "code": "SKY-AST-00002",
        "description": "Chiller-50T",
        "brand_id": 345,
        "product_id": 445,
        "model_number": "ICH-5000",
        "extra_details": None,
        "org_id": 273,
        "created_at": "2026-05-24T09:12:00.000000Z",
        "created_by_id": 1035,
        "updated_at": "2026-07-19T08:53:36.097420Z",
        "updated_by_id": 1035,
        "is_active": True,
        "attachments": [],
    },
]


class StarllyClientUnitTests(TestCase):
    """Unit tests for core/services/starlly_api.py client and validators."""

    def test_auth_header_format(self):
        client = StarllyApiClient(
            base_url="https://servybackdev.starlly.in",
            api_key="sk_test_secret_1234567890",
            org_id="273",
        )
        headers = client.get_auth_headers()
        self.assertEqual(headers["Authorization"], "Api-Key sk_test_secret_1234567890")
        self.assertNotIn("Bearer", headers["Authorization"])
        self.assertNotIn("Token", headers["Authorization"])
        self.assertEqual(headers["Accept"], "application/json")

    def test_incomplete_url_blocker(self):
        # https://.starlly.in is incomplete and must be reported as a configuration blocker
        incomplete_urls = [
            "https://.starlly.in",
            "http://.starlly.in",
            "https://starlly.in",
            "https://YOUR-ACTUAL-SUBDOMAIN.starlly.in",
            "",
            "ftp://test.starlly.in",
        ]
        for url in incomplete_urls:
            with self.assertRaises(StarllyConfigurationError) as ctx:
                validate_base_url(url)
            self.assertTrue(ctx.exception.is_blocker)
            self.assertTrue(
                "blocker" in str(ctx.exception).lower() or "not configured" in str(ctx.exception).lower() or "scheme" in str(ctx.exception).lower()
            )

    def test_valid_base_url(self):
        valid = validate_base_url("https://servybackdev.starlly.in/")
        self.assertEqual(valid, "https://servybackdev.starlly.in")

    def test_api_key_validation(self):
        with self.assertRaises(StarllyConfigurationError):
            validate_api_key("")
        with self.assertRaises(StarllyConfigurationError):
            validate_api_key("PASTE_YOUR_SECRET_KEY_HERE")
        self.assertEqual(validate_api_key("sk_valid_key"), "sk_valid_key")

    def test_mask_secret(self):
        self.assertEqual(mask_secret(None), "<none>")
        self.assertEqual(mask_secret("short"), "sk_***")
        self.assertEqual(mask_secret("sk_90f5f62d_secret_key"), "sk_9..._key")

    @patch("requests.get")
    def test_fetch_assets_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_STARLLY_ASSETS
        mock_get.return_value = mock_resp

        client = StarllyApiClient(
            base_url="https://servybackdev.starlly.in",
            api_key="sk_test_key_12345",
            org_id="273",
        )
        res = client.fetch_assets(limit=10, offset=0)

        self.assertEqual(res["count"], 2)
        self.assertEqual(res["total_returned"], 2)
        self.assertEqual(res["organisation_id"], 273)
        self.assertEqual(res["assets"][0]["id"], 332)
        self.assertEqual(res["assets"][0]["code"], "SKY-AST-00001")
        self.assertEqual(res["assets"][0]["org_id"], 273)

        # Ensure correct header and params were passed
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Api-Key sk_test_key_12345")
        self.assertEqual(kwargs["params"], {"limit": 10, "offset": 0})

    @patch("requests.get")
    def test_fetch_assets_paginated_format(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "count": 50,
            "results": SAMPLE_STARLLY_ASSETS,
        }
        mock_get.return_value = mock_resp

        client = StarllyApiClient(
            base_url="https://servybackdev.starlly.in",
            api_key="sk_test_key_12345",
            org_id=None,
        )
        res = client.fetch_assets(limit=2, offset=0)
        self.assertEqual(res["count"], 50)
        self.assertEqual(res["total_returned"], 2)

    @patch("requests.get")
    def test_fetch_assets_auth_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"detail": "Authentication credentials were not provided."}
        mock_get.return_value = mock_resp

        client = StarllyApiClient(
            base_url="https://servybackdev.starlly.in",
            api_key="sk_bad_key",
        )
        with self.assertRaises(StarllyAuthenticationError):
            client.fetch_assets()

    @patch("requests.get")
    def test_fetch_assets_timeout(self, mock_get):
        mock_get.side_effect = requests.Timeout("Connection timed out")

        client = StarllyApiClient(
            base_url="https://servybackdev.starlly.in",
            api_key="sk_test_key",
        )
        with self.assertRaises(StarllyTimeoutError):
            client.fetch_assets()

    @patch("requests.get")
    def test_fetch_assets_connection_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("DNS failure")

        client = StarllyApiClient(
            base_url="https://servybackdev.starlly.in",
            api_key="sk_test_key",
        )
        with self.assertRaises(StarllyConnectionError):
            client.fetch_assets()


@override_settings(
    STARLLY_API_BASE_URL="https://servybackdev.starlly.in",
    STARLLY_API_KEY="sk_test_dummy_key_12345",
    STARLLY_ORGANISATION_ID="273",
)
class ExternalAssetsApiViewTests(TestCase):
    """Integration tests for GET /api/external/assets/ endpoint."""

    def setUp(self):
        self.client = APIClient()

        # Create Tenant
        self.tenant = Tenant.objects.create(name="Starlly Tester", slug="starlly-tester")

        # Create Staff User
        self.staff_user = User.objects.create_user(
            username="test_tech", password="TechPassword123!", email="tech@starlly.in"
        )
        TenantMembership.objects.create(
            user=self.staff_user,
            tenant=self.tenant,
            role="technician",
            can_view_confidential=True,
        )

        # Create Customer User
        self.customer = Customer.objects.create(tenant=self.tenant, name="Acme Customer")
        self.customer_user = User.objects.create_user(
            username="test_cust", password="CustPassword123!", email="cust@acme.example"
        )
        TenantMembership.objects.create(
            user=self.customer_user,
            tenant=self.tenant,
            role="customer",
            customer=self.customer,
        )

        # Pre-seed one local asset with code 'SKY-AST-00001' to test is_synced resolution
        domain = ProductDomain.objects.create(tenant=self.tenant, name="HVAC")
        category = ProductCategory.objects.create(tenant=self.tenant, domain=domain, name="Air Conditioning")
        product = Product.objects.create(tenant=self.tenant, category=category, name="Split AC")
        site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Main Office")
        self.local_asset = Asset.objects.create(
            tenant=self.tenant,
            customer=self.customer,
            site=site,
            product=product,
            name="Existing Split AC",
            asset_code="SKY-AST-00001",
            status="active",
        )

    def test_anonymous_request_rejected(self):
        response = self.client.get("/api/external/assets/")
        self.assertIn(response.status_code, [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN])

    def test_customer_user_forbidden(self):
        self.client.force_authenticate(user=self.customer_user)
        session = self.client.session
        session["tenant_id"] = self.tenant.id
        session.save()

        response = self.client.get("/api/external/assets/")
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    @patch("core.services.starlly_api.requests.get")
    def test_staff_user_fetches_assets_with_sync_status(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_STARLLY_ASSETS
        mock_get.return_value = mock_resp

        self.client.force_authenticate(user=self.staff_user)
        session = self.client.session
        session["tenant_id"] = self.tenant.id
        session.save()

        initial_count = Asset.objects.count()

        response = self.client.get("/api/external/assets/?limit=10&offset=0")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        data = response.json()
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["sync_status"], "read_only")
        self.assertEqual(data["tenant"]["id"], self.tenant.id)

        assets = data["assets"]
        self.assertEqual(len(assets), 2)

        # Asset 1 (SKY-AST-00001) exists locally -> is_synced=True, local_asset_id=self.local_asset.id
        self.assertEqual(assets[0]["code"], "SKY-AST-00001")
        self.assertTrue(assets[0]["is_synced"])
        self.assertEqual(assets[0]["local_asset_id"], self.local_asset.id)
        self.assertEqual(assets[0]["id"], 332)
        self.assertEqual(assets[0]["org_id"], 273)

        # Asset 2 (SKY-AST-00002) does NOT exist locally -> is_synced=False, local_asset_id=None
        self.assertEqual(assets[1]["code"], "SKY-AST-00002")
        self.assertFalse(assets[1]["is_synced"])
        self.assertIsNone(assets[1]["local_asset_id"])

        # Critical: Verify zero DB mutations occurred (no auto-sync or overwrites)
        self.assertEqual(Asset.objects.count(), initial_count)

    @patch("core.services.starlly_api.requests.get")
    def test_configuration_blocker_incomplete_subdomain(self, mock_get):
        self.client.force_authenticate(user=self.staff_user)
        session = self.client.session
        session["tenant_id"] = self.tenant.id
        session.save()

        with override_settings(STARLLY_API_BASE_URL="https://.starlly.in"):
            response = self.client.get("/api/external/assets/")
            self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
            data = response.json()
            self.assertEqual(data["error"], "Configuration Blocker")
            self.assertTrue(data["is_blocker"])
            self.assertFalse(data["is_configured"])
            self.assertIn("incomplete", data["detail"].lower())

    @patch("core.services.starlly_api.requests.get")
    def test_starlly_auth_error_handling(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"detail": "Authentication credentials were not provided."}
        mock_get.return_value = mock_resp

        self.client.force_authenticate(user=self.staff_user)
        session = self.client.session
        session["tenant_id"] = self.tenant.id
        session.save()

        response = self.client.get("/api/external/assets/")
        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        data = response.json()
        self.assertEqual(data["error"], "Starlly Authentication Failed")
        # Ensure secret key is NEVER leaked in response
        self.assertNotIn("sk_test_dummy_key_12345", json.dumps(data))

    @patch("core.services.starlly_api.requests.get")
    def test_starlly_timeout_handling(self, mock_get):
        mock_get.side_effect = requests.Timeout("Gateway timeout")

        self.client.force_authenticate(user=self.staff_user)
        session = self.client.session
        session["tenant_id"] = self.tenant.id
        session.save()

        response = self.client.get("/api/external/assets/")
        self.assertEqual(response.status_code, status.HTTP_504_GATEWAY_TIMEOUT)
        data = response.json()
        self.assertEqual(data["error"], "Starlly API Timeout")
        self.assertNotIn("sk_test_dummy_key_12345", json.dumps(data))
