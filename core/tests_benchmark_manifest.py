from django.test import TestCase
from core.models import Tenant, Product, Asset
from core.ml.benchmark_manifest import (
    load_manifest,
    build_manifest,
    resolve_query_asset,
    ROLE_RELEVANCE,
    ROLE_ABSTENTION,
    ROLE_SECURITY,
)


class BenchmarkManifestTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.tenant9 = Tenant.objects.create(name="Skydream HVAC", slug="skydream")
        cls.prod_vrf = Product.objects.create(tenant=cls.tenant9, name="VRF Outdoor Module")
        cls.asset_vrf = Asset.objects.create(
            tenant=cls.tenant9,
            name="VRF Outdoor Module",
            product=cls.prod_vrf,
            serial_number="SKY-AST-00002",
        )

        cls.tenant6 = Tenant.objects.create(name="Starlly Tester", slug="starlly")
        cls.prod_pallet = Product.objects.create(tenant=cls.tenant6, name="Pallet Lift")
        cls.asset_pallet = Asset.objects.create(
            tenant=cls.tenant6,
            name="Pallet Lift-10",
            product=cls.prod_pallet,
            serial_number="PAL-010",
        )

    def setUp(self):
        self.manifest = build_manifest()

    def test_total_case_count_and_denominators(self):
        self.assertEqual(self.manifest["total_benchmark_cases"], 64)
        denoms = self.manifest["primary_denominators"]
        self.assertEqual(denoms["relevance_queries"], 43)
        self.assertEqual(denoms["abstention_queries"], 10)
        self.assertEqual(denoms["security_queries"], 11)
        self.assertEqual(denoms["total_abstain_expected"], 21)
        self.assertEqual(
            denoms["relevance_queries"] + denoms["total_abstain_expected"],
            64,
        )

    def test_unique_stable_ids(self):
        ids = [c["query_id"] for c in self.manifest["cases"]]
        self.assertEqual(len(ids), 64)
        self.assertEqual(len(set(ids)), 64)

    def test_all_14_categories_present(self):
        cats = self.manifest["category_counts"]
        self.assertEqual(len(cats), 14)
        expected_cats = {
            "exact_error_code", "general_symptom", "model_trap",
            "misspelled_query", "multi_symptom", "model_ambiguity",
            "technician_only", "equipment_crosstalk", "unsupported_question",
            "out_of_domain", "doc_id_attack", "revoked_doc_query",
            "tenant_isolation_attack", "boundary_injection",
        }
        self.assertEqual(set(cats.keys()), expected_cats)

    def test_non_overlapping_primary_roles(self):
        for c in self.manifest["cases"]:
            self.assertIn(
                c["primary_evaluation_role"],
                [ROLE_RELEVANCE, ROLE_ABSTENTION, ROLE_SECURITY],
            )
            self.assertTrue(c["scorable"])
            self.assertIsNone(c["unscorable_reason"])

    def test_resolve_query_asset_exact_and_product_mapping(self):
        # 1. Exact asset name match (Tenant 9)
        a9 = resolve_query_asset(self.tenant9.id, "VRF Outdoor Module")
        self.assertIsNotNone(a9)
        self.assertEqual(a9.id, self.asset_vrf.id)

        # 2. Product name match to specific asset (Tenant 6)
        a6 = resolve_query_asset(self.tenant6.id, "Pallet Lift")
        self.assertIsNotNone(a6)
        self.assertEqual(a6.id, self.asset_pallet.id)

        # 3. General or empty returns None
        self.assertIsNone(resolve_query_asset(self.tenant9.id, "general"))
        self.assertIsNone(resolve_query_asset(self.tenant9.id, ""))
