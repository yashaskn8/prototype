import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, SimpleTestCase, override_settings
from rest_framework.test import APIRequestFactory, force_authenticate

from core.models import (Tenant, Customer, Site, Asset, Product, KnowledgeDocument,
                         KnowledgeChunk, CustomerInteraction, TenantMembership,
                         ServiceCall, CustomerConversation, DiagnosticSession)
from core.services.customer_answer import build_customer_answer, open_citation, TITLES
from core.services.llm import extract_technical_numbers, extract_typed_quantities, check_evidence_sufficiency, verify_quantity_support


class NumericObservationTests(SimpleTestCase):
    def test_messy_fault_terms_normalize_without_changing_model_codes(self):
        from core.services.retriever import normalize_domain_terms
        self.assertEqual(
            normalize_domain_terms('VRF-A outdor unti low presure faault capcity'),
            'VRF-A outdoor unit low pressure fault capacity',
        )
        self.assertEqual(normalize_domain_terms('low preshure capcityy vrf hlep'),
                         'low pressure capacity vrf help')

    def test_misspelled_ice_formation_complaint_normalizes_for_retrieval(self):
        from core.services.retriever import normalize_domain_terms
        self.assertEqual(
            normalize_domain_terms("splitt ac icce forming on coolng coyl"),
            "split ac ice form on cooling coil",
        )
        # Technical model/error identifiers remain unchanged by word correction.
        self.assertEqual(
            normalize_domain_terms("VRF-A E-04 splitt"),
            "VRF-A E-04 split",
        )

    def test_repair_permission_guard_does_not_block_manual_fact_question(self):
        from core.services.customer_answer import _question_guard
        self.assertEqual(
            _question_guard('Can I use another equipment manual to fix this VRF fault?'),
            'safety',
        )
        self.assertEqual(_question_guard('May I repair the VRF unit?'), 'safety')
        self.assertIsNone(_question_guard('What does the VRF manual say about low-pressure faults?'))
        self.assertEqual(_question_guard('low preshure capcityy vrf?? hlep'), 'clarify')
        self.assertEqual(_question_guard('It is not cooling right. Is it pressure or something else?'), 'clarify')
        self.assertEqual(_question_guard('why is the split ac unit not working?'), 'clarify')
        self.assertIsNone(_question_guard('Why is the split AC unit not working after showing E-04?'))

    def test_exact_handoff_observation_is_not_mistaken_for_exact_pressure_request(self):
        from core.services.customer_answer import _question_guard
        self.assertIsNone(_question_guard(
            'What exact information from the display should I record for this low-pressure fault?'
        ))
        self.assertEqual(
            _question_guard('What exact pressure value is specified for this equipment?'),
            'unsupported',
        )
        self.assertEqual(
            _question_guard('What exact manufacturer-approved range applies?'),
            'unsupported',
        )

    def test_elapsed_symptom_not_required_specification(self):
        q = 'My VRF outdoor unit has weak cooling after 20 minutes. What does the equipment troubleshooting documentation say about possible causes?'
        self.assertNotIn('20', extract_technical_numbers(q))
        source = SimpleNamespace(title='Synthetic VRF outdoor unit troubleshooting',text='Weak cooling and capacity faults may be caused by low refrigerant charge or restricted airflow.',
            heading='Weak cooling capacity faults', score=1.0, source_verified=True,
            equipment_applicable=True, customer_action_authorized=False)
        self.assertTrue(check_evidence_sufficiency(q, None, [source]))

    def test_specs_timing_limits_and_repeated_numbers_are_preserved(self):
        for q in ('Pressure must remain below 20 bar', 'Must stop after 20 minutes',
                  'What is the maximum time after 20 minutes?', 'Cooling weak after 20 minutes at 20 bar',
                  'Is it safe to operate for 20 minutes?', 'Wait for 20 minutes before operation'):
            self.assertTrue(any('20' in n for n in extract_technical_numbers(q)), q)

    def test_explicit_timing_constraint_after_value_and_dimension(self):
        quantity=extract_typed_quantities('Stopping after 20 minutes is the maximum allowed.')[0]
        self.assertEqual((quantity.value,quantity.dimension,quantity.unit),(20,'time','minute'))
        self.assertFalse(verify_quantity_support(quantity,extract_typed_quantities('20 bar')))
        self.assertFalse(verify_quantity_support(quantity,extract_typed_quantities('20 seconds')))
        self.assertTrue(verify_quantity_support(quantity,extract_typed_quantities('Maximum 20 minutes')))
        self.assertEqual(extract_typed_quantities('Cooling weak after 20 minutes. The pressure must be 5 bar.')[0].dimension,'pressure')

    def test_presentation_cleanup_preserves_decimal_and_signed_values(self):
        from core.services.customer_answer import _clean_line
        self.assertEqual(_clean_line('3.4 bar can cause pressure faults.'),'3.4 bar can cause pressure faults.')
        self.assertEqual(_clean_line('-0.5 degrees is a recorded value.'),'-0.5 degrees is a recorded value.')
        self.assertEqual(_clean_line('- **Restricted airflow** can cause cooling faults.'),'Restricted airflow can cause cooling faults.')


@override_settings(SERVY_CUSTOMER_DATA_MODE='test_isolated', SERVY_RAG_MIN_SCORE=.04)
class CustomerAnswerSecurityTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.tenant=Tenant.objects.create(name='Synthetic answer test',slug='answer-test')
        cls.other=Tenant.objects.create(name='Other synthetic tenant',slug='answer-other')
        cls.customer=Customer.objects.create(tenant=cls.tenant,name='Synthetic customer')
        cls.foreign=Customer.objects.create(tenant=cls.tenant,name='Other synthetic customer')
        cls.site=Site.objects.create(tenant=cls.tenant,customer=cls.customer,name='Synthetic site')
        cls.product=Product.objects.create(tenant=cls.tenant,name='Synthetic cooling unit')
        cls.asset=Asset.objects.create(tenant=cls.tenant,customer=cls.customer,site=cls.site,
            product=cls.product,name='Synthetic VRF',model_number='VRF-A',asset_code='UNIT-A')
        cls.user=User.objects.create_user('answer-tester',password='SyntheticOnly!123')
        TenantMembership.objects.create(tenant=cls.tenant,user=cls.user,customer=cls.customer,role='customer')
        cls.doc=KnowledgeDocument.objects.create(tenant=cls.tenant,product=cls.product,
            title='Synthetic cooling guide',is_rag_enabled=True,index_status='INDEXED',checksum_sha256='a'*64,
            content_text='Synthetic source only')
        cls.chunk=KnowledgeChunk.objects.create(tenant=cls.tenant,document=cls.doc,
            heading='Weak cooling',text='## Weak cooling\nPossible causes:\n- Reduced airflow can cause weak cooling.\n- Low-pressure capacity faults can cause weak cooling.\nObserve and report the displayed cooling error code.')
        cls.interaction=CustomerInteraction.objects.create(tenant=cls.tenant,customer=cls.customer,asset=cls.asset,
            site=cls.site,created_by=cls.user,question='Weak cooling after 20 minutes: possible causes?',answer='Untrusted model output')

    def build(self, refs=None):
        return build_customer_answer(self.interaction,None,refs if refs is not None else [
            {'document_id':self.doc.id,'chunk_id':self.chunk.id,'score':1.0}])

    def context(self):
        return {'user':self.user,'tenant':self.tenant,'customer':self.customer,'role':'customer'}

    def token(self):
        return parse_qs(urlparse(self.build()['citations'][0]['url']).query)['citation'][0]

    def test_four_sections_claim_binding_and_no_heading_actions(self):
        result=self.build(); p=result['presentation']
        self.assertEqual([s['title'] for s in p['sections']],list(TITLES))
        self.assertEqual(len(p['sections']),4)
        ids={c['id'] for c in p['citations']}
        for section in p['sections']:
            for claim in section['claims']:
                self.assertTrue(set(claim['evidence_ids'])<=ids)
        self.assertNotIn('Untrusted model output',json.dumps(p))
        self.assertFalse(any(a['action'].startswith('##') for a in result['customer_safe_actions']))

    def test_exact_excerpt_dedup_and_no_operational_writes(self):
        models=(ServiceCall,CustomerInteraction,CustomerConversation,DiagnosticSession)
        before=[m.objects.count() for m in models]
        source=open_citation(self.token(),self.context())
        self.assertIn(source['excerpt'],self.chunk.text)
        refs=[{'document_id':self.doc.id,'chunk_id':self.chunk.id,'score':1.0}]*3
        self.assertEqual(len(self.build(refs)['citations']),len(self.build()['citations']))
        self.assertEqual(before,[m.objects.count() for m in models])

    def test_access_revocation_quarantine_confidential_and_deletion(self):
        token=self.token()
        for field,value in [('is_confidential',True),('is_rag_enabled',False),('disable_sharing',True),('customer_id',self.foreign.id)]:
            previous=getattr(self.doc,field)
            KnowledgeDocument.objects.filter(pk=self.doc.pk).update(**{field:value})
            with self.assertRaises(Exception):open_citation(token,self.context())
            KnowledgeDocument.objects.filter(pk=self.doc.pk).update(**{field:previous})
        KnowledgeChunk.objects.filter(pk=self.chunk.pk).update(is_quarantined=True)
        with self.assertRaises(Exception):open_citation(token,self.context())
        KnowledgeChunk.objects.filter(pk=self.chunk.pk).update(is_quarantined=False,text='Changed source')
        with self.assertRaises(Exception):open_citation(token,self.context())
        self.chunk.delete()
        with self.assertRaises(Exception):open_citation(token,self.context())

    def test_wrong_customer_tenant_user_and_tampered_tokens(self):
        token=self.token()
        for key,value in [('customer',self.foreign),('tenant',self.other),('user',SimpleNamespace(id=-1))]:
            with self.assertRaises(Exception):open_citation(token,{**self.context(),key:value})
        with self.assertRaises(Exception):open_citation(token+'tampered',self.context())

    def test_api_revalidates_current_identity_and_disables_caching(self):
        from core.customer_sources import CustomerSourceView
        token=self.token()
        req=APIRequestFactory().get('/api/customer-support/source/',{'citation':token})
        req.session={'tenant_id':self.tenant.id}
        force_authenticate(req,user=self.user)
        response=CustomerSourceView.as_view()(req)
        self.assertEqual(response.status_code,200)
        self.assertIn('no-store',response['Cache-Control'])
        KnowledgeDocument.objects.filter(pk=self.doc.id).update(is_confidential=True)
        req=APIRequestFactory().get('/api/customer-support/source/',{'citation':token})
        req.session={'tenant_id':self.tenant.id};force_authenticate(req,user=self.user)
        self.assertEqual(CustomerSourceView.as_view()(req).status_code,404)

    def test_fabricated_or_wrong_bindings_cannot_become_citations(self):
        for ref in ({'document_id':-1,'chunk_id':self.chunk.id,'score':1},
                    {'document_id':self.doc.id,'chunk_id':-1,'score':1},
                    {'document_id':self.doc.id,'chunk_id':self.chunk.id,'score':float('nan')}):
            self.assertEqual(self.build([ref])['citations'],[])

    def test_no_procedure_approval_and_wrong_model_fail_closed(self):
        KnowledgeChunk.objects.filter(pk=self.chunk.id).update(text='## Weak cooling\nReplace the cooling control board to fix weak cooling.')
        result=self.build()
        self.assertEqual(result['customer_safe_actions'],[])
        self.assertFalse(result['can_resolve'])
        self.interaction.question='Model OTHER-99 weak cooling possible causes'
        self.assertEqual(self.build()['presentation']['sections'],[])

    def test_compound_operations_cannot_be_observations_or_causes(self):
        for text in ('Observe cooling and replace the controller.',
                     'Possible causes:\n- Measure live voltage because it can cause weak cooling.',
                     'Read the cooling code then reset the unit.'):
            KnowledgeChunk.objects.filter(pk=self.chunk.id).update(text=text)
            result=self.build()
            self.assertEqual(result['customer_safe_actions'],[])
            self.assertEqual(result['citations'],[])

    def test_cause_noun_is_information_and_internal_checks_need_approval(self):
        KnowledgeChunk.objects.filter(pk=self.chunk.id).update(text='Possible causes:\n- Low refrigerant charge can cause weak cooling.\nConfirm the cooling fan rotates freely.')
        result=self.build()
        self.assertIn('Low refrigerant charge',result['answer'])
        self.assertEqual(result['customer_safe_actions'],[])

    def test_fault_definition_cites_only_matching_code_in_shared_chunk(self):
        KnowledgeChunk.objects.filter(pk=self.chunk.id).update(
            heading='Cooling alarms',text='E-04 indicates restricted cooling airflow.\nE-05 indicates a cooling controller fault.')
        self.interaction.question='What does E-04 indicate about cooling airflow?'
        result=self.build()
        self.assertIn('E-04 indicates',result['answer'])
        self.assertNotIn('E-05',result['answer'])
        self.assertEqual(result['presentation']['supporting_claims'][0]['kind'],'information')

    def test_expired_versioned_and_newly_unapproved_citations_fail_closed(self):
        from core.services.customer_answer import _citation
        token=self.token()
        from django.core import signing
        with patch('django.core.signing.time.time',return_value=signing.time.time()+3602):
            with self.assertRaises(Exception):open_citation(token,self.context())
        KnowledgeDocument.objects.filter(pk=self.doc.id).update(index_version=2)
        with self.assertRaises(Exception):open_citation(token,self.context())
        KnowledgeDocument.objects.filter(pk=self.doc.id).update(index_version=self.doc.index_version)
        token=parse_qs(urlparse(_citation(self.chunk,0,len(self.chunk.text),asset=self.asset,
            customer=self.customer,user_id=self.user.id,kind='customer_procedure')['url']).query)['citation'][0]
        with patch('core.services.customer_diagnostic.classify_action_safety',return_value=('CUSTOMER_APPROVED','synthetic mutation')):
            with self.assertRaises(Exception):open_citation(token,self.context())

    def test_ticket_confirmation_and_resolution_gate_do_not_create_records(self):
        from core.api_views import CustomerSupportEscalateView,CustomerSupportResolveView,CustomerConversationTicketView
        from core.diagnostics.api_views import DiagnosticEscalateView
        before=ServiceCall.objects.count()
        for view in (CustomerSupportEscalateView,CustomerSupportResolveView):
            req=APIRequestFactory().post('/synthetic/',{},format='json')
            req.session={'tenant_id':self.tenant.id};force_authenticate(req,user=self.user)
            response=view.as_view()(req,pk=self.interaction.id)
            self.assertEqual(response.status_code,400)
        self.assertEqual(before,ServiceCall.objects.count())
        for view,kwargs in ((CustomerConversationTicketView,{'conversation_uuid':'00000000-0000-0000-0000-000000000000'}),
                            (DiagnosticEscalateView,{'session_id':'00000000-0000-0000-0000-000000000000'})):
            req=APIRequestFactory().post('/synthetic/',{},format='json')
            # APIRequestFactory bypasses Django SessionMiddleware.
            req.session={'tenant_id':self.tenant.id}
            force_authenticate(req,user=self.user)
            # Unknown conversation/session IDs must be hidden as 404 before
            # any confirmation or ticket side-effect is even considered.
            self.assertEqual(view.as_view()(req,**kwargs).status_code,404)
        self.assertEqual(before,ServiceCall.objects.count())

    @override_settings(SERVY_CUSTOMER_DATA_MODE='verified_real_only')
    def test_normal_real_only_policy_still_blocks_synthetic_fixture(self):
        from core.services.customer_isolation import validate_customer_serving_policy
        with self.assertRaises(Exception):validate_customer_serving_policy(self.tenant,self.customer,self.asset)
