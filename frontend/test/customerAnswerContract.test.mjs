import test from 'node:test';
import assert from 'node:assert/strict';
import { answerTitles, validateCustomerPresentation } from '../src/components/customerAnswerContract.js';

export function fixture() {
  return { schema_version: 'customer-answer-v1', response_type: 'DIAGNOSTIC_GUIDANCE',
    equipment: { id: 1 }, hazard: { severity: 'none', message: '' },
    citations: [{ id: 'a', title: 'Synthetic cooling guide', heading: 'Possible causes', url: '/api/customer-support/source/?citation=synthetic' }],
    sections: answerTitles.map((title, i) => ({ number: i+1, title, claims: [{
      text: i === 0 ? 'Possible documented cause: restricted airflow.' : 'The cause remains unverified.',
      kind: i === 0 ? 'possible_cause' : 'uncertainty', evidence_ids: i === 0 ? ['a'] : [],
    }] })),
  };
}
test('four sections with bound claims are accepted', () => assert.equal(validateCustomerPresentation(fixture()).sections.length, 4));
test('signed local read-only demo source path is accepted', () => {
  const p=fixture();p.citations[0].url='/demo-api/source/?citation=synthetic';
  assert.equal(validateCustomerPresentation(p).citations.length,1);
});
test('unrelated source padding removed', () => {
  const p=fixture();p.citations.push({id:'unused',title:'Unrelated purpose',url:'/api/customer-support/source/?citation=unused'});
  assert.equal(validateCustomerPresentation(p).citations.length,1);
});
test('invented citations, arbitrary links and duplicate IDs rejected', () => {
  for(const url of ['javascript:alert(1)','https://example.invalid/source','//example.invalid','/api/assets/1/']) {
    const p=fixture();p.citations[0].url=url;assert.equal(validateCustomerPresentation(p),null);
  }
  const p=fixture();p.citations.push(p.citations[0]);assert.equal(validateCustomerPresentation(p),null);
});
test('missing malformed oversized and misnumbered answers rejected', () => {
  for(const change of [p=>p.sections.pop(),p=>p.sections[0].number=4,p=>p.sections[0].claims[0].text='x'.repeat(1001),
    p=>p.sections[0].claims[0].evidence_ids=[],p=>p.sections[0].claims[0].kind='verified_diagnosis',
    p=>p.sections[0]=null,p=>p.sections[0].claims=[null],p=>p.citations=[null]]) {
    const p=fixture();change(p);assert.equal(validateCustomerPresentation(p),null);
  }
  assert.equal(validateCustomerPresentation(null),null);
});
test('urgent warning independent of ordinary formatting', () => {
  const p=fixture();p.response_type='ESCALATION_REQUIRED';p.hazard={severity:'immediate',message:'Synthetic verified hazard'};
  assert.equal(validateCustomerPresentation(p).hazard.message,p.hazard.message);
  p.hazard.severity='none';assert.equal(validateCustomerPresentation(p).hazard,null);
  const broken=fixture();broken.sections=null;broken.hazard={severity:'immediate',message:'Synthetic hazard'};
  assert.equal(validateCustomerPresentation(broken).hazard.message,'Synthetic hazard');
});
test('a refusal links only its matching signed safety warning', () => {
  const p=fixture();p.response_type='INSUFFICIENT_EVIDENCE';p.sections=[];
  p.message='I cannot approve that operation. The source warns: Stop unsafe operation.';
  p.supporting_claims=[{id:'c1',kind:'safety_warning',text:'Stop unsafe operation.',evidence_ids:['a']}];
  assert.equal(validateCustomerPresentation(p).citations.length,1);
  p.message='I cannot approve that operation.';
  assert.equal(validateCustomerPresentation(p).citations.length,0);
  p.message='The source warns: Stop unsafe operation.';
  p.citations[0].url='https://example.invalid/source';
  assert.equal(validateCustomerPresentation(p).citations.length,0);
});

test('source-backed document facts keep the backend heading and validated citation kind', () => {
  const p = fixture();
  p.sections[0].title = 'What the document says';
  p.sections[0].claims[0].kind = 'document_fact';
  assert.equal(validateCustomerPresentation(p)?.sections.length, 4);
  p.sections[0].title = 'Arbitrary heading';
  assert.equal(validateCustomerPresentation(p), null);
});
