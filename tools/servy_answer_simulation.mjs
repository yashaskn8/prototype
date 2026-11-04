// Static contract checks on 100,000 synthetic assignments. No server calls,
// company records, model inference, human participants, or usability claims.
import fs from 'node:fs';
import path from 'node:path';
import { answerTitles, validateCustomerPresentation } from '../frontend/src/components/customerAnswerContract.js';

const dimensions={experience:4,reading:5,question:4,device:5,evidence:5,citations:4,safety:5,authorization:5,accessibility:3,symptoms:8};
const combinations=Object.values(dimensions).reduce((total,size)=>total*size,1);
const coverage=Object.fromEntries(Object.keys(dimensions).map(k=>[k,new Set()]));
let passed=0;const failures=[];
for(let i=0;i<100000;i++) {
  let assignment=(i*7919)%combinations;const profile={id:`SIM-${String(i+1).padStart(6,'0')}`};
  for(const [key,size] of Object.entries(dimensions)){profile[key]=assignment%size;assignment=Math.floor(assignment/size);coverage[key].add(profile[key]);}
  const normal=profile.evidence>0 && profile.authorization>0;
  const data={schema_version:'customer-answer-v1',response_type:normal?'DIAGNOSTIC_GUIDANCE':'INSUFFICIENT_EVIDENCE',
    equipment:{id:1},message:normal?'':'Evidence or authorization is unavailable.',
    hazard:{severity:profile.safety===0?'immediate':'none',message:'Synthetic warning for contract validation only.'},
    citations:[{id:'s1',title:'Synthetic source',url:'/api/customer-support/source/?citation=synthetic'}],
    sections:normal?answerTitles.map((title,n)=>({number:n+1,title,claims:[{text:n===0?'Possible documented cause: airflow restriction.':'The cause remains unverified.',kind:n===0?'possible_cause':'uncertainty',evidence_ids:n===0?['s1']:[]}]})):[]};
  // Deliberate malformed contract variants exercise fail-closed client behavior.
  const malformed=normal && profile.citations===0;
  if(malformed)data.sections[0].claims[0].evidence_ids=['invented'];
  const result=validateCustomerPresentation(data);
  // A malformed ordinary answer must never suppress an independent warning.
  const ok=malformed?(profile.safety===0
    ? !!result && result.sections.length===0 && result.citations.length===0 && result.hazard?.severity==='immediate'
    : result===null):!!result && result.sections.length===(normal?4:0)
    && (profile.safety!==0 || result.hazard?.severity==='immediate');
  if(ok)passed++;else failures.push({test_id:profile.id,attack_input:profile,expected_safe_behavior:'Reject malformed evidence or preserve bounded four-section/exception contract.',observed_behavior:result,root_cause:'Static contract invariant failed',severity:'HIGH',affected_component:'customerAnswerContract',fix_implemented:'pending',regression_test_result:'FAIL'});
}
const report={kind:'static_synthetic_assignments',designed:100000,executed:100000,passed,failed:failures.length,
  coverage:Object.fromEntries(Object.entries(coverage).map(([k,v])=>[k,[...v].sort()])),failures,
  limitations:['No browser interaction in this simulation.','Device and accessibility values are assignments, not comprehension measurements.','Authorization enforcement is tested separately in Django.'],human_participants:0};
const dest=path.resolve(process.argv[2]||'private_training_data/customer_answer_validation/simulation.json');
if(!dest.split(path.sep).includes('private_training_data'))throw Error('Use private output directory');
fs.mkdirSync(path.dirname(dest),{recursive:true});fs.writeFileSync(dest,JSON.stringify(report,null,2),{flag:'wx'});
console.log(JSON.stringify({...report,failures:undefined}));
if(failures.length)process.exitCode=1;
