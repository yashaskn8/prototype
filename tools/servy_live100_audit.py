"""Conservative local evidence audit. Unresolved semantics never become PASS.

Exact source transcription is a mechanical finding, not a technical expert
judgment or a customer-comprehension measurement. Does not call any model/API.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import re
import sys
from urllib.parse import parse_qs,urlparse
from types import SimpleNamespace

from servy_live100_benchmark import (ROOT,read,save,private_directory,database_read_only,
                                    scope,snapshot,verify_frozen)


def normalized(text):
    text=text.replace('**','').replace('__','').replace('`','')
    for _ in range(3):
        text=re.sub(r'^\s*(?:Step\s+\d+\s*[:.\-—]?|\d+[.)]\s+|[-*•]\s+)','',text,flags=re.I)
    text=re.sub(r'^(?:Possible documented cause:|Why:|Expected result:)\s*','',text,flags=re.I)
    return ' '.join(text.replace('**','').replace('__','').replace('`','').casefold().split()).strip(' .')


def raw_claims(answer):
    section='unlabelled'
    for line in answer.splitlines():
        line=line.strip()
        if not line:continue
        if line.isupper() or (line.endswith(':') and len(line.split())<10):
            section=line
            continue
        if len(line)<15:continue
        yield line,section


def fraction(n,d):
    return {'numerator':n,'denominator':d,'percent':round(100*n/d,2) if d else None}


def audit(folder):
    from django.core import signing
    from core.services.customer_answer import SALT,source_chunk,_approved_source_action,_noninvasive_observation
    from core.services.customer_diagnostic import classify_action_safety
    from core.services.llm import check_evidence_sufficiency
    from django.conf import settings
    frozen=verify_frozen(folder)
    tenant,asset=scope()
    corpus=read(folder/'authorized_source_snapshot.json')
    if snapshot(tenant,asset)!=corpus:raise RuntimeError('Source snapshot changed; baseline audit cannot be mixed.')
    sources={(c['document_id'],c['chunk_id']):c for c in corpus}
    questions=read(folder/'servy_100_hard_questions.json')
    refs={r['id']:r for r in read(folder/'servy_100_reference_answers.json')}
    actual={a['id']:a for a in read(folder/'servy_100_actual_answers.json')}
    if set(actual)!={q['id'] for q in questions}:raise RuntimeError('Audit requires all 100 distinct query IDs, not a partial sample.')
    rows=[];cases=[];retrieval=[];counts=Counter();point_counts=Counter()
    for q in questions:
        identifier=q['id'];a=actual[identifier];reference=refs[identifier]
        raw=a.get('raw') or {};formatted=a.get('formatted') or {};presentation=formatted.get('presentation') or {}
        rawrefs=raw.get('references') or []
        source_pairs={(r.get('document_id'),r.get('chunk_id')) for r in rawrefs}
        valid_pairs=source_pairs & sources.keys()
        if source_pairs-valid_pairs:counts['out_of_scope_or_missing_reference']+=len(source_pairs-valid_pairs)
        candidates=raw.get('retrieved_candidates') or []
        replay=check_evidence_sufficiency(q['question'],asset,[SimpleNamespace(**item) for item in raw.get('retrieved',[])])
        diagnostics={'returned_candidates':len(candidates),
            'source_verified':sum(bool(c.get('source_verified')) for c in candidates),
            'equipment_applicable':sum(bool(c.get('equipment_applicable')) for c in candidates),
            'evidence_supported':sum(bool(c.get('evidence_support')) for c in candidates),
            'replayed_evidence_sufficiency':bool(replay),'replayed_grounding_reason':replay.reason,
            'note':'Deterministic replay over captured result objects; no additional model query.'}
        target_docs=set(reference['correct_document_ids']);target_chunks=set(reference['correct_chunk_ids'])
        if reference['expected_answerability']=='source_supported_with_limits':
            first=candidates[0].get('document_id') if candidates else None
            top5={c.get('chunk_id') for c in candidates[:5]}
            retrieval.append({'id':identifier,'document_hit1':first in target_docs,
                              'chunk_hit5':bool(top5&target_chunks),'gold_status':'PROVISIONAL_SOURCE_DERIVED'})
        if raw.get('engine')=='insufficient-evidence':counts['raw_insufficient_evidence']+=1
        if raw.get('engine')=='policy-guard':counts['raw_policy_guard']+=1
        if a['status']=='ERROR':counts['query_errors']+=1
        case_rows=[];warnings=[]
        if raw.get('engine') not in ('insufficient-evidence','policy-guard'):
            for index,(text,section) in enumerate(raw_claims(raw.get('answer','')),1):
                n=normalized(text)
                matches=[sources[p] for p in valid_pairs if n and n in normalized(sources[p]['text'])]
                source=matches[0] if matches else None
                level,_=classify_action_safety(text,asset)
                imperative=bool(re.match(r'(?:open|remove|replace|disconnect|measure|probe|clean|adjust|bypass|charge|inspect|check|confirm|restart|reset|rinse|flush)\b',n))
                technician_context=bool(re.search(r'technician.only|technician.required|do not perform',section,re.I))
                potential_unsafe=imperative and not technician_context and level not in ('INFORMATIONAL_OBSERVATION','CUSTOMER_APPROVED')
                if potential_unsafe:warnings.append(f'Raw candidate action {index}: approval/technician interpretation requires review ({level}).')
                case_rows.append({'question_id':identifier,'view':'raw','point':'unstructured','claim_id':f'raw-{index}',
                    'claim_text':text,'document_id':source['document_id'] if source else '',
                    'chunk_id':source['chunk_id'] if source else '',
                    'source_passage':next((line.strip() for line in source['text'].splitlines() if n in normalized(line)),source['text']) if source else '',
                    'source_in_scope':bool(source),'exact_source_text_match':bool(source),
                    'citation_binding':'global references only; no claim-bound citation ID',
                    'classification':'Requires expert review','mechanical_finding':'Exact source transcription' if source else 'No exact transcription match; paraphrase/unsupported status requires review',
                    'customer_action_classification':level,'potential_safety_concern':potential_unsafe})
        citation_map={c['id']:c for c in presentation.get('citations',[]) if isinstance(c,dict) and 'id' in c}
        citation_findings={};critical=[]
        for cid,citation in citation_map.items():
            finding={'valid':False,'excerpt':'','document_id':citation.get('document_id'),'chunk_id':citation.get('chunk_id')}
            try:
                token=parse_qs(urlparse(citation['url']).query)['citation'][0]
                payload=signing.loads(token,salt=SALT,max_age=3600)
                if (payload['tenant']!=tenant.id or payload['customer']!=asset.customer_id or payload['asset']!=asset.id
                    or payload['document']!=citation['document_id'] or payload['chunk']!=citation['chunk_id']):raise ValueError('Wrong scope')
                chunk=source_chunk(tenant=tenant,customer=asset.customer,asset=asset,
                                   document_id=payload['document'],chunk_id=payload['chunk'])
                if chunk.document.index_version!=payload['version'] or hashlib.sha256(chunk.text.encode()).hexdigest()!=payload['sha']:
                    raise ValueError('Changed source')
                if not (0<=payload['start']<payload['end']<=len(chunk.text)):raise ValueError('Invalid span')
                finding.update(valid=True,excerpt=chunk.text[payload['start']:payload['end']],chunk=chunk)
            except Exception as exc:
                finding['failure_type']=type(exc).__name__;critical.append('Invalid formatted citation '+cid)
            citation_findings[cid]=finding
        sections=presentation.get('sections') or []
        normal_expected=reference['expected_answerability']=='source_supported_with_limits'
        special_expected=reference['expected_answerability'] in ('unsupported','mixed')
        points=[]
        for number in range(1,5):
            section=next((s for s in sections if s.get('number')==number),None)
            applicable=bool(sections) or normal_expected
            point={'point':number,'applicable':applicable,'score_0_to_5':None,'status':'NEEDS EXPERT REVIEW',
                   'score_basis':'Automated structural/presentation finding only; no expert technical judgment is implied.',
                   'citation_verification':'NO MATERIAL CLAIMS','technical_accuracy':'NEEDS EXPERT REVIEW',
                   'requirements_review':reference['four_point_rubric'][str(number)],'findings':[]}
            if not applicable:
                point.update(status='NOT APPLICABLE — special-case response',findings=['Review actual abstention/policy/hazard behavior separately; no four sections fabricated.'])
            elif not section:
                point.update(score_0_to_5=0,status='MISSING — conditional on provisional normal-answer applicability',
                             findings=['This point is absent. Whether refusal was appropriate requires source/reference review.'])
            else:
                technical=[c for c in section.get('claims',[]) if c.get('kind') not in ('uncertainty','next_step')]
                point['citation_verification']='MECHANICAL SOURCE LINKS VERIFIED; semantic relevance requires review' if technical else 'NO MATERIAL CLAIMS'
                for claim in technical:
                    links=[citation_findings.get(cid,{}) for cid in claim.get('evidence_ids',[])]
                    valid=[f for f in links if f.get('valid')]
                    exact=[f for f in valid if normalized(claim.get('source_text',''))==normalized(f['excerpt'])]
                    level,_=classify_action_safety(claim['text'],asset)
                    unauthorized=False
                    if claim.get('kind')=='customer_procedure':
                        unauthorized=not any(_approved_source_action(claim.get('source_text',''),f['chunk'],asset,tenant) for f in exact)
                    elif claim.get('kind')=='observation':
                        unauthorized=not _noninvasive_observation(claim.get('source_text','')) or level in ('PROHIBITED','TECHNICIAN_ONLY')
                    if unauthorized:critical.append(f'Unauthorized formatted action {claim.get("id")}');point.update(score_0_to_5=0,status='CRITICAL FAILURE')
                    if not exact:point['findings'].append('Material claim has no verified exact supporting span; review paraphrase versus unsupported claim.')
                    finding=exact[0] if exact else valid[0] if valid else None
                    case_rows.append({'question_id':identifier,'view':'formatted','point':number,'claim_id':claim.get('id',''),
                        'claim_text':claim['text'],'document_id':finding['document_id'] if finding else '',
                        'chunk_id':finding['chunk_id'] if finding else '',
                        'source_passage':finding['excerpt'] if finding else '',
                        'source_in_scope':bool(valid),'exact_source_text_match':bool(exact),
                        'citation_binding':'claim evidence_ids → signed exact span' if links else 'MISSING',
                        'classification':'Unsupported' if unauthorized or not valid else 'Requires expert review',
                        'mechanical_finding':'Exact authorized source span; relevance, omitted context and completeness require independent review' if exact else 'Source support not mechanically established',
                        'customer_action_classification':level,'potential_safety_concern':unauthorized})
                if number==3 and section.get('claims') and all(c.get('kind')=='uncertainty' for c in section['claims']):
                    text=' '.join(c['text'] for c in section['claims'])
                    if 'cause and equipment condition remain unverified' in text:
                        point.update(score_0_to_5=3,status='NOT PERFECT',findings=['Generic uncertainty statement does not identify the question-specific unresolved distinctions or evidence gaps.'])
                if number==4:
                    text=' '.join(c['text'] for c in section.get('claims',[]))
                    if 'If the issue continues or a check is outside your approved scope' in text:
                        point.update(score_0_to_5=None,status='NEEDS EXPERT REVIEW',findings=[
                            'Generic conditional technician advice is present. Its situation-specific appropriateness is not adjudicated; generic policy advice is not automatically an incorrect technical recommendation.'])
                if number in (1,2) and not point['findings']:
                    point['findings'].append('Source bindings can be verified locally, but full factual relevance, omissions and customer comprehension are not independently adjudicated.')
            if point['score_0_to_5']==5:point_counts[f'point{number}_perfect']+=1
            if applicable:point_counts[f'point{number}_applicable']+=1
            points.append(point)
        if normal_expected and not sections:counts['formatted_normal_expected_but_exception']+=1
        if sections:counts['formatted_four_sections']+=1
        if normal_expected and len(reference['correct_chunk_ids'])>1:
            cited={c.get('chunk_id') for c in presentation.get('citations',[])}
            if not set(reference['correct_chunk_ids']).issubset(cited):counts['multiple_source_targets_not_all_cited']+=1
        requirement_results=[]
        for req in reference['requirements']:
            requirement_results.append({**req,'assessment':'NEEDS EXPERT REVIEW','correctly_addressed':None,
                'evidence_to_review':[r['claim_id'] for r in case_rows if r['view']=='formatted'],
                'reason':'Exact quotations/keyword overlap cannot establish complete semantic satisfaction of this requirement.'})
        special={'expected':special_expected,'actual_response_type':presentation.get('response_type'),
                 'actual_raw_engine':raw.get('engine'),'verdict':'NEEDS EXPERT REVIEW'}
        if reference['expected_answerability']=='unsupported' and raw.get('engine') in ('insufficient-evidence','policy-guard'):
            counts['unsupported_refusal_candidates']+=1
        verdict='FAIL' if critical else 'PARTIAL — NOT FULL PASS' if any(p['status']=='NOT PERFECT' for p in points) else 'HUMAN REVIEW'
        root=('source/citation or customer-action validation' if critical else
              'formatting loses question-specific uncertainty detail; next-step appropriateness remains unadjudicated' if verdict.startswith('PARTIAL') else
              'raw generator abstained; normal sections absent; whether evidence gating was justified requires source review' if normal_expected and not sections and raw.get('engine')=='insufficient-evidence' else
              'unresolved; source/reference and refusal adjudication required')
        if normal_expected and not sections and raw.get('engine')=='insufficient-evidence':
            root+='; replayed grounding reason: '+str(replay.reason)
        cases.append({'id':identifier,'question':q,'reference':reference,'execution_status':a['status'],
            'actual_raw_answer':raw.get('answer',''),'actual_formatted_answer':formatted.get('answer',''),
            'raw_engine':raw.get('engine'),'retrieval_backend':raw.get('retrieval_backend'),
            'reranker_applied':raw.get('reranker_applied'),'reranker_model':raw.get('reranker_model'),
            'points':points,'requirements':requirement_results,'requirement_coverage':{'correct':None,'applicable':len(requirement_results),'status':'NOT ADJUDICATED'},
            'raw_six_dimensions':{k:'NEEDS EXPERT REVIEW' for k in ('completeness','technical_accuracy','citation_accuracy','grounding','safety','understandability')},
            'special_case_review':special,'critical_findings':critical,'raw_safety_review_flags':warnings,
            'claim_audit':case_rows,'verdict':verdict,'root_cause':root,
            'expert_review_required':True})
        cases[-1]['diagnostics']=diagnostics
        rows.extend(case_rows)
    summary={'questions_executed':len(actual),'execution_errors':counts['query_errors'],
        'raw_engines':dict(Counter((a.get('raw') or {}).get('engine','ERROR') for a in actual.values())),
        'retrieval_backends':dict(Counter((a.get('raw') or {}).get('retrieval_backend','ERROR') for a in actual.values())),
        'reranker_applied':sum(bool((a.get('raw') or {}).get('reranker_applied')) for a in actual.values()),
        'counts':dict(counts),'points':dict(point_counts),'verdicts':dict(Counter(c['verdict'] for c in cases)),
        'certified_full_passes':0,'certified_accuracy_percent':None,'human_review_cases':len(cases),
        'requirement_coverage':{'correct':None,'applicable':sum(len(c['requirements']) for c in cases),'status':'NEEDS EXPERT REVIEW'},
        'provisional_document_hit1':fraction(sum(r['document_hit1'] for r in retrieval),len(retrieval)),
        'provisional_exact_chunk_hit5':fraction(sum(r['chunk_hit5'] for r in retrieval),len(retrieval)),
        'formatted_exact_source_span_matches':fraction(sum(r['exact_source_text_match'] for r in rows if r['view']=='formatted'),sum(r['view']=='formatted' for r in rows)),
        'raw_exact_source_transcriptions':fraction(sum(r['exact_source_text_match'] for r in rows if r['view']=='raw'),sum(r['view']=='raw' for r in rows)),
        'claim_citation_precision':{'numerator':None,'denominator':None,'status':'NOT SEMANTICALLY ADJUDICATED'},
        'unsupported_claim_rate':{'numerator':None,'denominator':None,'candidate_units_to_review':len(rows),'status':'NEEDS EXPERT REVIEW; raw unit segmentation is provisional'},
        'formatted_source_validation_failures':sum(len(c['critical_findings']) for c in cases),
        'raw_potential_safety_flags':sum(len(c['raw_safety_review_flags']) for c in cases),
        'formatter_errors':sum(a.get('formatter_status')=='ERROR' for a in actual.values()),
        'live_http_portal_tested':False,'expert_or_human_participants':0,
        'limitations':['Reference quotations and targets are provisional, not expert-approved ground truth.',
            'Scores are assigned only for observable missing/generic sections or critical validation failures; unresolved scores are null, not 5.',
            'Exact transcription/span matching does not prove technical relevance, completeness or customer understanding.',
            'Formatter used real company evidence and an UNSAVED interaction; no authenticated customer source-viewer session was impersonated.',
            'No count of confirmed correct/incorrect refusals or semantic citation precision is claimed without review.']}
    summary['point_score_distributions']={str(i):dict(Counter(
        'NOT_APPLICABLE' if not c['points'][i-1]['applicable'] else 'NEEDS_EXPERT_REVIEW' if c['points'][i-1]['score_0_to_5'] is None
        else str(c['points'][i-1]['score_0_to_5']) for c in cases)) for i in range(1,5)}
    summary['perfect_point_scores_certified']={str(i):fraction(0,sum(c['points'][i-1]['applicable'] for c in cases)) for i in range(1,5)}
    summary['formatted_claim_citation_binding_coverage']=fraction(
        sum(r['citation_binding']=='claim evidence_ids → signed exact span' for r in rows if r['view']=='formatted'),
        sum(r['view']=='formatted' for r in rows))
    summary['confirmed_factually_incorrect_answers']=None
    summary['confirmed_correct_abstentions']=None
    summary['confirmed_incorrect_refusals']=None
    summary['raw_ranking_note']='Hit metrics use final returned rankings after reranking and provisional frozen source targets, not expert-approved independent accuracy.'
    summary['returned_embedding_model_versions']=dict(Counter(c.get('model_version') for a in actual.values() for c in (a.get('raw') or {}).get('retrieved',[])))
    summary['replayed_abstention_reasons']=dict(Counter(c['diagnostics']['replayed_grounding_reason'] for c in cases if c['raw_engine']=='insufficient-evidence'))
    save(folder/'servy_100_four_point_audit.json',cases)
    save(folder/'servy_100_scorecard.json',summary)
    fieldnames=['question_id','view','point','claim_id','claim_text','document_id','chunk_id','source_passage',
        'source_in_scope','exact_source_text_match','citation_binding','classification','mechanical_finding',
        'customer_action_classification','potential_safety_concern']
    with (folder/'servy_100_claim_citation_audit.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
    report=['# Servy: actual 100-question live local baseline','',
        'PRIVATE — authorized company evaluation; outside Git and OneDrive. No company data writes.',
        '', '## Scope and interpretation', '',
        'Customer 13 / Asset 60 / SKY-AST-00002 only. Direct ask_rag(top_k=5, staff_mode=False, include_confidential=False).',
        'References were source-derived and SHA-256-frozen before inference. They remain provisional. No semantic or human validation is invented.',
        'The formatted result is the actual current backend contract, invoked with an unsaved interaction, not a mocked answer or HTTP customer session.',
        '', '## Measured scorecard','', '```json',json.dumps(summary,indent=2),'```',
        '', 'Perfect-score and full-pass counts are zero CERTIFIED, not a measured 0% technical accuracy. Unadjudicated metrics remain null.',
        'Mechanical span/transcription checks and provisional retrieval hits must not be reported as answer or citation accuracy.',
        '', '## Every question: complete output, frozen reference and four-point review','']
    for c in cases:
        report += [f'### {c["id"]}', '', '**Question:** '+c['question']['question'], '',
            '**Frozen reference quotations and requirements:**','```json',json.dumps(c['reference'],indent=2,ensure_ascii=False),'```','',
            f'**Actual engine:** {c["raw_engine"]}; backend: {c["retrieval_backend"]}; V5 applied: {c["reranker_applied"]}; model: {c["reranker_model"]}.',
            '', '**Full actual raw answer:**','```text',c['actual_raw_answer'],'```',
            '', '**Full actual formatted answer:**','```text',c['actual_formatted_answer'],'```',
            '', '**Independent section checks (unresolved scores are not passes):**','```json',json.dumps(c['points'],indent=2),'```',
            '', '**Completeness:** '+json.dumps(c['requirement_coverage']),
            '**Six raw-answer dimensions:** '+json.dumps(c['raw_six_dimensions']),
            '**Citation/claim audit:** see CSV rows and JSON claim_audit for exact source passages and validation findings.',
            '**Safety:** '+('CRITICAL FINDING: '+str(c['critical_findings']) if c['critical_findings'] else 'No formatted scope/procedure-validation failure detected; full technical safety requires review.'),
            '**Raw action review flags:** '+str(c['raw_safety_review_flags']),
            '**Captured/replayed evidence diagnostics:** '+json.dumps(c['diagnostics']),
            '**Verdict:** '+c['verdict'], '**Root cause:** '+c['root_cause'],'']
    (folder/'servy_100_full_accuracy_report.md').write_text('\n'.join(report),encoding='utf-8')
    worst=sorted(cases,key=lambda c:(-len(c['critical_findings']),-sum(p['score_0_to_5']==0 for p in c['points']),-sum(p['status']=='NOT PERFECT' for p in c['points']),c['id']))[:10]
    failure=['# Failure analysis — baseline only; no corrections implemented','',
        'Ranking is based on observable section omissions/validation findings. It is not a ranking of certified technical accuracy.',
        '', '## Ranked weaknesses', '',
        '1. High abstention rate on compound questions: inspect the captured candidate/evidence indicators before changing any threshold.',
        '2. Generic uncertainty text loses question-specific limits. Generic next-step appropriateness needs individual review, not an automatic failure.',
        '3. Mandatory multi-section synthesis needs review beyond topically related retrieval or a two-claim summary.',
        '4. Raw physical instructions need audience/approval review even when transcribed from an authentic manual.',
        '5. Technical relevance, correct refusals and comprehension require independent adjudication; lexical matching is insufficient.',
        '', '## Ten worst by observable presentation deficits','']
    sanitized=[]
    for c in worst:
        failed=[p['point'] for p in c['points'] if p['score_0_to_5'] is not None and p['score_0_to_5']<5]
        safe_excerpt=(c['actual_formatted_answer'] if not (c['actual_formatted_answer'].startswith('1.') or c['actual_formatted_answer'].startswith('1. ')) else 'These documented possibilities are not a confirmed diagnosis. The cause and equipment condition remain unverified.')[:250]
        sanitized.append({'id':c['id'],'points_below_perfect_or_missing':failed,'reason':c['root_cause'],
                          'sanitized_actual_excerpt':safe_excerpt,'expert_review_required':True,'diagnostics':c['diagnostics']})
        failure += [f'### {c["id"]}',f'Points below perfect or missing: {failed}.',
                    'Actual full answers and requirements are in the private full report.',
                    'Reason: '+c['root_cause'],'Sanitized actual excerpt: '+safe_excerpt,'']
    save(folder/'servy_100_worst_sanitized.json',sanitized)
    (folder/'servy_100_failure_analysis.md').write_text('\n'.join(failure),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=__import__('pathlib').Path,required=True)
    args=parser.parse_args();folder=private_directory(args.data_dir)
    import django
    django.setup()
    with database_read_only():audit(folder)
