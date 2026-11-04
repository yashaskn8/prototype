import React, { useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Bot } from 'lucide-react';
import { CustomerAnswer } from '../src/components/CustomerAnswer';
import { answerTitles } from '../src/components/customerAnswerContract';
import '../src/index.css';
import '../src/pages/RealDemoPage.css';

const fixture = () => ({schema_version:'customer-answer-v1',response_type:'DIAGNOSTIC_GUIDANCE',equipment:{id:1},hazard:{severity:'none'},
  citations:[{id:'s1',title:'Synthetic cooling guide',heading:'Possible causes',url:'/api/customer-support/source/?citation=synthetic'}],
  sections:answerTitles.map((title,i)=>({number:i+1,title,claims:[{
    id:`c${i}`,text:i===0?'Possible documented cause: restricted airflow.':i===1?'No physical procedure is approved. Report what you observed.':i===2?'This is not a confirmed diagnosis.':'Ask a technician if the issue continues. No service ticket has been created.',
    kind:i===0?'possible_cause':'uncertainty',evidence_ids:i===0?['s1']:[],
  }]}))});
export function Preview() {
  const [value,setValue]=useState(fixture);const [fail,setFail]=useState(false);const [requests,setRequests]=useState(0);
  const [slow,setSlow]=useState(false);const [assetId,setAssetId]=useState(1);const pending=useRef(null);
  async function loadSource() {
    setRequests(n=>n+1);
    if(slow)await new Promise(resolve=>{pending.current=resolve;});
    if(fail)throw Error('Synthetic network failure');
    return {title:'Synthetic cooling guide',heading:'Possible causes',excerpt:'Restricted airflow may cause weak cooling. <script>window.syntheticInjection=true</script>'};
  }
  return <main style={{maxWidth:1200,margin:'auto',padding:24}}>
    <h1>Synthetic answer regression fixture</h1><p>No company data or operational API calls.</p>
    <button onClick={()=>{setValue(fixture());setAssetId(1);setFail(false);}}>Reset answer</button>{' '}
    <button onClick={()=>setFail(f=>!f)}>Toggle source failure</button>{' '}
    <button onClick={()=>setValue({...fixture(),response_type:'ESCALATION_REQUIRED',sections:[],message:'Synthetic urgent response',hazard:{severity:'immediate',message:'Synthetic immediate hazard warning.'}})}>Show hazard fixture</button>{' '}
    <button onClick={()=>setValue({...fixture(),response_type:'INSUFFICIENT_EVIDENCE',sections:[],message:'No sufficient evidence. Ask a technician.',hazard:{severity:'none'}})}>Show missing evidence</button>
    <button onClick={()=>setSlow(v=>!v)}>Toggle delayed source</button>{' '}
    <button onClick={()=>setAssetId(2)}>Switch equipment</button>{' '}
    <button onClick={()=>pending.current?.()}>Complete delayed source</button>
    <p>Source requests: {requests}</p>
    <section className="card real-demo-result"><div className="real-demo-result-header">
      <h2><span className="real-demo-result-icon"><Bot size={20}/></span> Servy answer</h2>
      <div className={`real-demo-result-status ${value.response_type === 'DIAGNOSTIC_GUIDANCE' ? 'is-guidance' : 'is-limited'}`}>
        {value.response_type.replaceAll('_',' ')}</div></div>
      <CustomerAnswer presentation={value} assetId={assetId} loadSource={loadSource}/>
    </section>
  </main>;
}
createRoot(document.getElementById('root')).render(<Preview/>);
