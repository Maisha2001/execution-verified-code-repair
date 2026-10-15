import './style.css';

type Evidence = {status:string; seconds?:number; evidence?:{passed:number;failed:number;errors:number;skipped:number;output?:string}};
type Attempt = {number:number;response?:string;source?:string;rejection?:string;visible_passed:boolean;run?:Evidence;feedback?:unknown;edit_size?:{token_edits:number;changed_lines:number}};
type Candidate = {number:number;patch:string;source:string};
type Result = {decision:string;eligibility:string;buggy?:Evidence;reference?:Evidence;regression?:Evidence;regression_reference?:Evidence;attempts:Attempt[];ranked_attempts:number[];selected_candidate?:Candidate;accepted_candidate?:Candidate};
type Job = {id:string;task:string;status:string;original_source:string;seconds?:number;error?:string;events:{phase:string;test?:string;evidence?:Evidence;response?:string}[];result?:Result;request:{budget:number;feedback:boolean;seconds:number}};
type Config = {token:string;tasks:string[];identity:{adapter?:{checkpoint:string}};notice:string};
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `<header><div class="brand">REPAIR / LAB</div><span class="local">● Local session</span></header>
<main><section class="intro"><p class="eyebrow">EXECUTION-VERIFIED PYTHON</p><h1>A patch is only a proposal.<br><span>Inspect the evidence.</span></h1><p>Run a bounded repair, follow each attempt, and see which changes survive the tests.</p></section>
<div id="error" role="alert" hidden></div>
<div class="layout"><aside><form id="form"><h2>Start a repair</h2><label for="task">Development example</label><select id="task"></select><label for="budget">Attempt budget</label><select id="budget"><option>1</option><option>2</option><option selected>4</option></select><label for="seconds">Time budget</label><select id="seconds"><option value="60">1 minute</option><option value="300" selected>5 minutes</option><option value="600">10 minutes</option></select><label class="check"><input type="checkbox" id="feedback" checked> Use visible-test feedback</label><button id="start" type="submit">Run repair <span>→</span></button><button id="cancel" type="button" class="secondary" disabled>Cancel repair</button></form><p class="small">One repair at a time. Cancellation waits for the active generation or isolated test to finish.</p><p id="model" class="small"></p><h2 class="history-title">Session history</h2><div id="history"></div></aside>
<section class="workspace"><div class="status-row"><div><p class="eyebrow">WORKBENCH</p><h2 id="title">Choose an example</h2></div><span id="status" class="badge">Ready</span></div><p id="progress" role="status" aria-live="polite">Loading local service…</p><div id="summary"></div><div id="evidence"></div><details open id="source-details"><summary>Original Python module</summary><pre id="source"></pre></details><div id="attempts"></div><div id="selection"></div><a id="export" hidden>Download complete evidence (JSON) ↓</a></section></div>
<footer>Public development examples · Test passes are evidence, not proof of correctness.<br>QuixBugs, pinned upstream revision. Authored regression checks are separate from feedback.</footer></main>`;
const $ = <T extends HTMLElement>(id:string) => document.getElementById(id) as T;
const task = $<HTMLSelectElement>('task');
const start = $<HTMLButtonElement>('start'), cancel = $<HTMLButtonElement>('cancel');
let config:Config, current:string|undefined, polling=false, active:string|null=null;
const terminal = (status:string) => ['completed','cancelled','failed','interrupted'].includes(status);
const label = (value:string) => value.replaceAll('_',' ');
function error(message=''){ $('error').textContent=message; $('error').hidden=!message; }
async function api<T>(path:string, body?:unknown):Promise<T>{
 const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Repair-Token':config.token},body:JSON.stringify(body)});
 const data=await response.json();
 if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:`Request rejected (${response.status}). Check the limits and try again.`);
 return data;
}
function pre(text:string){ const node=document.createElement('pre');node.textContent=text;return node; }
function details(title:string,text:string){const node=document.createElement('details');const heading=document.createElement('summary');heading.textContent=title;node.append(heading,pre(text));return node;}
function testCard(title:string,run?:Evidence){
 const card=document.createElement('div');card.className='test-card';const name=document.createElement('strong');name.textContent=title;card.append(name);
 const status=document.createElement('span');status.textContent=run?label(run.status):'Not run';status.className=run?.status==='passed'?'pass':'neutral';card.append(status);
 if(run?.evidence){const e=run.evidence;const counts=document.createElement('p');counts.textContent=`${e.passed} passed · ${e.failed} failed · ${e.errors} errors · ${e.skipped} skipped`;card.append(counts);if(e.output)card.append(details('Test output',e.output));}
 return card;
}
function render(job:Job){
 $('title').textContent=label(job.task);$('status').textContent=label(job.status);$('status').className='badge '+(job.status==='failed'?'bad':'');
 const r=job.result;const final=terminal(job.status);const last=job.events.at(-1);
 $('progress').textContent=job.error || (final?`Finished in ${job.seconds?.toFixed(1) ?? '—'} seconds. ${r?label(r.decision)+'.':''} Budget: ${job.request.budget} attempts · ${job.request.seconds}s · feedback ${job.request.feedback?'on':'off'}.`:job.status==='cancelling'?'Cancellation requested. Waiting for the active operation to clean up.':`${last?label(last.phase):'Starting'}${last?.test?' · '+last.test:''}…`);
 cancel.disabled=final || job.status==='cancelling'; start.disabled=!!active;
 $('source').textContent=job.original_source;$('summary').replaceChildren();$('evidence').replaceChildren();$('attempts').replaceChildren();$('selection').replaceChildren();
 if(r){
  const summary=document.createElement('p');summary.className='decision';summary.textContent=r.accepted_candidate?'Patch passed both test stages. Review the changes below.':r.selected_candidate?'A patch was selected, but no repair was accepted.':'No repair was accepted.';$('summary').append(summary);
  $('evidence').append(testCard('Original failure',r.buggy),testCard('Reference audit',r.reference),testCard('Selected patch regression',r.regression));
  for(const a of r.attempts){const box=document.createElement('article');box.className='attempt';const title=document.createElement('h3');title.textContent=`Attempt ${a.number} · ${a.rejection?label(a.rejection):a.visible_passed?'Visible tests passed':'Visible tests did not pass'}`;box.append(title);
   if(a.edit_size){const metric=document.createElement('p');metric.className='small';metric.textContent=`${a.edit_size.token_edits} token edits · ${a.edit_size.changed_lines} changed lines · ${r.ranked_attempts.includes(a.number)?'rank '+(r.ranked_attempts.indexOf(a.number)+1):'not ranked'}`;box.append(metric);}
   if(a.run)box.append(testCard('Visible tests',a.run));if(a.source)box.append(details('Candidate module',a.source));if(a.response)box.append(details('Raw model response',a.response));if(a.feedback)box.append(details('Feedback supplied to this attempt',JSON.stringify(a.feedback,null,2)));$('attempts').append(box);}
  if(r.selected_candidate){const title=document.createElement('h2');title.textContent=`Selected change · attempt ${r.selected_candidate.number}`;$('selection').append(title,pre(r.selected_candidate.patch));}
 }else if(last?.evidence){$('evidence').append(testCard('Latest isolated test',last.evidence));}
 const download=$<HTMLAnchorElement>('export');download.hidden=!final;download.href=`/api/jobs/${job.id}/export`;
}
async function history(){const [jobs,health]=await Promise.all([api<{id:string;task:string;status:string}[]>('/api/jobs'),api<{active_job:string|null}>('/api/health')]);active=health.active_job;start.disabled=!!active;$('history').replaceChildren();for(const job of jobs){const button=document.createElement('button');button.className='history';button.type='button';button.textContent=`${label(job.task)} · ${label(job.status)}`;button.onclick=()=>{current=job.id;void refresh();};$('history').append(button);}return jobs;}
async function refresh(){if(polling)return;polling=true;try{await history();if(current)render(await api<Job>(`/api/jobs/${current}`));error();}catch(e){error(String(e));}finally{polling=false;}}
task.onchange=async()=>{if(current)return;try{$('source').textContent=(await api<{source:string}>(`/api/tasks/${task.value}`)).source;$('title').textContent=label(task.value);}catch(e){error(String(e));}};
$('form').onsubmit=async(event)=>{event.preventDefault();start.disabled=true;try{const job=await api<Job>('/api/jobs',{task:task.value,budget:Number($<HTMLSelectElement>('budget').value),feedback:$<HTMLInputElement>('feedback').checked,seconds:Number($<HTMLSelectElement>('seconds').value)});current=job.id;active=job.id;render(job);error();}catch(e){error(String(e));start.disabled=!!active;}};
cancel.onclick=async()=>{if(!current)return;cancel.disabled=true;try{render(await api<Job>(`/api/jobs/${current}/cancel`,{}));}catch(e){error(String(e));}};
async function init(){try{config=await api<Config>('/api/config');for(const name of config.tasks){const option=document.createElement('option');option.value=name;option.textContent=label(name);task.append(option);}task.value='gcd';$('model').textContent=config.identity.adapter?`Local Qwen with adapter ${config.identity.adapter.checkpoint}.`:'Local Qwen · base model.';const jobs=await history();current=jobs[0]?.id;if(current)await refresh();else{$('source').textContent=(await api<{source:string}>('/api/tasks/gcd')).source;$('title').textContent='gcd';$('progress').textContent='Ready. Select an attempt budget and run a repair.';}setInterval(()=>{if(active)void refresh();},1000);}catch(e){error(`Service unavailable: ${String(e)}`);start.disabled=true;}}
void init();
