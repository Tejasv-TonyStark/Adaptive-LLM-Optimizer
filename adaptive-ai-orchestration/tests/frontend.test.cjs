// Run with: node --test tests/frontend.test.cjs
// A small DOM fixture exercises the public demo's live API integration.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('Frontend/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(chatResponse) {
  const elements = new Map();
  class Element {
    constructor() {
      this.children=[]; this.style={}; this.attrs={}; this.value=''; this.disabled=false;
      this.hidden=false; this.text=''; this.className=''; this.dataset={};
      this.classList={add(){},remove(){}};
    }
    set textContent(value) { this.text=String(value); this.children=[]; }
    get textContent() { return (this.text || '') + this.children.map(c=>c.textContent).join(' '); }
    setAttribute(k,v) { this.attrs[k]=v; }
    appendChild(child) { child.parent=this; this.children.push(child); return child; }
    append(...children) { children.forEach(child=>this.appendChild(child)); }
    replaceChildren(...children) { this.children=[]; this.text=''; this.append(...children); }
    addEventListener() {} focus() {}
    get lastChild() { return this.children[this.children.length-1]; }
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) elements.set(match[1], new Element());
  const storage={values:new Map(),getItem(k){return this.values.get(k)||null;},setItem(k,v){this.values.set(k,String(v));}};
  const calls=[];
  const response=(data,status=200)=>({ok:status>=200&&status<300,status,json:async()=>data});
  const sandbox={console:{error(){}},globalThis:{},document:{
    getElementById:id=>elements.get(id)||null, createElement:()=>new Element(), querySelectorAll:()=>[]
  },localStorage:storage,setTimeout(){},fetch:async(url,options)=>{
    calls.push({url,options});
    if (url.endsWith('/api/chat')) return chatResponse(response);
    if (url.includes('/evaluation')) return response({status:'completed',quality_score:.91});
    return response({status:'ok'});
  }};
  sandbox.globalThis=sandbox;
  vm.createContext(sandbox); vm.runInContext(script,sandbox);
  elements.get('query-input').value='What is the Sankalpa remote work policy?';
  return {sandbox,elements,calls,run:code=>vm.runInContext(code,sandbox),
    answer:()=>elements.get('answer-body').textContent,
    routing:()=>elements.get('routing-list').textContent,
    state:()=>elements.get('demo-state').textContent};
}

const good={response:'**Policy:** Remote work needs manager approval. [1]',model_used:'llama3-8b',selected_model:'llama3-8b',complexity:'low',strategy_used:'rag',latency_ms:42,query_id:1,evaluation_status:'skipped',sources:[{source:'sankalpa.pdf',page:4}]};

test('portfolio page contains the educational disclaimer and real technical sections',()=>{
  for (const phrase of ['Hi, I’m Tejasv.', 'Educational AI systems project', 'The Sankalpa handbook is synthetic by design.', 'Cold-start note.', 'final = 0.7 × vector_score + 0.3 × lexical_overlap']) assert.match(html, new RegExp(phrase.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')));
  assert.match(html,/href="\/api\/handbook"/);
  assert.doesNotMatch(html,/Sign in|Create account/);
});
test('live demo sends a public chat request and shows answer, sources, and trace',async()=>{
  const p=page(r=>r(good)); await p.sandbox.sendQuery();
  assert.match(p.answer(),/Remote work needs manager approval/);
  assert.match(p.routing(),/Llama 3.1 8B/);
  assert.match(p.routing(),/42 ms/);
  assert.match(p.elements.get('source-list').textContent,/sankalpa.pdf/);
  const request=p.calls.find(call=>call.url.endsWith('/api/chat'));
  assert.equal(request.options.method,'POST');
  assert.equal(JSON.parse(request.options.body).query,'What is the Sankalpa remote work policy?');
  assert.equal(p.elements.get('send-btn').disabled,false);
});
test('answer markdown is bold-only and model text is never interpreted as HTML',()=>{
  const p=page(r=>r(good)); const element=p.run('(()=>{const e=document.createElement("div");appendBoldMarkdown(e,"**Conclusion:** <img onerror=alert(1)>");return e})()');
  assert.equal(element.children[0].textContent,'Conclusion:');
  assert.equal(element.children[1].textContent,' <img onerror=alert(1)>');
  assert.equal(element.children[1].innerHTML,undefined);
});
test('structured backend failure remains useful and unlocks the form',async()=>{
  const p=page(r=>r({detail:{message:'A required service is unavailable.'}},503)); await p.sandbox.sendQuery();
  assert.match(p.state(),/required service is unavailable/i);
  assert.equal(p.elements.get('send-btn').disabled,false);
  assert.match(p.routing(),/request failed/);
});
test('network failures are clearly distinguished from answer errors',async()=>{
  const p=page(()=>{throw new TypeError('Failed to fetch');}); await p.sandbox.sendQuery();
  assert.match(p.state(),/Cannot connect to the backend/);
  assert.equal(p.elements.get('send-btn').disabled,false);
});
test('invalid successful payloads do not render undefined answers',async()=>{
  const p=page(r=>r({response:null})); await p.sandbox.sendQuery();
  assert.match(p.state(),/returned no answer/);
  assert.doesNotMatch(p.answer(),/undefined/);
});
