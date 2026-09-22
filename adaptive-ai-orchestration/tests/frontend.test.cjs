// Run with: node --test tests/frontend.test.cjs
// Execute the entire shipped inline script, including startup, against a small DOM fixture.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('Frontend/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(chatResponse) {
  const elements = new Map();
  class Element {
    constructor() { this.children=[]; this.style={}; this.attrs={}; this.value=''; this.disabled=false; this.isConnected=true;
      this.classList={add(){},remove(){},toggle(){}}; }
    set textContent(value) { this.text=String(value); this.children=[]; }
    get textContent() { return (this.text || '')+this.children.map(c=>c.textContent).join(' '); }
    setAttribute(k,v) { this.attrs[k]=v; }
    getAttribute(k) { return this.attrs[k]; }
    appendChild(child) { child.parent=this; this.children.push(child); if(child.id)elements.set(child.id,child); }
    insertBefore(child) { child.parent=this; this.children.unshift(child); }
    remove() { this.isConnected=false; if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this); if(this.id)elements.delete(this.id); }
    replaceChildren() { this.children.forEach(c=>c.isConnected=false); this.children=[]; this.text=''; }
    querySelector() { return this.children.find(c=>Object.hasOwn(c.attrs,'data-evaluation-status')) || this.children.map(c=>c.querySelector()).find(Boolean); }
    addEventListener() {} focus() {}
    get firstChild() { return this.children[0]; }
  }
  for(const match of html.matchAll(/id="([^"]+)"/g))elements.set(match[1],new Element());
  const storage={getItem(){return null;},setItem(){},removeItem(){}};
  const calls=[];
  const response=(data,status=200)=>({ok:status>=200&&status<300,status,json:async()=>data});
  const sandbox={console:{error(){}},window:{location:{origin:'http://127.0.0.1:8000'},open(){}},
    document:{documentElement:new Element(),getElementById:id=>elements.get(id)||null,
      createElement:()=>new Element(),querySelectorAll:()=>elements.get('side-content').children},
    localStorage:storage,sessionStorage:storage,setTimeout(){},setInterval(){},
    fetch:async(url,options)=>{calls.push({url,options});
      if(url.endsWith('/api/chat'))return chatResponse(response);
      if(url.includes('/evaluation'))return response({status:'completed',quality_score:.9});
      if(url.endsWith('/api/auth/login'))return response({access_token:'test-token'});
      if(url.endsWith('/api/auth/me'))return response({username:'tester'});
      return response({status:'ok'});
    }};
  vm.createContext(sandbox);
  vm.runInContext(script,sandbox);
  vm.runInContext('authToken="test-token";',sandbox);
  elements.get('query-input').value='What is Python?';
  return {sandbox,elements,calls,response,run:code=>vm.runInContext(code,sandbox),
    messages:()=>elements.get('messages-inner').textContent,
    cards:()=>elements.get('side-content').textContent};
}
const good={response:'Python is a programming language.',model_used:'llama3-70b',selected_model:'llama3-70b',
  complexity:'high',strategy_used:'reasoning',latency_ms:42,query_id:1,evaluation_status:'skipped'};

test('full page loads and renders answer plus routing card',async()=>{
  const p=page(r=>r(good)); await p.sandbox.sendQuery();
  assert.match(p.messages(),/Python is a programming language/);
  assert.match(p.cards(),/Llama 3.3 70B/);
  assert.doesNotMatch(p.messages()+p.cards(),/not defined|Request failed|Server running/);
  assert.equal(p.elements.get('send-btn').disabled,false);
  assert.equal(p.run('isLoading'),false);
});
test('malformed optional metadata cannot break answer rendering',async()=>{
  const p=page(r=>r({...good,quality_score:'0.8',sources:{length:1},routing_reasons:'reason',latency_ms:'bad'}));
  await p.sandbox.sendQuery(); assert.match(p.cards(),/Routing Decision/); assert.doesNotMatch(p.messages(),/⚠/);
});
test('routing metadata is inserted as text, not executable markup',async()=>{
  const p=page(r=>r({...good,model_used:'<img onerror=alert(1)>',complexity:'__proto__',strategy_used:'<script>x</script>'}));
  await p.sandbox.sendQuery(); assert.match(p.cards(),/<img onerror/);
  function check(node){assert.equal(node.innerHTML,undefined);node.children.forEach(check);}
  p.elements.get('side-content').children.forEach(check);
});
test('answer headings render bold Markdown without accepting HTML',()=>{
  const p=page(r=>r(good));
  assert.equal(p.run('(()=>{const e=document.createElement("div");appendBoldMarkdown(e,"**Conclusion:** A safe answer");return e.children[0].textContent})()'),'Conclusion:');
  assert.equal(p.run('(()=>{const e=document.createElement("div");appendBoldMarkdown(e,"<img onerror=alert(1)>");return e.children[0].textContent})()'),'<img onerror=alert(1)>');
});
test('non-JSON HTTP 500 gets a useful message and unlocks send',async()=>{
  const p=page(()=>({ok:false,status:500,json:async()=>{throw new SyntaxError('Unexpected token I');}}));
  await p.sandbox.sendQuery(); assert.match(p.messages(),/HTTP 500/);
  assert.doesNotMatch(p.messages(),/Unexpected token|not defined/);
  assert.equal(p.elements.get('send-btn').disabled,false);
});
test('invalid success response does not render an undefined answer',async()=>{
  const p=page(r=>r({response:null}));await p.sandbox.sendQuery();
  assert.match(p.messages(),/returned no answer/);assert.doesNotMatch(p.messages(),/undefined/);
});
test('structured backend failure preserves its message',async()=>{
  const p=page(r=>r({detail:{message:'A required service is unavailable.'}},503));
  await p.sandbox.sendQuery();assert.match(p.messages(),/required service is unavailable/);
});
test('network failure is distinguished from rendering failure',async()=>{
  const p=page(()=>{throw new TypeError('Failed to fetch');});await p.sandbox.sendQuery();
  assert.match(p.messages(),/Cannot connect to the backend/);assert.equal(p.run('isLoading'),false);
});
test('routing display failure does not label a successful answer as failed',async()=>{
  const p=page(r=>r(good));p.run('addRoutingCard=()=>{throw new Error("fixture render error")};');
  await p.sandbox.sendQuery();assert.match(p.messages(),/Python is a programming language/);
  assert.match(p.cards(),/Answer received/);assert.doesNotMatch(p.messages()+p.cards(),/Request failed|Server running/);
  assert.equal(p.run('isLoading'),false);
});
test('expired session returns to sign-in cleanly',async()=>{
  const p=page(r=>r({detail:'Expired'},401));await p.sandbox.sendQuery();
  assert.equal(p.run('authToken'),'');assert.match(p.elements.get('auth-error').textContent,/session expired/);
  assert.equal(p.run('isLoading'),false);
});
test('a response arriving after sign-out is not displayed',async()=>{
  let resolve;const pending=new Promise(r=>resolve=r);const p=page(()=>pending);
  const sending=p.sandbox.sendQuery();p.sandbox.logoutUser();resolve(p.response(good));await sending;
  assert.doesNotMatch(p.messages(),/Python is a programming language/);assert.equal(p.run('isLoading'),false);
});
test('pending evaluation updates without missing helper errors',async()=>{
  const p=page(r=>r({...good,evaluation_status:'pending'}));await p.sandbox.sendQuery();
  await new Promise(resolve=>setImmediate(resolve));assert.match(p.cards(),/0.90/);
});
test('login runs through account loading',async()=>{
  const p=page(r=>r(good));await p.sandbox.loginUser({preventDefault(){}});
  assert.equal(p.elements.get('user-badge').textContent,'tester');assert.equal(p.elements.get('login-submit').disabled,false);
});
