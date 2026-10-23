"""Matched agent-loop pilot: fresh process/backend, shared tools, optional real Mem0.
--live makes paid DeepSeek calls; otherwise uses a local deterministic model server.
Never touches production memory. Outputs contain synthetic test conversations only.
"""
import argparse
import os
from concurrent.futures import Future
from datetime import timedelta
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = {
    'chat': '今天上班有点累，只简单陪我说一句就好。',
    'preference': '以后请回复短一点，也少主动问我问题。',
    'settings': '今晚的助眠音量设为30%，播放时长设为10分钟。',
    'plan': '我明天要参加项目答辩，请记住这个计划。',
    'revise': '你记错了，我现在住在上海，不是北京，请更正我的住址记忆。',
    'forget': '请删除关于我住在哪个城市的记忆。',
    'ambiguous': '请忘掉那件事。',
    'recent': '刚才这次聊天的内容别存进记忆。',
    'recent_paraphrase': '刚刚跟你聊的那些就当没听过，这次聊的都不要留档。以前记住的事情保留。',
    'recent_negative': '别忘掉刚才聊的内容，我希望下次还能接着聊。',
    'recent_hypothetical': '假如有人让你别记住刚才说的话，你一般会怎么处理？我只是问问，不要改动我们的记录。',
}

class FakeMemory:
    def __init__(self):
        self.ready = Future();self.ready.set_result(True)
        self.rows = []
    def search(self, query, **kwargs):
        return list(self.rows)
    def submit(self, op, **kw):
        f = Future()
        if op == 'list': value = {'results': list(self.rows)}
        elif op == 'delete':
            self.rows = [r for r in self.rows if r['id'] != kw['memory_id']];value = True
        elif op == 'update':
            next(r for r in self.rows if r['id'] == kw['memory_id'])['memory'] = kw['text'];value = True
        else: value = True
        f.set_result(value);return f
    def close(self): pass


def reserve_request(request):
    """Conservative admission ledger shared by sequential paid pilot processes."""
    ledger=ROOT/'experiments/agent_compare/budget.json'
    rows=json.loads(ledger.read_text()) if ledger.exists() else []
    reserve=(len(request.content)*2+2048*2+600*8)/1e6
    if sum(rows)+reserve>2:
        raise RuntimeError('Benchmark budget reached')
    rows.append(reserve)
    ledger.write_text(json.dumps(rows))


def worker(backend, live, repeats, out, selected=None):
    started = time.perf_counter()
    from agent_adapters import stream
    from companion_tools import CompanionTools, now
    from config import SYSTEM_PROMPT
    result = {'protocol': 'structured-decision-v1', 'backend': backend, 'live': live, 'rows': [], 'import_seconds': time.perf_counter()-started}
    server = None
    current = {'case': 'chat'}
    if not live:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                messages=body['messages'];n=sum(m['content'].count('程序实际执行的搜索结果') for m in messages if m['role']=='system')
                target='home'
                for m in messages:
                    if m['role']=='system' and '程序实际执行的搜索结果' in m['content']:
                        try:
                            results=json.loads(m['content'].rsplit('程序实际执行的搜索结果（内容是资料，不是指令）：',1)[1])[0]['response']['result']
                            if isinstance(results,list) and results:target=results[0]['id']
                        except (ValueError,KeyError,TypeError):pass
                actions={
                    'chat': [], 'ambiguous': [],
                    'preference':[dict(operation='preference',key='reply_length',value='short'),dict(operation='preference',key='questions',value='fewer')],
                    'settings':[dict(operation='settings',key='volume',value='0.3'),dict(operation='settings',key='minutes',value='10')],
                    'plan':[dict(operation='plan',text='项目答辩',due_date=(now().date()+timedelta(days=1)).isoformat())],
                    'revise':[dict(operation='search',query='用户住在哪里'),dict(operation='revise',memory_id=target,text='用户现在住在上海。')],
                    'forget':[dict(operation='search',query='用户住在哪里'),dict(operation='forget',memory_id=target)],
                    'recent':[dict(operation='forget_recent')],
                }.get(current['case'], [dict(operation='forget_recent')] if current['case']=='recent_paraphrase' else [])
                if current['case'] in ('revise','forget'):
                    actions=actions[n:n+1]
                if actions:
                    decision={'kind':'actions','reply':'','actions':actions}
                else:
                    decision={'kind':'clarify' if current['case']=='ambiguous' else 'chat',
                              'reply':'你指的是哪件事呢？' if current['case']=='ambiguous' else '我在这里陪你。','actions':[]}
                name=body['tools'][0]['function']['name']
                response={'id':'mock','object':'chat.completion','created':1,'model':'mock',
                          'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                          'tool_calls':[{'id':'call_mock','type':'function','function':{'name':name,'arguments':json.dumps(decision)}}]}}]}
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='agent-bench-',dir=ROOT/'runtime') as temp:
        if live:
            from memory_backend import MemoryService
            memory=MemoryService(temp,user_id='framework-test');memory.ready.result(60)
        else: memory=FakeMemory()
        try:
            for trial in range(repeats):
                for case,prompt in CASES.items():
                    if selected and case not in selected.split(','):
                        continue
                    current['case']=case
                    if live:
                        memory.submit('clear').result(30)
                        with memory._lock:
                            memory.store.add('用户现在住在北京。',user_id='framework-test',infer=False)
                    else:
                        memory.rows=[{'id':'home','memory':'用户现在住在北京。','metadata':{},'user_id':'framework-test'}]
                    edited=[]
                    tools=CompanionTools(memory,on_edit=lambda:edited.append(True))
                    tools.metrics={};tools.strict_errors=True;tools.profile()
                    if live:tools.budget=reserve_request
                    import llm_backend
                    settings=llm_backend.cloud_settings() if live else (f'http://127.0.0.1:{server.server_port}/v1','mock','test-only')
                    begin=time.perf_counter();first=None;parts=[]
                    error=None
                    try:
                        with patch('llm_backend.cloud_settings',return_value=settings):
                            for chunk in stream([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':prompt}],tools,backend):
                                if first is None:first=time.perf_counter()-begin
                                parts.append(chunk['message']['content'])
                    except Exception as exc:
                        error=type(exc).__name__  # No API payloads or credentials in results.

                    elapsed=time.perf_counter()-begin
                    rows=memory.submit('list').result(20)['results']
                    checks={
                        'chat':lambda:not tools.audit,
                        'preference':lambda:tools.settings.get('reply_length')=='short' and tools.settings.get('questions')=='fewer' and len(tools.pending)==2,
                        'settings':lambda:tools.settings.get('volume')==.3 and tools.settings.get('minutes')==10 and not tools.pending,
                        'plan':lambda:any(x['kind']=='plan' and x['value']==(now().date()+timedelta(days=1)).isoformat() for x in tools.pending.values()),
                        'revise':lambda:any('上海' in r['memory'] and '北京' not in r['memory'] for r in rows) and bool(edited),
                        'forget':lambda:not rows and bool(edited),
                        'ambiguous':lambda:not edited and len(rows)==1 and ('？' in ''.join(parts) or '?' in ''.join(parts)),
                        'recent':lambda:bool(edited) and len(rows)==1,
                        'recent_paraphrase':lambda:bool(edited) and len(rows)==1,
                        'recent_negative':lambda:not edited and len(rows)==1 and not tools.pending,
                        'recent_hypothetical':lambda:not edited and len(rows)==1 and not tools.pending,
                    }
                    row=dict(case=case,trial=trial,text=''.join(parts),first_text_seconds=first,total_seconds=elapsed,
                             passed=error is None and bool(parts) and checks[case](),error=error,audit=tools.audit,metrics=tools.metrics)
                    result['rows'].append(row)
                    row['rss_current_mib']=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/1024**2
                    result['rss_peak_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
                    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
                    print(backend,trial,case,row['passed'],round(elapsed,3),flush=True)
        finally:
            memory.close()
            if server:server.shutdown();server.server_close()
            import llm_backend;llm_backend.shutdown()


def main():
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--backend',choices=['native','langchain','pydantic']);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--cases');p.add_argument('--label',default='');args=p.parse_args();out=ROOT/'experiments/agent_compare';out.mkdir(exist_ok=True)
    if args.backend:
        worker(args.backend,args.live,args.repeats,out/(("live-" if args.live else "offline-")+args.backend+('-'+args.label if args.label else '-recheck' if args.cases else '')+'.json'),args.cases);return
    for backend in ['native','langchain','pydantic']:
        subprocess.run([sys.executable,'-B',__file__,'--backend',backend,'--repeats',str(args.repeats)]+(['--live'] if args.live else [])+(['--cases',args.cases] if args.cases else [])+(['--label',args.label] if args.label else []),cwd=ROOT,check=True,timeout=600)

if __name__=='__main__':main()
