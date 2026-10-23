"""Plan-opening acceptance on isolated real Mem0; no cloud calls."""
import json
from datetime import timedelta
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from memory_backend import MemoryService
from chat_engine import ChatEngine
from companion_tools import now
from config import CURRENT_CONFIG
from test_memory import IO

checks={}
with tempfile.TemporaryDirectory(prefix='greeting-check-',dir=ROOT/'runtime') as directory:
    m=MemoryService(directory);m.ready.result(60)
    today=now().date()
    try:
        for age in [-1,0,1,7,8]:
            m.submit('clear').result(20)
            m.submit('managed',kind='plan',key='test',text='项目答辩',
                     value=(today-timedelta(days=age)).isoformat()).result(20)
            actual=m.submit('followup',today=today.isoformat()).result(20)
            checks[f'age_{age}_days']=(actual=='项目答辩') if age in [1,7] else actual is None
            if age in [1,7]:
                checks[f'age_{age}_once']=m.submit('followup',today=today.isoformat()).result(20) is None
        m.submit('clear').result(20)
        m.submit('managed',kind='plan',key='restart',text='项目答辩',value=(today-timedelta(days=1)).isoformat()).result(20)
    finally:m.close()
    def greet():
        with patch.dict(CURRENT_CONFIG,{'memory_dir':directory,'long_term_memory':True,'agent_backend':'native'}):
            engine=ChatEngine(IO())
            try:
                engine.memory.ready.result(60)
                return engine.build_greeting()
            finally:engine.close()
    first,used=greet();checks['restart_plan_opening']=used and '项目答辩' in first
    second,used=greet();checks['restart_not_repeated']=not used and '项目答辩' not in second
    m=MemoryService(directory);m.ready.result(60)
    try:
        m.submit('managed',kind='plan',key='delete',text='搬家',value=(today-timedelta(days=1)).isoformat()).result(20)
        rows=m.submit('list').result(20)['results']
        identity=next(r['id'] for r in rows if (r.get('metadata') or {}).get('key')=='delete')
        m.submit('delete',memory_id=identity).result(20)
    finally:m.close()
    deleted,used=greet();checks['deleted_plan_not_recalled']=not used and '搬家' not in deleted
out={'checks':checks,'passed':sum(checks.values()),'total':len(checks),'greetings':[first,second,deleted],'cloud_calls':0}
(ROOT/'experiments/memory_greeting_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
assert all(checks.values())
