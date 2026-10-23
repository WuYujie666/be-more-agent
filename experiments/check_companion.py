"""Real Mem0 storage acceptance, no LLM requests; synthetic isolated records."""
import json
from pathlib import Path
import sys
import tempfile
from datetime import timedelta
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory_backend import MemoryService
from companion_tools import CompanionTools, now
from chat_engine import ChatEngine
from config import CURRENT_CONFIG
from test_memory import IO


def run():
    checks={}
    with tempfile.TemporaryDirectory(prefix='companion-check-',dir=ROOT/'runtime') as temp:
        m=MemoryService(temp);m.ready.result(60)
        try:
            tools=CompanionTools(m)
            tools.profile()
            result=json.loads(tools.execute(dict(operation='preference',key='minutes',value='15')))
            checks['staged_preference']=result['ok'] and not m.submit('list').result(10)['results']
            tools.commit()
            m.close();m=MemoryService(temp);m.ready.result(60)
            tools=CompanionTools(m)
            checks['preference_restart']=tools.profile()['minutes']==15
            tools.execute(dict(operation='settings',key='minutes',value='10'))
            checks['temporary_override']=tools.settings['minutes']==10 and not tools.pending
            checks['temporary_not_persisted']=CompanionTools(m).profile()['minutes']==15
            checks['invalid_setting_rejected']=not json.loads(tools.execute(dict(operation='settings',key='volume',value='9')))['ok']
            checks['unknown_memory_rejected']=not json.loads(tools.execute(dict(operation='forget',memory_id='invented')))['ok']
            with m._lock:
                seed=m.store.add('用户目前住在北京。',user_id='default',infer=False)['results'][0]['id']
            tools.execute(dict(operation='search',query='用户住在哪个城市'))
            checks['revise_success']=json.loads(tools.execute(dict(operation='revise',memory_id=seed,text='用户目前住在上海。')))['ok']
            m.close();m=MemoryService(temp);m.ready.result(60)
            checks['revise_restart']=any('上海' in r['memory'] for r in m.search('我住在哪里'))
            tools=CompanionTools(m);tools.execute(dict(operation='search',query='用户住在哪个城市'))
            checks['forget_success']=json.loads(tools.execute(dict(operation='forget',memory_id=seed)))['ok']
            checks['forget_no_history']=not m.store.history(seed)
            yesterday=(now().date()-timedelta(days=1)).isoformat()
            tomorrow=(now().date()+timedelta(days=1)).isoformat()
            m.submit('managed',kind='plan',key='past',text='项目答辩',value=yesterday).result(20)
            m.submit('managed',kind='plan',key='future',text='周末出游',value=tomorrow).result(20)
            checks['followup_due']=m.submit('followup',today=now().date().isoformat()).result(20)=='项目答辩'
            m.close();m=MemoryService(temp);m.ready.result(60)
            checks['followup_once_restart']=m.submit('followup',today=now().date().isoformat()).result(20) is None
            # The queued action commits even when there are no episodic messages.
            m.close()
            with patch.dict(CURRENT_CONFIG,{'memory_dir':temp,'summaries_file':str(Path(temp)/'summary.json'),'long_term_memory':True,'agent_backend':'native'}):
                engine=ChatEngine(IO());engine.memory.ready.result(60)
                engine.tools.execute(dict(operation='preference',key='questions',value='fewer'))
                engine.finalize_session_memory()
                checks['engine_commits_without_transcript']=any((r.get('metadata') or {}).get('key')=='questions' for r in engine.memory.submit('list').result(10)['results'])
                engine.session_memory=[{'role':'user','content':'不想保存的事情'}];engine.chat_user_texts=['不想保存的事情']
                engine.tools.execute(dict(operation='forget_recent'))
                checks['forget_recent_blocks_extraction']=not engine.session_memory and not engine.chat_user_texts
                engine.handle_turn('清空记忆')
                checks['clear_all']=not engine.memory.submit('list').result(10)['results'] and not engine.tools.pending
                engine.close()
        finally:m.close()
    out={'checks':checks,'passed':sum(checks.values()),'total':len(checks)}
    (ROOT/'experiments/agent_compare/storage.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(out,ensure_ascii=False,indent=2));assert all(checks.values())

if __name__=='__main__':run()
