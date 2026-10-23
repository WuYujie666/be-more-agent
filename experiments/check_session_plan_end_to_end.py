"""Real end-of-session worker, facts + plans, restart and opening acceptance."""
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory_backend import MemoryService
from chat_engine import ChatEngine
from companion_tools import now
from config import CURRENT_CONFIG
from test_memory import IO
results={}
with tempfile.TemporaryDirectory(prefix='session-e2e-',dir=ROOT/'runtime') as d:
 current=now()
 with patch.dict(CURRENT_CONFIG,{'memory_dir':d,'long_term_memory':True,'agent_backend':'native'}):
  e=ChatEngine(IO());e.memory.ready.result(60)
  e.session_memory=[{'role':'user','content':'我明天要考试，有点紧张。我喜欢西红柿鸡蛋面。','observed_at':current.isoformat()},
                    {'role':'assistant','content':'听起来有些紧张，我陪你。'}]
  e.finalize_session_memory();e.memory.submit('flush').result(120)
  records=e.memory.submit('list').result(20)['results']
  plans=[r for r in records if (r.get('metadata') or {}).get('kind')=='plan']
  results['plan_saved']=len(plans)==1 and plans[0]['metadata']['value']==(current.date()+timedelta(days=1)).isoformat()
  results['ordinary_fact_saved']=any('西红柿' in r['memory'] for r in records)
  results['records']=records;e.close()
  e=ChatEngine(IO());e.memory.ready.result(60)
  with patch('companion_tools.now',return_value=current+timedelta(days=2)):
   opening,used=e.build_greeting()
  results['restart_opening']=used and '考试' in opening
  results['opening']=opening;e.close()
(ROOT/'experiments/session_plan_end_to_end_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False,indent=2))
assert results['plan_saved'] and results['ordinary_fact_saved'] and results['restart_opening']
