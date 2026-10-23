"""Real model plan registration with isolated storage; no production memories."""
import json,sys,tempfile,time
from pathlib import Path
from datetime import timedelta
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from config import CURRENT_CONFIG
from chat_engine import ChatEngine
from companion_tools import now
from test_memory import IO

cases=['我明天要考试']*3+['明天有一场考试，有点紧张。','记一下，我明天要考试。','如果明天要考试，我可能会紧张。','我明天没有考试。']
rows=[]
with tempfile.TemporaryDirectory(prefix='plan-registration-',dir=ROOT/'runtime') as temp:
 for i,prompt in enumerate(cases):
  with patch.dict(CURRENT_CONFIG,{'memory_dir':str(Path(temp)/str(i)),'long_term_memory':True,'agent_backend':'native','debug_prompt':False}):
   e=ChatEngine(IO());e.memory.ready.result(60);e.tools.strict_errors=True;e.tools.metrics={}
   expected=(now().date()+timedelta(days=1)).isoformat()
   try:
    start=time.perf_counter();answer=e.chat_and_respond(prompt)
    pending=list(e.tools.pending.values())
    # Persist only actual tool registrations; ordinary extraction is a separate test.
    e.tools.commit()
    stored=e.memory.submit('list').result(20)['results']
    row={'input':prompt,'answer':answer,'expected_date':expected,'pending':pending,'stored':stored,'audit':e.tools.audit,'metrics':e.tools.metrics,'seconds':time.perf_counter()-start}
    rows.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
    (ROOT/'experiments/plan_registration_results.json').write_text(json.dumps({'scope':'Live native decisions and actual plan persistence; ordinary extraction not run','rows':rows},ensure_ascii=False,indent=2))
   finally:
    e._memory_submitted=True;e.close()
