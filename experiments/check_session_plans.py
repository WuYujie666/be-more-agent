"""Live extraction and real isolated Mem0 plan lifecycle; no production data."""
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory_backend import MemoryService
from session_plans import extract_plans,apply_plans

cases=[('plain',['我明天要考试'],None),('plain_repeat',['我明天要考试'],None),
('emotion',['明天有一场考试，有点紧张。'],None),
('hypothetical',['如果明天考试，我可能会紧张。'],None),
('negative',['我明天没有考试。'],None),
('reschedule',['我明天要考试','考试改到后天了'],None),
('cancel_new',['我明天要考试','考试取消了'],None),
('forget_new',['我明天要考试','刚才考试的事别保存'],None),
('midnight',['我明天要考试'],None),
('cancel_existing',['我之前说的考试取消了'],'cancel'),
('move_existing',['之前那场考试改到后天了'],'move'),
('forget_existing',['把之前考试的计划忘掉，别再存着了'],'forget')]
rows=[]
with tempfile.TemporaryDirectory(prefix='session-plan-check-',dir=ROOT/'runtime') as d:
 m=MemoryService(d);m.ready.result(60)
 try:
  for name,texts,seed in cases:
   m.submit('clear').result(20)
   if seed:m.submit('managed',kind='plan',key='考试:2026-10-23',text='考试',value='2026-10-23').result(20)
   stamp='2026-10-22T23:59:00+08:00' if name=='midnight' else '2026-10-22T12:00:00+08:00'
   messages=[{'role':'user','content':t,'observed_at':stamp} for t in texts]
   with m._lock:
    existing=m.store.get_all(filters={'user_id':'default'},top_k=100)['results']
    batch=extract_plans(m.store.llm,messages,existing,'2026-10-23T00:05:00+08:00' if name=='midnight' else stamp)
    apply_plans(m.store,'default',batch,existing,name,stamp)
   records=m.submit('list').result(20)['results']
   active=[r for r in records if (r.get('metadata') or {}).get('status','active')=='active']
   expected='2026-10-24' if name in ('reschedule','move_existing') else '2026-10-23'
   negative=name in ('hypothetical','negative','cancel_new','forget_new','cancel_existing','forget_existing')
   passed=not active if negative else len(active)==1 and active[0]['metadata']['value']==expected
   rows.append({'case':name,'input':messages,'changes':batch.model_dump(mode='json'),'records':records,'passed':passed})
   print(name,passed,batch.model_dump(mode='json'),flush=True)
   (ROOT/'experiments/session_plan_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 finally:m.close()
assert all(r['passed'] for r in rows)
