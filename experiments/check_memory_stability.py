"""Small Chinese scenario suite inspired by LongMemEval, not an official score.
Real Mem0 extraction and production agent answers; isolated data, fresh processes.
"""
import argparse,json,re,subprocess,sys,tempfile,time
from unittest.mock import patch
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

SESSIONS=[
 [('user','我叫林晓。我喜欢西红柿鸡蛋面，不喜欢香菜。'),('assistant','知道了，我会结合你的喜好聊天。')],
 [('user','我目前住在苏州，养了一只叫豆包的猫。'),('assistant','你住苏州，猫叫豆包。')],
 [('user','我之前提到的猫豆包，今年四岁了。'),('assistant','豆包四岁了。')],
]
QUESTIONS=[
 ('preference','我喜欢吃什么面？',['西红柿','鸡蛋']),
 ('negative_preference','我不喜欢吃哪种配菜？',['香菜']),
 ('cross_session','我养的宠物叫什么，今年几岁？',['豆包','四']),
 ('residence','我住在哪个城市？',['苏州']),
 ('abstention','我的生日是几月几号？',[]),
 ('irrelevant_emotion','今天好累，只想安静地歇一会儿。',[]),
 ('irrelevant_story','给我讲一个月亮的小故事，短一点。',[]),
 ('irrelevant_goodnight','晚安，简单说一句就好。',[]),
 ('irrelevant_injected','今晚想听一句简短的晚安。',[]),
]

def phase(name,directory,backend="native"):
 if not directory.resolve().is_relative_to((ROOT/'runtime').resolve()) or not directory.name.startswith('stability-'):
  raise ValueError('This test requires an isolated stability-* directory')
 from config import CURRENT_CONFIG
 CURRENT_CONFIG.update(long_term_memory=True,memory_dir=str(directory/'memory'),agent_backend=backend)
 from memory_backend import MemoryService
 out={'phase':name,'backend':backend,'checks':{},'rows':[]}
 if name=='write':
  m=MemoryService(directory/'memory');m.ready.result(60)
  try:
   for i,session in enumerate(SESSIONS):
    start=time.perf_counter();result=m.remember_session([{'role':r,'content':t} for r,t in session],f'stability-{i}').result(90)
    out['rows'].append({'input':session,'result':result,'seconds':time.perf_counter()-start})
   out['records']=m.submit('list').result(15)['results']
   out['checks']['nonempty_extraction']=bool(out['records'])
  finally:m.close()
 else:
  from chat_engine import ChatEngine
  from test_memory import IO
  for key,question,expected in QUESTIONS:
   engine=ChatEngine(IO());engine.memory.ready.result(60);engine.tools.strict_errors=True;engine.tools.metrics={}
   try:
    retrieved=engine.memory.search(question,strict=True)
    injected=key=='irrelevant_injected'
    supplied=engine.memory.submit('list').result(15)['results'][:4] if injected else retrieved
    start=time.perf_counter()
    with patch.object(engine.memory,'search',return_value=supplied) if injected else nullcontext():
     answer=engine.chat_and_respond(question)
    passed=all(token in answer for token in expected)
    if key=='cross_session':passed='豆包' in answer and ('四' in answer or '4' in answer)
    if key=='abstention':passed=any(token in answer for token in ['不知道','不清楚','没告诉','没有告诉','没提','没有提','没查到','没有查到','不确定','没有记','没记','没有你的生日']) and not re.search(r'\d{1,2}月\d{1,2}',answer)
    if key.startswith('irrelevant_'):
     passed=not any(token in answer for token in ['林晓','西红柿','鸡蛋','香菜','苏州','豆包','四岁','4岁'])
    out['checks'][key]=bool(answer) and passed
    out['rows'].append({'case':key,'question':question,'retrieved':retrieved,'injected_irrelevant_context':injected,'supplied_context':supplied,'answer':answer,'passed':out['checks'][key],'seconds':time.perf_counter()-start,'metrics':engine.tools.metrics})
   finally:
    engine._memory_submitted=True # Evaluation questions/answers must not become training memories.
    engine.close()
 (directory/(name+'.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print(name,out['checks'],flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--live',action='store_true',required=True);p.add_argument('--phase',choices=['write','read']);p.add_argument('--directory',type=Path);p.add_argument('--backend',choices=['native','langchain','pydantic'],default='native');args=p.parse_args()
 if args.phase:return phase(args.phase,args.directory,args.backend)
 out={'backend':args.backend,'scope':'3 synthetic Chinese sessions, actual extraction, fresh-process recall; not official LongMemEval','phases':[]}
 with tempfile.TemporaryDirectory(prefix='stability-',dir=ROOT/'runtime') as temp:
  directory=Path(temp)
  for name in ['write','read']:
   run=subprocess.run([sys.executable,__file__,'--live','--phase',name,'--directory',temp,'--backend',args.backend],timeout=240)
   if (directory/(name+'.json')).exists():out['phases'].append(json.loads((directory/(name+'.json')).read_text()))
   target=ROOT/'experiments'/f'memory_stability_{args.backend}_results.json';target.write_text(json.dumps(out,ensure_ascii=False,indent=2))
   run.check_returncode()
 assert all(value for phase in out['phases'] for value in phase['checks'].values()), 'Review retained failures'

if __name__=='__main__':main()
