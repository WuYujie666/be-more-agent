"""Paired retrieval-threshold vs prompt ablation, with fixed previously extracted facts.
No production data or configuration mutations. --live required for cloud generation.
Backend selectable so the same behavioral cases can follow a future framework change.
"""
import argparse,json,re,sys,tempfile,time,statistics
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/memory_relevance';OUT.mkdir(exist_ok=True)
STRICT=('只使用与本轮问题直接相关且回答必需的记忆；无关时完全忽略。'
        '不要为了展示记忆而额外称呼检索出的姓名、提及住址饮食宠物或转移到旧话题。'
        '普通问候、安慰、故事不应硬套个人资料。')
FACTS={'name':['林晓'],'food':['西红柿','鸡蛋'],'dislike':['香菜'],'city':['苏州'],
       'pet':['豆包'],'age':['四岁','4岁']}
# Calibration is separate from the held-out answer cases below.
CALIBRATION=[
 ('我叫什么？',['name']),('我爱吃哪种面？',['food']),('我讨厌什么配菜？',['dislike']),
 ('我在哪里生活？',['city']),('我的小猫叫什么名字？',['pet']),('我的猫多大？',['age']),
 ('晚上好。',[]),('今天累坏了。',[]),('说一个短童话。',[]),('谢谢你。',[]),
 ('我现在想安静一下。',[]),('睡前说句温柔的话吧。',[]),
]
CASES=[
 ('food_direct','我喜欢吃什么面？',['food'],'recall'),
 ('food_paraphrase','还记得我偏爱的面条口味吗？',['food'],'recall'),
 ('dislike','给我点餐时，哪种配菜应该避开？',['dislike'],'recall'),
 ('city','我现在住的城市叫什么？',['city'],'recall'),
 ('pet','我家那只猫的名字你记得吗？',['pet'],'recall'),
 ('pet_age','我的猫叫什么，今年多大了？',['pet','age'],'recall'),
 ('multi','我住哪儿，养的宠物又叫什么？',['city','pet'],'recall'),
 ('name','我之前告诉过你我的名字，你记得吗？',['name'],'recall'),
 ('tired','今天好累，只想安静地歇一会儿。',[],'irrelevant'),
 ('story','给我讲一个月亮的小故事，短一点。',[],'irrelevant'),
 ('night','晚安，简单说一句就好。',[],'irrelevant'),
 ('thanks','谢谢你陪我聊天。',[],'irrelevant'),
 ('quiet','我想安静一会儿，你陪着就好。',[],'irrelevant'),
 ('breathing','陪我做一次简单的深呼吸。',[],'irrelevant'),
 ('stress','工作让我有点烦，先听我说，不用出主意。',[],'irrelevant'),
 ('rain','外面下雨了，听着挺舒服的。',[],'irrelevant'),
 ('birthday','我的生日是几月几号？',[],'unknown'),
 ('job','我做什么职业？',[],'unknown'),
 ('school','我在哪所大学读过书？',[],'unknown'),
 ('travel','我去年去过哪个国家旅行？',[],'unknown'),
 ('injected_night','今晚想听一句简短的晚安。',[],'injected'),
 ('injected_story','讲一个关于星星的简短故事。',[],'injected'),
 ('injected_tired','好累啊，请只安慰我一句。',[],'injected'),
 ('injected_thanks','谢谢，今晚就聊到这里吧。',[],'injected'),
]

def has_fact(text,key):
 tokens=FACTS[key]
 return any(t in text for t in tokens) if key=='age' else all(t in text for t in tokens)

def grade(answer, expected, kind):
 hits=[key for key in FACTS if has_fact(answer,key)]
 if kind=='recall':passed=all(has_fact(answer,key) for key in expected)
 elif kind=='unknown':
  passed=bool(re.search(r'不(知道|清楚|确定)|没.{0,16}(记|提|说|存|告诉|信息|资料)|不太了解|还不知',answer))
 else:passed=not hits
 return {'passed':bool(answer) and passed,'personal_mentions':hits,
         'name_only':hits==['name'],'unrelated_fact_mentions':[k for k in hits if k!='name'] if not expected else []}

class Budget:
 def __init__(self):self.committed=0.;self.pending=[]
 def reserve(self,request):
  # UTF-8 byte count + framing allowance, all input charged at peak uncached price.
  reserve=((len(request.content)+4096)*2+600*8)/1e6
  if self.committed+sum(self.pending)+reserve>1:raise RuntimeError('One RMB test budget reached')
  self.pending.append(reserve)
 def settle(self,metrics):
  usage=metrics.get('usage',[])
  if len(usage)==len(self.pending):self.committed+=sum((u['input']*2+u['output']*8)/1e6 for u in usage)
  else:self.committed+=sum(self.pending) # Unknown usage keeps conservative reservation.
  self.pending=[]

def run(args):
 from config import CURRENT_CONFIG
 from memory_backend import MemoryService,memory_context
 from chat_engine import ChatEngine
 from test_memory import IO
 source=ROOT/'experiments/memory_stability_native_results.json'
 records=json.loads(source.read_text())['phases'][0]['records']
 result={'backend':args.backend,'seed_source':str(source.relative_to(ROOT)),
         'seed_records':records,'calibration':[],'retrieval':[],'rows':[],
         'scope':'Fixed six synthetic facts; not a full memory benchmark; injected arms bypass retrieval deliberately'}
 budget=Budget();target=OUT/f'{args.backend}.json'
 def save():
  result['conservative_cost_cny']=budget.committed
  target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with tempfile.TemporaryDirectory(prefix='relevance-',dir=ROOT/'runtime') as temp:
  m=MemoryService(temp);m.ready.result(60)
  try:
   with m._lock:
    for r in records:m.store.add(r['memory'],user_id='default',infer=False,metadata=r.get('metadata') or {})
   def retrieve(query,threshold):
    with m._lock:return m.store.search(query,filters={'user_id':'default'},top_k=4,threshold=threshold)['results']
   for question,expected in CALIBRATION:
    found=retrieve(question,0.)
    result['calibration'].append({'question':question,'expected':expected,'candidates':found})
   thresholds=[.45,.475,.5,.525,.55,.6]
   scores=[]
   for threshold in thresholds:
    positives=[];negatives=[]
    for row in result['calibration']:
     kept=[r for r in row['candidates'] if r['score']>=threshold]
     if row['expected']:positives.append(all(any(has_fact(r['memory'],k) for r in kept) for k in row['expected']))
     else:negatives.append(not kept)
    score={'threshold':threshold,'recall':sum(positives)/len(positives),'reject_irrelevant':sum(negatives)/len(negatives)}
    score['balanced']=(score['recall']+score['reject_irrelevant'])/2;scores.append(score)
   # Compare a genuinely higher cutoff; ties prefer higher recall, then lower cutoff.
   chosen=max(scores[1:],key=lambda s:(s['balanced'],s['recall'],-s['threshold']))['threshold']
   result['threshold_selection']={'scores':scores,'selected':chosen}
   for key,question,expected,kind in CASES:
    found=retrieve(question,0.)
    result['retrieval'].append({'case':key,'question':question,'expected':expected,'kind':kind,'candidates':found})
   save();print('SELECTED',chosen,'GRID',scores,flush=True)
   if not args.live:return
   arms={'baseline':(.45,False),'threshold':(chosen,False),'prompt':(.45,True)}
   for repeat in range(args.repeats):
    for i,(key,question,expected,kind) in enumerate(CASES):
     order=list(arms);rotation=(i+repeat)%3;order=order[rotation:]+order[:rotation]
     for arm in order:
      threshold,strong=arms[arm];trace=[]
      with patch.dict(CURRENT_CONFIG,{'long_term_memory':True,'agent_backend':args.backend,'debug_prompt':False}),patch('chat_engine.MemoryService',return_value=m):
       engine=ChatEngine(IO())
      engine.tools.strict_errors=True;engine.tools.metrics={};engine.tools.budget=budget.reserve
      def search(query,**kwargs):
       if not query.strip():raise ValueError('Empty query')
       rows=m.submit('list').result(15)['results'][:4] if kind=='injected' else retrieve(query,threshold)
       trace.append({'query':query,'rows':rows});return rows
      def context(rows):
       context=memory_context(rows)
       if context and strong:context={'role':'system','content':context['content']+'\n'+STRICT}
       return context
      start=time.perf_counter()
      with patch.object(m,'search',side_effect=search),patch('chat_engine.memory_context',side_effect=context):answer=engine.chat_and_respond(question)
      budget.settle(engine.tools.metrics)
      row={'repeat':repeat,'case':key,'kind':kind,'arm':arm,'threshold':threshold,'question':question,
           'expected':expected,'answer':answer,'seconds':time.perf_counter()-start,'retrieval_trace':trace,
           'audit':engine.tools.audit,'metrics':engine.tools.metrics,**grade(answer,expected,kind)}
      result['rows'].append(row);save()
      print(repeat,key,arm,row['passed'],row['personal_mentions'],round(row['seconds'],2),flush=True)
  finally:m.close()
 save()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--backend',choices=['native','langchain','pydantic'],default='native');p.add_argument('--repeats',type=int,default=2)
 run(p.parse_args())
