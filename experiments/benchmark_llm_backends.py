"""Matched raw ChatML prompts, same GGUF, greedy CPU generation, sequential runs."""
import json,sys,time
from pathlib import Path
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import CURRENT_CONFIG as cfg
root=Path(__file__).resolve().parent
CACHE='--prefix-cache' in sys.argv
s=httpx.Client(trust_env=False)
prompts=['今天上班很累，项目还没完成，心里一直放不下。','今天和朋友见面很开心，但现在有点孤单。']
def request(backend,prompt,limit):
 if backend=='ollama':
  url='http://127.0.0.1:11434/api/generate'
  body=dict(model='qwen2.5:1.5b',prompt=prompt,raw=True,stream=True,keep_alive=-1,options=dict(num_thread=4,num_ctx=2048,num_predict=limit,temperature=0,seed=1234,repeat_penalty=1.0,stop=['<|im_end|>','<|endoftext|>']))
 else:
  url='http://127.0.0.1:18080/completion'
  body=dict(prompt=prompt,n_predict=limit,stream=True,cache_prompt=CACHE,temperature=0,seed=1234,repeat_penalty=1.0,stop=['<|im_end|>','<|endoftext|>'])
 begin=time.perf_counter();first=None;text='';last={}
 with s.stream('POST',url,json=body,timeout=180) as r:
  r.raise_for_status()
  for line in r.iter_lines():
   if not line:continue
   if line.startswith('data: '):line=line[6:]
   if line=='[DONE]':continue
   x=json.loads(line);piece=x.get('response','') if backend=='ollama' else x.get('content','')
   if piece and first is None:first=time.perf_counter()-begin
   text+=piece;last=x
 return dict(first_token=first,total=time.perf_counter()-begin,text=text,metrics=last)
rows=[]
for trial in range(3):
 for index,user in enumerate(prompts):
  prompt=f"<|im_start|>system\n{cfg['system_prompt']}<|im_end|>\n<|im_start|>user\n{user}\n请用中文回答。<|im_end|>\n<|im_start|>assistant\n"
  for backend in (['ollama','llama.cpp'] if trial%2==0 else ['llama.cpp','ollama']):
   warm=f"<|im_start|>system\n{cfg['system_prompt']}<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" if CACHE else 'Warmup.'
   request(backend,warm,1)
   result=request(backend,prompt,64)
   # Preserve metrics but avoid duplicating response text embedded in final message.
   row=dict(trial=trial,prompt_index=index,backend=backend,prompt=prompt,**result);rows.append(row)
   (root/('llm_backend_cached_results.json' if CACHE else 'llm_backend_results.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
   print(trial,index,backend,'TTFT',round(result['first_token'] or 0,3),'total',round(result['total'],3),repr(result['text']),flush=True)
