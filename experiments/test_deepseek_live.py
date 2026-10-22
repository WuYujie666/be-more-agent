"""Small, explicit cloud test. Sends synthetic prompts only; no saved memories."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import llm_backend as llm
from config import CURRENT_CONFIG,SYSTEM_PROMPT
CURRENT_CONFIG['llm_backend']='deepseek'
rows=[]
cases=[('emotion',[{'role':'user','content':'今天上班很累，项目还没完成，心里一直放不下。'}]),('context',[{'role':'user','content':'明天要做项目汇报，我最担心被问到预算。'},{'role':'assistant','content':'最让你挂心的是预算那部分，今晚先把这份担心放在这里。'},{'role':'user','content':'对，我担心的具体是什么？'}]),('audio',[{'role':'user','content':'我想听白噪音睡觉。'}])]
for label,messages in cases:
 start=time.perf_counter();first=None;parts=[]
 stream=llm.chat(model='unused',messages=[{'role':'system','content':SYSTEM_PROMPT}]+messages,stream=True,options={'num_predict':160})
 try:
  for chunk in stream:
   if first is None:first=time.perf_counter()-start
   parts.append(chunk['message']['content'])
 finally:stream.close()
 text=''.join(parts)
 assert text,'Empty response'
 row=dict(case=label,first_token=first,total=time.perf_counter()-start,text=text);rows.append(row)
 print(json.dumps(row,ensure_ascii=False),flush=True)
 Path('experiments/deepseek_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
response=llm.chat(model='unused',messages=[{'role':'user','content':'请把“今天工作很累，但终于完成了汇报”总结成一句简短中文。'}],options={'num_predict':80})
assert response['message']['content']
print('Non-stream summary call passed',flush=True)
