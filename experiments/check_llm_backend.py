"""Offline adapter contract checks (no model service needed)."""
import json,sys
from pathlib import Path
from unittest.mock import patch
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import llm_backend as llm
seen=[]
def handler(request):
 body=json.loads(request.content);seen.append(body)
 if body['stream']:
  return httpx.Response(200,text='data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: {"choices":[{"delta":{"content":"晚安"}}]}\n\ndata: [DONE]\n\n',headers={'content-type':'text/event-stream'})
 return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':'晚安'}}]})
with httpx.Client(transport=httpx.MockTransport(handler)) as client, patch.object(llm,'_client',client),patch.object(llm,'ensure_server',return_value='http://test'),patch.dict(llm.CURRENT_CONFIG,{'llm_backend':'llama.cpp'}):
 args=dict(model='unused',messages=[{'role':'user','content':'你好'}],options={'num_predict':80,'temperature':.2})
 assert llm.chat(**args)['message']['content']=='晚安'
 assert list(llm.chat(**args,stream=True))==[{'message':{'content':'晚安'}}]
 assert seen[0]['max_tokens']==80 and seen[0]['temperature']==.2
 assert seen[0]['cache_prompt'] is True
with patch.dict(llm.CURRENT_CONFIG,{'llm_backend':'ollama'}),patch.object(llm._ollama,'chat',return_value={'ok':True}) as call:
 assert llm.chat(model='test',messages=[],options={'keep_alive':-1})=={'ok':True}
 assert 'keep_alive' not in call.call_args.kwargs['options']
print('PASS streaming, nonstream summary, option mapping, Ollama fallback')
with httpx.Client(transport=httpx.MockTransport(handler)) as client,patch.object(llm,'_cloud_client',client),patch.object(llm,'cloud_settings',return_value=('https://api.deepseek.com','deepseek-flash','test-only-key')),patch.dict(llm.CURRENT_CONFIG,{'llm_backend':'deepseek'}):
 assert list(llm.chat(model='local-ignored',messages=[{'role':'user','content':'你好'}],stream=True))==[{'message':{'content':'晚安'}}]
 assert seen[-1]['thinking']=={'type':'disabled'}
 assert seen[-1]['model']=='deepseek-flash'
 assert 'cache_prompt' not in seen[-1] and 'top_k' not in seen[-1]
print('PASS DeepSeek request mapping and content-only streaming')
