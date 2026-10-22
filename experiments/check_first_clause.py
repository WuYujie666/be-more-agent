"""Exercise production streaming segmentation without hardware or model calls."""
import contextlib,io,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chat_engine import ChatEngine
class Output:
 def __init__(self):self.parts=[];self.display=[]
 def speak(self,s):self.parts.append(s)
 def token(self,s):self.display.append(s)
 def is_interrupted(self):return False
 def set_state(self,*args):pass
 def bot_start(self):pass
 def wait_speech(self):pass
cases=[('下班回家后可以先放松一下身体，喝杯热水，慢慢休息。','下班回家后可以先放松一下身体，'),('今天辛苦了，早点休息吧。','今天辛苦了，早点休息吧。'),('哦？今天过得如何？','哦？'),('今天我们聊了许多事情；现在休息吧。','今天我们聊了许多事情；'),('今天我们聊了许多事情:现在休息吧。','今天我们聊了许多事情:'),('今天我们聊了许多事情,现在休息吧。','今天我们聊了许多事情,'),('这是没有结束标点的一段回复','这是没有结束标点的一段回复'),('下班回家后可以先放松一下身体，[AUDIO:white_noise]喝杯热水。','下班回家后可以先放松一下身体，')]
count=0
for text,first in cases:
 for size in [1,2,5,len(text)]:
  output=Output();engine=ChatEngine.__new__(ChatEngine);engine.io=output;engine.memory=None;engine.permanent_memory=[];engine.session_memory=[]
  chunks=[{'message':{'content':text[i:i+size]}} for i in range(0,len(text),size)]
  with patch('chat_engine.llm.chat',return_value=iter(chunks)),contextlib.redirect_stdout(io.StringIO()):
   result=engine.chat_and_respond('测试')
  assert output.parts[0]==first,(size,text,output.parts)
  expected=text.replace('[AUDIO:white_noise]','')
  assert ''.join(output.parts)==expected,(text,output.parts)
  assert ''.join(output.display)==expected
  assert result==expected
  count+=1
print(f'PASS {count} streaming cases: first boundary, no lost/duplicated text, split control tags')
