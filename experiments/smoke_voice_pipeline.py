"""One real GUI/ASR/LLM/TTS turn from a file; no microphone or memory writes.
Uses configured backend (cloud mode makes API calls). Plays actual audio.
"""
import sys,time,tkinter as tk
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import agent
agent.CURRENT_CONFIG["long_term_memory"] = False
from chat_engine import ChatEngine
completed=[]
ChatEngine.finalize_session_memory=lambda self:None
# No saved personal summary in cloud smoke-test context.
def greeting(self):
 text='晚上好。';self.session_memory.append({'role':'assistant','content':text});return text,False
ChatEngine.build_greeting=greeting
def record(self,filename='input.wav'):
 self.response_speech_end=self.response_recording_end=time.perf_counter()
 return 'experiments/q8_clips/3s.wav'
agent.BotGUI.record_voice_vad=record
original=ChatEngine.chat_and_respond
def once(self,*args,**kwargs):
 result=original(self,*args,**kwargs)
 assert result,'No successful model reply'
 completed.append(result)
 print('SMOKE PIPELINE COMPLETED',flush=True)
 self.io.gui.master.after(0,self.io.gui.safe_exit)
 return result
ChatEngine.chat_and_respond=once
root=tk.Tk();app=agent.BotGUI(root)
root.after(90000,app.safe_exit)
root.mainloop()
assert completed,'Pipeline did not finish successfully'
