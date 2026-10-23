"""Real 25s file ASR -> each agent -> production Sherpa renderer, repeated twice.
No microphone/VAD/GUI. Synthesized first audio is measured; optional playback.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',required=True,action='store_true');parser.add_argument('--play',action='store_true');args=parser.parse_args()
    import asr_backend
    import agent
    from chat_engine import ChatEngine
    from config import CURRENT_CONFIG
    render=agent.BotGUI.__new__(agent.BotGUI);render._init_sherpa_tts();assert render.sherpa_tts
    render._render_sherpa('晚上好。')
    asr_backend.warmup()
    for name in ['native','langchain','pydantic']:importlib.import_module('agent_backends.'+name)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='voice-compare-',dir=ROOT/'runtime') as temp:
        for trial,order in enumerate([['native','langchain','pydantic'],['pydantic','langchain','native']]):
            for backend in order:
                begin=time.perf_counter();text=asr_backend.transcribe(ROOT/'input_test_data.wav');asr_done=time.perf_counter()
                class IO:
                    def __init__(self):self.pool=ThreadPoolExecutor(max_workers=1);self.jobs=[];self.first_ready=None;self.first_audio=None
                    def speak(self,sentence):
                        def synth():
                            audio=render._render_sherpa(sentence)
                            if self.first_ready is None:self.first_ready=time.perf_counter();self.first_audio=audio
                            return audio
                        self.jobs.append(self.pool.submit(synth))
                    def wait_speech(self):
                        for f in self.jobs:f.result(60)
                    def is_interrupted(self):return False
                    def __getattr__(self,name):return lambda *a,**kw:None
                io=IO()
                with patch.dict(CURRENT_CONFIG,{'agent_backend':backend,'long_term_memory':False,'summaries_file':str(Path(temp)/'summary.json')}):
                    engine=ChatEngine(io);engine.tools.metrics={}
                    from experiments.benchmark_agents import reserve_request
                    engine.tools.budget=reserve_request;engine.tools.strict_errors=True
                    answer=engine.chat_and_respond(text);io.wait_speech()
                    rows.append(dict(backend=backend,trial=trial,transcript=text,answer=answer,
                                     asr_seconds=asr_done-begin,
                                     first_audio_after_asr_seconds=io.first_ready-asr_done if io.first_ready else None,
                                     first_audio_from_file_seconds=io.first_ready-begin if io.first_ready else None,
                                     metrics=engine.tools.metrics))
                    engine.close()
                io.pool.shutdown()
                if args.play and trial==0 and backend=='langchain':
                    import sounddevice as sd
                    samples,rate=io.first_audio;sd.play(samples,rate);sd.wait()
                (ROOT/'experiments/agent_compare/voice-decision.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
                print(backend,trial,'first_audio',rows[-1]['first_audio_after_asr_seconds'],flush=True)
                assert answer and io.first_ready

if __name__=='__main__':main()
