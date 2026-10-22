"""Research only: actual overlapping Ollama generation + first-chunk Sherpa TTS.
No recording/playback, no application config or memory writes. Uses local Ollama.
Measures first synthesized audio ready, excluding STT/device output latency.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import ollama
import sherpa_onnx
from config import CURRENT_CONFIG as cfg, OLLAMA_OPTIONS

out=Path('/tmp/first-audio-research');out.mkdir(exist_ok=True)
client=ollama.Client(host='http://127.0.0.1:11434',trust_env=False,timeout=120)
model=cfg['text_model'];directory=root/cfg['sherpa_model_dir']

def make_tts(threads):
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(directory/'vits-aishell3.onnx'),lexicon=str(directory/'lexicon.txt'),tokens=str(directory/'tokens.txt')),
            num_threads=threads,provider='cpu'),
        rule_fsts=','.join(str(directory/name) for name in ['date.fst','number.fst','phone.fst','new_heteronym.fst']),
        rule_fars=str(directory/'rule.far'),max_num_sentences=1))

tts={n:make_tts(n) for n in [1,2,4]}
# Warm each TTS model. Does not play the generated samples.
for n,engine in tts.items():engine.generate('晚上好。',sid=cfg.get('sherpa_speaker_id',0),speed=cfg.get('sherpa_speed',1.0))

prompts=['今天上班很累，项目还没完成，心里一直放不下。','今天和朋友见面很开心，但现在有点孤单。']
variants=[('sentence_4_4',False,4,4),('clause_4_4',True,4,4),('clause_3_1',True,3,1),('clause_2_2',True,2,2)]
rows=[]
for trial in range(2):
    for prompt_id,prompt in enumerate(prompts):
        for name,early,llm_threads,tts_threads in (variants if trial==0 else list(reversed(variants))):
            options={k:v for k,v in OLLAMA_OPTIONS.items() if k!='keep_alive'}
            options.update(num_thread=llm_threads,seed=1234,num_predict=128)
            system={'role':'system','content':cfg['system_prompt']}
            # Restore the same short prefix before every measurement. This
            # avoids benchmarking one variant with the full answer cached.
            client.chat(model=model,messages=[system,{'role':'user','content':'你好'}],
                        options={**options,'num_predict':1},keep_alive=-1)
            start=time.perf_counter();first_token=None;submitted=None;first_text=None
            text='';pending='';buffer='';future=None;last=None
            def synth(sentence):
                clean=re.sub(r'[^\w\s,.!?:-，。！？、；：]','',sentence)
                begin=time.perf_counter()
                audio=tts[tts_threads].generate(clean,sid=cfg.get('sherpa_speaker_id',0),speed=cfg.get('sherpa_speed',1.0))
                end=time.perf_counter()
                return dict(ready=end-start,synthesis=end-begin,audio_seconds=len(audio.samples)/audio.sample_rate)
            with ThreadPoolExecutor(max_workers=1) as pool:
                for chunk in client.chat(model=model,messages=[system,{'role':'user','content':prompt+'\n请用中文回答。'}],
                                         stream=True,options=options,keep_alive=-1):
                    last=chunk
                    piece=chunk.message.content
                    if piece and first_token is None:first_token=time.perf_counter()-start
                    text+=piece;pending+=piece
                    # Avoid feeding internal bracketed control tags to TTS.
                    emitted=''
                    while pending:
                        i=pending.find('[')
                        if i<0:emitted+=pending;pending='';break
                        emitted+=pending[:i];pending=pending[i:]
                        j=pending.find(']')
                        if j<0:break
                        bracket=pending[:j+1]
                        if 'AUDIO:' not in bracket.upper():emitted+=bracket
                        pending=pending[j+1:]
                    buffer+=emitted
                    if future is None:
                        cut=None
                        if early:
                            for i,ch in enumerate(buffer):
                                if ch in '.!?\n。！？' or (ch in '，,；;：:' and len(re.findall(r'\w',buffer[:i]))>=8):
                                    cut=i+1;break
                        elif any(ch in emitted for ch in '.!?\n。！？'):cut=len(buffer)
                        if cut is not None:
                            first_text=buffer[:cut];submitted=time.perf_counter()-start
                            future=pool.submit(synth,first_text)
                llm_done=time.perf_counter()-start
                if future is None:
                    first_text=buffer;submitted=time.perf_counter()-start
                    future=pool.submit(synth,first_text)
                metrics=future.result()
            row=dict(trial=trial,prompt=prompt_id,variant=name,first_token=first_token,first_queued=submitted,
                     first_text=first_text,reply=text,llm_done=llm_done,done_reason=last.done_reason,
                     eval_count=last.eval_count,**metrics)
            rows.append(row);(out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
            print(f'{trial} prompt={prompt_id} {name} token={first_token:.2f}s queued={submitted:.2f}s ready={metrics["ready"]:.2f}s synth={metrics["synthesis"]:.2f}s first={first_text!r}',flush=True)
print('DONE',out/'results.json')
