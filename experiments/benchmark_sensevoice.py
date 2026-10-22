"""SenseVoice ONNX pilot: report load and warm inference separately, same references."""
import json,sys,time,wave,resource
from pathlib import Path
import numpy as np
import sherpa_onnx
from score_cer import normalize,distance
root=Path(__file__).resolve().parent
model=Path(sys.argv[1])
refs=json.loads((root/'q8_clips/references.json').read_text())['clips']
t=time.perf_counter()
recognizer=sherpa_onnx.OfflineRecognizer.from_sense_voice(model=str(model/'model.int8.onnx'),tokens=str(model/'tokens.txt'),num_threads=4,language='zh',use_itn=True)
load=time.perf_counter()-t
rows=[]
for trial in range(3):
 for ref in (refs if trial%2==0 else refs[::-1]):
  begin=time.perf_counter()
  with wave.open(str(root/'q8_clips'/f"{ref['clip']}.wav")) as w:
   assert w.getnchannels()==1 and w.getsampwidth()==2
   rate=w.getframerate();samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
  stream=recognizer.create_stream();stream.accept_waveform(rate,samples);recognizer.decode_stream(stream)
  elapsed=time.perf_counter()-begin;text=stream.result.text
  errors=distance(normalize(ref['reference']),normalize(text))
  rows.append(dict(trial=trial,clip=ref['clip'],seconds=elapsed,text=text,errors=errors,reference_chars=len(normalize(ref['reference']))))
  result=dict(model=str(model),engine='sherpa-onnx SenseVoice INT8 (not FunASR Python runtime)',threads=4,model_load_seconds=load,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows)
  (root/'sensevoice_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
  print(trial,ref['clip'],round(elapsed,3),errors,text,flush=True)
print('load',load,'max RSS MiB',result['max_rss_kib']/1024)
