"""Five fixed excerpts, FP16/Q8 A/B, three rounds; no microphone/playback."""
import hashlib,json,re,subprocess,time,wave
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'experiments/q8_clips';out.mkdir(exist_ok=True)
source=root/'input_test_data.wav'
clips=[('3s',0,3),('5s',3,8),('7s',8,15),('10s',13,23),('13s',0,13)]
with wave.open(str(source)) as w:
 p=w.getparams();data=w.readframes(w.getnframes())
for name,start,end in clips:
 assert end<=p.nframes/p.framerate
 stride=p.nchannels*p.sampwidth
 with wave.open(str(out/(name+'.wav')),'wb') as w:
  w.setparams(p);w.writeframes(data[int(start*p.framerate)*stride:int(end*p.framerate)*stride])
rows=[]
for trial in range(3):
 for name,start,end in clips:
  variants=[('fp16','ggml-small.bin'),('q8','ggml-small-q8_0.bin')]
  if (trial+clips.index((name,start,end)))%2:variants.reverse()
  for variant,model in variants:
   cmd=[str(root/'whisper.cpp/build/bin/whisper-cli'),'-m',str(root/'whisper.cpp/models'/model),'-l','zh','-t','4','-bs','1','-bo','1','-f',str(out/(name+'.wav')),'--prompt','以下是普通话的句子。']
   begin=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);seconds=time.perf_counter()-begin
   (out/f'{name}-{variant}-{trial}.log').write_text(r.stdout+'\n'+r.stderr)
   r.check_returncode()
   text=''.join(re.sub(r'^\[[^]]+\]\s*','',line.strip()) for line in r.stdout.splitlines() if line.strip())
   rows.append(dict(trial=trial,clip=name,start=start,end=end,variant=variant,seconds=seconds,text=text,sha256_audio=hashlib.sha256((out/(name+'.wav')).read_bytes()).hexdigest()))
   (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
   print(f'{trial} {name} {variant} {seconds:.3f}s {text}',flush=True)
print('DONE',flush=True)
