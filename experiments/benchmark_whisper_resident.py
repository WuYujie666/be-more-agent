"""Sequential requests to a separately started local whisper-server with Q8."""
import json,time
from pathlib import Path
import httpx
from score_cer import normalize,distance
root=Path(__file__).resolve().parent
refs=json.loads((root/'q8_clips/references.json').read_text())['clips'];rows=[]
s=httpx.Client(trust_env=False)
for trial in range(3):
 for ref in refs if trial%2==0 else refs[::-1]:
  with (root/'q8_clips'/f"{ref['clip']}.wav").open('rb') as f:
   t=time.perf_counter();r=s.post('http://127.0.0.1:18081/inference',files={'file':f},data={'response_format':'json'},timeout=90);elapsed=time.perf_counter()-t
  r.raise_for_status();text=r.json()['text'].strip()
  errors=distance(normalize(ref['reference']),normalize(text));rows.append(dict(trial=trial,clip=ref['clip'],seconds=elapsed,text=text,errors=errors,reference_chars=len(normalize(ref['reference']))))
  (root/'whisper_resident_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
  print(trial,ref['clip'],round(elapsed,3),errors,text,flush=True)
