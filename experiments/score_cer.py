"""Score saved ASR runs against user-reviewed references. No model calls."""
import json
from pathlib import Path
import unicodedata
ROOT=Path(__file__).resolve().parent

def normalize(text):
    # Ignore punctuation and whitespace; preserve words, digits and symbols.
    return ''.join(c for c in unicodedata.normalize('NFKC',text) if not c.isspace() and not unicodedata.category(c).startswith('P'))

def distance(reference,hypothesis):
    previous=list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        current=[i]
        for j,b in enumerate(hypothesis,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]

def main():
    refs=json.loads((ROOT/'q8_clips/references.json').read_text())
    rows=json.loads((ROOT/'q8_clips/results.json').read_text())
    scores=[]
    for ref in refs['clips']:
        r=normalize(ref['reference'])
        for variant in ['fp16','q8']:
            runs=[row for row in rows if row['clip']==ref['clip'] and row['variant']==variant]
            assert len(runs)==3
            texts=set(row['text'] for row in runs)
            assert len(texts)==1,'Transcripts vary: score every run separately'
            hypothesis=texts.pop();errors=distance(r,normalize(hypothesis))
            scores.append(dict(clip=ref['clip'],variant=variant,reference=ref['reference'],hypothesis=hypothesis,reference_chars=len(r),errors=errors,cer=errors/len(r)))
    for score in scores:
        ref=next(r for r in refs['clips'] if r['clip']==score['clip'])
        reference=normalize(score['reference']);hypothesis=normalize(score['hypothesis'])
        for filler in ref.get('optional_fillers', []):
            reference=reference.replace(filler,'');hypothesis=hypothesis.replace(filler,'')
        score['filler_tolerant_errors']=distance(reference,hypothesis)
        score['filler_tolerant_reference_chars']=len(reference)
        score['filler_tolerant_cer']=distance(reference,hypothesis)/len(reference)
    totals={}
    for variant in ['fp16','q8']:
        selected=[s for s in scores if s['variant']==variant]
        errors=sum(s['errors'] for s in selected);chars=sum(s['reference_chars'] for s in selected)
        totals[variant]=dict(errors=errors,reference_chars=chars,cer=errors/chars,filler_tolerant_errors=sum(s['filler_tolerant_errors'] for s in selected),filler_tolerant_reference_chars=sum(s['filler_tolerant_reference_chars'] for s in selected))
        totals[variant]['filler_tolerant_cer']=totals[variant]['filler_tolerant_errors']/totals[variant]['filler_tolerant_reference_chars']
    result=dict(normalization='NFKC; remove Unicode punctuation and whitespace',formula='Levenshtein character edit distance / reference characters',notes='Each clip counted once, not three repetitions. Clips overlap: not independent speech samples. User reviewed all text and supplied corrections.',scores=scores,totals=totals)
    (ROOT/'q8_clips/cer.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for s in scores:
        if s['variant']=='q8':print(f"{s['clip']}: {s['errors']}/{s['reference_chars']} = {s['cer']:.2%}; reference: {s['reference']}")
    print('Totals:',totals)

if __name__=='__main__':main()
