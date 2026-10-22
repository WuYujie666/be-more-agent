"""One Chinese LoCoMo conversation; session ingestion; 30 stratified questions.
Paid API requests require --live. Hard admission budget uses peak pricing,
UTF-8 byte input upper bounds and maximum output tokens; usage refunds reserve.
Dataset/official judge prompt hashes are pinned. Never opens production memory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import statistics
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'cache/locomo-zh'
OUT = ROOT / 'experiments/locomo_session'
DB = ROOT / 'runtime/locomo-session-benchmark'
USER = 'locomo-conv26-session-pilot'
HASHES = {'zh.json':'3e3d9aa70c59c35b0254553314e15a0df4a340a2aea7d3781b95c7d1944c7f5d',
          'prompts.py':'8ebac1ef60e9ab5caf99079fdaac038b85472e81491ed35e2d2655f3927c76c2'}


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


class Budget:
    def __init__(self, path, limit=2):
        self.path, self.limit = path, limit
        self.rows = json.loads(path.read_text()) if path.exists() else []
        self.phase = 'unknown'

    @property
    def spent(self):
        return sum(r['peak_cny_reserved_or_actual'] for r in self.rows)

    def request(self, request):
        body = json.loads(request.content)
        # Token count cannot exceed UTF-8 byte count for the text tokenizer.
        # Count the entire JSON envelope plus a chat-format margin as well.
        upper_input = len(request.content) + 1024 + 64 * len(body.get('messages', []))
        upper_output = body.get('max_tokens', 4000)
        reserve = (upper_input * 2 + upper_output * 8) / 1_000_000
        if self.spent + reserve > self.limit:
            raise RuntimeError('Budget admission limit: next request could exceed 2 CNY')
        request.extensions['budget_index'] = len(self.rows)
        self.rows.append(dict(phase=self.phase, status='reserved',
                              peak_cny_reserved_or_actual=reserve,
                              reserved_input_tokens=upper_input,
                              reserved_output_tokens=upper_output))
        save(self.path, self.rows)

    def response(self, response):
        index = response.request.extensions['budget_index']
        response.read()
        if not response.is_success:
            self.rows[index]['status'] = f'http_{response.status_code}'
            save(self.path, self.rows)
            return  # Unknown billing: keep the conservative reservation.
        payload = response.json()
        usage = payload.get('usage')
        if not usage:
            raise RuntimeError('No usage returned; stop before spending further')
        prompt = usage['prompt_tokens']; completion = usage['completion_tokens']
        hits = usage.get('prompt_cache_hit_tokens', 0)
        cost = ((prompt-hits)*2 + hits*.04 + completion*8)/1_000_000
        self.rows[index].update(status='complete', usage=usage,
                                peak_cny_reserved_or_actual=cost,
                                idle_cny_estimate=cost/2,
                                model=payload.get('model'),
                                finish_reason=payload['choices'][0].get('finish_reason'))
        save(self.path, self.rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, digest in HASHES.items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest
    entry = json.loads((DATA/'zh.json').read_text())[0]
    conv = entry['conversation']
    spec = importlib.util.spec_from_file_location('locomo_prompts', DATA/'prompts.py')
    prompts = importlib.util.module_from_spec(spec); spec.loader.exec_module(prompts)
    rng = random.Random(20261022)
    selected = []
    for category, count in {1:6, 2:7, 3:3, 4:14}.items():
        selected += rng.sample([i for i,q in enumerate(entry['qa']) if q['category']==category], count)
    selected.sort()
    state_path = OUT/'results.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        protocol='Chinese community translation; session-level ingestion; pilot, not official leaderboard',
        dataset_revision='e209267490c686980b29c1bd9571a01edcde1877', hashes=HASHES,
        conversation=entry['sample_id'], turns=419, sessions_expected=19,
        question_indices=selected, excluded_category=5, top_k=4, threshold=.45,
        output_token_limits={'ingest':4000,'answer':500,'judge':256},
        ingest=[], answers=[], complete=False)
    assert state['question_indices'] == selected
    budget = Budget(OUT/'usage.json')
    import httpx
    from memory_backend import create_store
    from llm_backend import cloud_settings
    store = create_store(DB)
    url, model, key = cloud_settings()
    from openai import OpenAI
    transport = httpx.Client(timeout=60, event_hooks={'request':[budget.request], 'response':[budget.response]})
    client = OpenAI(api_key=key, base_url=url, http_client=transport, max_retries=0)
    store.llm.client.close()
    store.llm.client = client
    store.llm.config.max_tokens = 4000  # Whole benchmark sessions exceed a five-turn bedtime chat.
    started = time.perf_counter()
    try:
        sessions = sorted((k for k,v in conv.items() if k.startswith('session_') and isinstance(v,list)),
                          key=lambda k:int(k.split('_')[1]))
        for session in sessions:
            if any(x['session']==session for x in state['ingest']):continue
            date = conv[session+'_date_time']
            stamp = datetime.strptime(date, '%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc).isoformat()
            messages = []
            for t in conv[session]:
                # Both speakers in LoCoMo are humans. Preserve name/date; never
                # label one as an assistant and silently discard their facts.
                text = f"[会话时间：{date}] {t['speaker']}：{t['text']}"
                if t.get('blip_caption'):text += ' [图片描述：'+t['blip_caption']+']'
                if t.get('query'):text += ' [图片搜索描述：'+t['query']+']'
                messages.append({'role':'user', 'content':text})
            budget.phase = 'ingest'
            before = len(budget.rows); start = time.perf_counter()
            result = store.add(messages,user_id=USER,
                               metadata={'session_id':session,'created_at':stamp,
                                         'source':'LoCoMo Chinese','source_turn_ids':[t['dia_id'] for t in conv[session]]})
            if any(r.get('finish_reason')=='length' for r in budget.rows[before:]):
                raise RuntimeError('Extraction truncated; stop, do not silently score incomplete ingestion')
            state['ingest'].append(dict(session=session,turns=len(messages),
                                       seconds=time.perf_counter()-start,results=result['results']))
            save(state_path,state)
            print(f"INGEST {len(state['ingest'])}/19: {len(result['results'])} memories; peak-price cost {budget.spent:.4f} CNY",flush=True)
        state['memories'] = store.get_all(filters={'user_id':USER},top_k=10000)['results']
        save(state_path,state)
        def generate(messages, limit, json_mode=False):
            kwargs = dict(model=model,messages=messages,max_tokens=limit,
                          extra_body={'thinking':{'type':'disabled'}})
            if json_mode:kwargs['response_format']={'type':'json_object'}
            response = client.chat.completions.create(**kwargs)
            if response.choices[0].finish_reason == 'length':
                raise RuntimeError('Answer/judge truncated; inspect before continuing')
            return response.choices[0].message.content
        for qi in selected:
            if any(x['index']==qi for x in state['answers']):continue
            q = entry['qa'][qi]
            start = time.perf_counter()
            rows = store.search(q['question'],filters={'user_id':USER},top_k=4,threshold=.45)['results']
            search_s = time.perf_counter()-start
            # Scoring labels and evidence IDs are NEVER passed to retrieval or answerer.
            budget.phase = 'answer'
            prompt = prompts.get_answer_generation_prompt(q['question'],rows,reference_date='October 22, 2023')
            start = time.perf_counter()
            answer = generate([{'role':'user','content':prompt+'\n请直接用中文给出简短最终答案，以 ANSWER: 开头，不输出推理过程。'}],500)
            answer_s = time.perf_counter()-start
            answer = answer.split('ANSWER:')[-1].strip()
            budget.phase = 'judge'
            gold = prompts.preprocess_answer(q['category'],str(q['answer']))
            judge = json.loads(generate([
                {'role':'system','content':prompts.JUDGE_SYSTEM_PROMPT},
                {'role':'user','content':prompts.get_judge_prompt(q['category'],q['question'],gold,answer)}],256,True))
            if judge.get('label') not in ('CORRECT','WRONG'):raise ValueError('Invalid judge label')
            state['answers'].append(dict(index=qi,category=q['category'],question=q['question'],
                                          gold=gold,evidence=q.get('evidence',[]),answer=answer,judge=judge,
                                          retrieved=rows,search_seconds=search_s,answer_seconds=answer_s))
            save(state_path,state)
            print(f"QA {len(state['answers'])}/30: {judge['label']}; peak-price cost {budget.spent:.4f} CNY",flush=True)
        state['complete'] = True
    finally:
        state['run_seconds'] = state.get('run_seconds',0)+time.perf_counter()-started
        state['rss_peak_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        state['peak_price_cost_or_reserved_cny'] = budget.spent
        state['idle_price_estimated_cny'] = sum(r.get('idle_cny_estimate',r['peak_cny_reserved_or_actual']) for r in budget.rows)
        state['api_calls'] = len(budget.rows)
        state['database_bytes'] = sum(p.stat().st_size for p in DB.rglob('*') if p.is_file())
        save(state_path,state)
        store.vector_store.client.close();store.db.close();client.close()
    print('COMPLETE',sum(x['judge']['label']=='CORRECT' for x in state['answers']),'/30',flush=True)

if __name__=='__main__':main()
