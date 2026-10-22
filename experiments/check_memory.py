"""Synthetic Mem0 checks; --live invokes DeepSeek. Never uses personal memory."""
import argparse,json,resource,statistics,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory_backend import MemoryService

def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true',required=True)
    parser.parse_args()
    output={'synthetic':True,'mem0_version':'2.2.1','checks':{},'times':{}}
    with tempfile.TemporaryDirectory(prefix='memory-eval-',dir='runtime') as d:
        m=MemoryService(d,user_id='synthetic')
        try:
            start=time.perf_counter();m.ready.result(60)
            output['times']['startup']=time.perf_counter()-start
            def add(text):
                start=time.perf_counter();r=m.remember(text,'test-session').result(60)
                output['times'].setdefault('add',[]).append(time.perf_counter()-start)
                return r
            add('请记住，我叫林小雨，我现在住在北京，睡前最喜欢听雨声。')
            rows=m.submit('list').result(10)['results']
            output['initial_records']=rows
            output['checks']['extract_name']=any('林小雨' in r['memory'] for r in rows)
            output['checks']['semantic_recall']=any('雨' in r['memory'] for r in m.search('我睡觉的时候偏爱什么声音？'))
            add('我已经从北京搬到上海了，现在住上海。')
            output['after_move']=m.search('我现在住在哪个城市？')
            output['checks']['new_location_retrieved']=any('上海' in r['memory'] for r in output['after_move'])
            import llm_backend
            from memory_backend import memory_context
            response=llm_backend.chat(model='unused', messages=[memory_context(output['after_move']), {'role':'user','content':'只回答城市名称：我现在住在哪里？'}],options={'num_predict':50})['message']['content']
            output['current_location_answer']=response
            output['checks']['current_location_answer']='上海' in response and '北京' not in response
            # Explicit API update verifies SQLite history independently of LLM inference.
            rows=m.submit('list').result(10)['results']
            target=next(r for r in rows if '林小雨' in r['memory'] and '北京' in r['memory'])
            m.submit('update',memory_id=target['id'],text='用户叫林小雨，偏好被称呼为小雨。').result(10)
            hist=m.submit('history',memory_id=target['id']).result(10)
            output['checks']['update_history']=any(r['event']=='UPDATE' for r in hist)
            m.close();m=MemoryService(d,user_id='synthetic');m.ready.result(60)
            output['checks']['restart_persistence']=any('小雨' in r['memory'] for r in m.search('我叫什么名字？'))
            times=[]
            for _ in range(10):
                start=time.perf_counter();m.search('我睡前喜欢听什么？');times.append(time.perf_counter()-start)
            output['times']['search_median']=statistics.median(times)
            output['times']['search_max']=max(times)
            m.submit('delete',memory_id=target['id']).result(15)
            output['checks']['delete_history']=not m.store.history(target['id'])
            m.submit('clear').result(30)
            output['checks']['clear_records']=not m.submit('list').result(10)['results']
            output['checks']['clear_sql']=all(m.store.db.connection.execute('SELECT count(*) FROM '+t).fetchone()[0]==0 for t in ('history','messages'))
            m.close();m=MemoryService(d,user_id='synthetic');m.ready.result(60)
            output['checks']['clear_survives_restart']=not m.search('我叫什么，住在哪里？')
            m.close()
            # Real ChatEngine across two fresh sessions, with the production prompt.
            from chat_engine import ChatEngine
            from config import CURRENT_CONFIG
            from unittest.mock import patch
            from test_memory import IO
            with patch.dict(CURRENT_CONFIG, {'long_term_memory':True,'memory_dir':d,
                                            'summaries_file':str(Path(d)/'summary.json')}):
                # This test directory already belongs to synthetic; use the same
                # identity through the service factory without touching real data.
                with patch('chat_engine.MemoryService',side_effect=lambda directory: MemoryService(directory,user_id='synthetic')):
                    engine=ChatEngine(IO());engine.memory.ready.result(40)
                    try:
                        engine.chat_and_respond('请记住，我睡前最喜欢听雨声。')
                        engine.finalize_session_memory()
                        engine.memory.submit('flush').result(40)
                    finally:engine.close()
                    engine=ChatEngine(IO());engine.memory.ready.result(40)
                    try:
                        answer=engine.chat_and_respond('我睡前最喜欢听什么声音？')
                        output['chat_engine_answer']=answer
                        output['checks']['chat_engine_cross_session']='雨' in answer
                        engine._handle_forget('清空记忆')
                        output['checks']['chat_engine_forget']=not engine.memory.submit('list').result(10)['results']
                    finally:engine.close()
            output['rss_peak_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        finally:
            m.close()
            Path('experiments/memory_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))
    assert all(output['checks'].values()), output['checks']

if __name__=='__main__':run()
