"""Offline regression checks for memory integration and forgetting races."""
from concurrent.futures import Future
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock
from memory_backend import MemoryService, memory_context
from chat_engine import ChatEngine
from config import CURRENT_CONFIG


class IO:
    def __init__(self):self.spoken=[]
    def speak(self,text):self.spoken.append(text)
    def __getattr__(self,name):return lambda *a,**kw: None


class MemoryTests(unittest.TestCase):
    def engine(self):
        with patch.dict(CURRENT_CONFIG,{'long_term_memory':False,'agent_backend':'legacy'}):return ChatEngine(IO())

    def test_forget_is_explicit_not_substring(self):
        e=self.engine();e.session_memory=[{'role':'user','content':'我喜欢雨声'}]
        self.assertFalse(e._handle_forget('不要清空记忆'))
        self.assertTrue(e.session_memory)

    def test_session_saved_once_at_end_not_after_each_turn(self):
        e=self.engine();e.memory=Mock()
        e.memory.search.return_value=[]
        e.session_memory=[{'role':'assistant','content':'开场问候'}]
        with patch('chat_engine.llm.chat',side_effect=lambda **kw:iter([{'message':{'content':'好的。'}}])):
            e.chat_and_respond('我搬家了')
            e.chat_and_respond('现在住在上海')
        e.memory.remember_session.assert_not_called()
        e.finalize_session_memory();e.finalize_session_memory();e.close()
        e.memory.remember_session.assert_called_once()
        messages=e.memory.remember_session.call_args.args[0]
        self.assertEqual([m['content'] for m in messages if m['role']=='user'],
                         ['我搬家了','现在住在上海'])
        self.assertNotIn('开场问候',[m['content'] for m in messages])
        self.assertEqual(len(messages),4)

    def test_forget_before_session_end_prevents_later_write(self):
        e=self.engine();e.memory=Mock()
        e.session_memory=[{'role':'user','content':'secret'}]
        with tempfile.TemporaryDirectory() as d, patch.dict(CURRENT_CONFIG,{'summaries_file':str(Path(d)/'summary.json')}):
            e._handle_forget('清空记忆')
            e.close()
        e.memory.remember_session.assert_not_called()

    def test_finalization_does_not_request_separate_summary(self):
        e=self.engine();e.memory=Mock()
        e.session_memory=[{'role':'user','content':'我喜欢雨声'}]
        with patch('chat_engine.llm.chat') as cloud:
            e.finalize_session_memory()
        cloud.assert_not_called()
        e.memory.remember_session.assert_called_once()

    def test_failed_clear_does_not_claim_success(self):
        e=self.engine()
        class Failed:
            def submit(self,*a):
                f=Future();f.set_exception(RuntimeError('offline'));return f
        e.memory=Failed();e._handle_forget('清空记忆')
        self.assertIn('没能',e.io.spoken[-1])

    def test_inflight_write_then_clear_cannot_restore_memory(self):
        started=threading.Event();release=threading.Event();rows=[]
        class Store:
            custom_instructions=""
            llm=Mock(generate_response=lambda **kw: '{"changes":[]}')
            def get_all(self,**kw):return {"results":[]}
            def __init__(self):
                from types import SimpleNamespace
                self.vector_store=SimpleNamespace(client=SimpleNamespace(close=lambda:None))
                self.db=Mock()
                self.db.connection=Mock()
                self.db.connection.__enter__=Mock(return_value=self.db.connection)
                self.db.connection.__exit__=Mock(return_value=False)
            def add(self,*a,**kw):
                started.set();release.wait(3);rows.append('secret');return {'results':[]}
            def delete_all(self,**kw):rows.clear()
        with tempfile.TemporaryDirectory() as d:
            m=MemoryService(d,store_factory=lambda d:Store())
            try:
                m.ready.result(2)
                e=self.engine();e.memory=m
                e.session_memory=[{'role':'user','content':'secret'}]
                e.finalize_session_memory()
                self.assertTrue(started.wait(2))
                t=threading.Thread(target=e._handle_forget,args=('清空记忆',));t.start()
                release.set();t.join(3)
                self.assertFalse(t.is_alive())
                e.finalize_session_memory();m.submit('flush').result(2)
                self.assertEqual(rows,[])
            finally:release.set();m.close()

    def test_search_busy_does_not_wait_for_cloud(self):
        started=threading.Event();release=threading.Event()
        class FakeStore:
            custom_instructions=""
            llm=Mock(generate_response=lambda **kw: '{"changes":[]}')
            def get_all(self,**kw):return {"results":[]}
            def add(self,*a,**kw):started.set();release.wait(3);return {'results':[]}
            def __init__(self):
                from types import SimpleNamespace
                self.vector_store=SimpleNamespace(client=SimpleNamespace(close=lambda:None))
                self.db=SimpleNamespace(close=lambda:None)
        with tempfile.TemporaryDirectory() as d:
            m=MemoryService(d,store_factory=lambda d:FakeStore())
            try:
                m.ready.result(2);f=m.remember('hello','test')
                self.assertTrue(started.wait(2));start=time.monotonic()
                self.assertEqual(m.search('hello',timeout=.02),[])
                self.assertLess(time.monotonic()-start,.2)
                release.set();f.result(2)
            finally:release.set();m.close()

    def test_context_bounded_and_not_persisted_in_prompt(self):
        rows=[{'memory':'雨'*1000} for _ in range(20)]
        context=memory_context(rows)
        self.assertLess(len(context['content']),2200)
        e=self.engine()
        class FakeMemory:
            def search(self,text):return [{'memory':'喜欢雨声'}]
            def remember(self,*args):pass
        e.memory=FakeMemory()
        captured=[]
        def chat(**kwargs):
            captured.append(kwargs['messages'])
            return iter([{'message':{'content':'好的。'}}])
        with patch('chat_engine.llm.chat',side_effect=chat):e.chat_and_respond('今天还好')
        self.assertTrue(any('喜欢雨声' in x['content'] for x in captured[0]))
        self.assertFalse(any('喜欢雨声' in x['content'] for x in e.session_memory))

if __name__=='__main__':unittest.main()
