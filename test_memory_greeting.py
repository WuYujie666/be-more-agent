"""Opening continuity: same Mem0 records, conservative dates, no summary fallback."""
from concurrent.futures import Future
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from chat_engine import ChatEngine
from config import CURRENT_CONFIG
from test_memory import IO

NOW = datetime.fromisoformat('2026-10-22T20:00:00+08:00')


def record(text='用户在准备答辩。', age=1, **metadata):
    return {'id': 'recent', 'memory': text, 'metadata': {
        'source': 'user_transcript', 'session_id': 'previous',
        'observed_at': (NOW - timedelta(days=age)).isoformat(), **metadata}}


class Store:
    def __init__(self, rows=(), plan=None, fail=False):
        self.rows=list(rows); self.plan=plan; self.fail=fail
        self.ready=Future(); self.ready.set_result(True)
    def submit(self, operation, **args):
        future=Future()
        if self.fail:
            future.set_exception(RuntimeError('unavailable'))
        elif operation == 'list':future.set_result({'results':self.rows})
        elif operation == 'followup':future.set_result(self.plan)
        else:raise AssertionError(operation)
        return future


class GreetingTests(unittest.TestCase):
    def greet(self, store, backend='native'):
        with patch.dict(CURRENT_CONFIG, {'long_term_memory':True, 'agent_backend':backend}), \
                patch('chat_engine.MemoryService', return_value=store), \
                patch('companion_tools.now', return_value=NOW):
            engine=ChatEngine(IO())
            return engine, engine.build_greeting()

    def test_ordinary_facts_do_not_trigger_opening(self):
        _,(text,used)=self.greet(Store([record()]))
        self.assertFalse(used);self.assertNotIn('答辩',text)

    def test_empty_and_unavailable_use_neutral_greeting(self):
        for store in [Store(),Store(fail=True)]:
            with self.subTest(store=store):
                _,(text,used)=self.greet(store)
                self.assertFalse(used);self.assertEqual(text,'晚上好。今天有什么想分享的吗？')

    def test_old_style_preference_does_not_suppress_plan(self):
        pref=record(kind='preference',key='questions',value='fewer')
        _,(text,used)=self.greet(Store([record(),pref],plan='答辩'))
        self.assertTrue(used);self.assertIn('答辩',text)

    def test_plan_priority_preserved(self):
        _,(text,used)=self.greet(Store([record()],plan='答辩'))
        self.assertTrue(used);self.assertIn('现在想聊聊这件事吗',text)
        self.assertNotIn('已经完成',text)

    def test_deleted_record_not_replaced_by_legacy_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            old=Path(temp)/'summaries.json';old.write_text('[{"summary":"旧秘密"}]')
            with patch.dict(CURRENT_CONFIG,{'summaries_file':str(old)}):
                store=Store([record()],plan='答辩');self.assertTrue(self.greet(store)[1][1])
                store.rows.clear();store.plan=None
                _,(text,used)=self.greet(store)
                self.assertFalse(used);self.assertNotIn('旧秘密',text)
                self.assertTrue(old.exists()) # Legacy data is never read or mutated.

    def test_same_opening_across_backends(self):
        answers=[self.greet(Store([record()]),b)[1] for b in ['native','langchain','pydantic','legacy']]
        self.assertTrue(all(a==answers[0] for a in answers))


if __name__ == '__main__':unittest.main()
