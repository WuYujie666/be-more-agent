"""Offline tests for cancellation, tool ownership and the shared fallback boundary."""
import threading
import time
import unittest
from unittest.mock import patch
import httpx
from companion_tools import CompanionTools
from agent_adapters import stream
from experiments.benchmark_agents import FakeMemory

class RuntimeTests(unittest.TestCase):
    def test_cancel_cannot_mutate_and_next_turn_is_usable(self):
        tools=CompanionTools(FakeMemory())
        attempted=threading.Event()
        def fake(messages, scoped, settings):
            yield '第一段'
            time.sleep(.35)
            scoped.execute(dict(operation='settings',key='minutes',value='99'))
            attempted.set()
            yield '后续'
        with patch('agent_decisions.run',side_effect=lambda messages, scoped, decide, limit: fake(messages, scoped, None)),patch('llm_backend.cloud_settings',return_value=('test','test','test')):
            output=stream([{'role':'user','content':'你好'}],tools,'native')
            next(output);output.close()
            self.assertTrue(attempted.wait(2))
        self.assertNotIn('minutes',tools.settings)
        self.assertFalse(tools.cancelled())

    def test_partial_failure_preserves_action_results_without_replay(self):
        tools=CompanionTools(FakeMemory())
        def fake(messages, scoped, settings):
            scoped.execute(dict(operation='settings',key='minutes',value='10'))
            yield '已经设置'
            raise httpx.ReadTimeout('fake')
        local_messages=[]
        def local(messages):
            local_messages.extend(messages)
            yield {'message':{'content':'，晚安。'}}
        with patch('agent_decisions.run',side_effect=lambda messages, scoped, decide, limit: fake(messages, scoped, None)),patch('llm_backend.cloud_settings',return_value=('test','test','test')),patch('agent_adapters._local',side_effect=local):
            result=list(stream([{'role':'user','content':'播放十分钟'}],tools,'native'))
        self.assertEqual(len(tools.audit),1)
        self.assertIn('已经设置',str(local_messages))
        self.assertIn('minutes',str(local_messages))
        self.assertEqual(len(result),2)

    def test_privacy_request_is_not_intercepted_by_phrase_matching(self):
        called=[]
        tools=CompanionTools(FakeMemory(),on_edit=lambda:called.append(True))
        received=[]
        def fake(messages, scoped, settings):
            received.extend(messages)
            yield '模型回复'
        with patch('agent_decisions.run',side_effect=lambda messages, scoped, decide, limit: fake(messages, scoped, None)),patch('llm_backend.cloud_settings',return_value=('test','test','test')):
            result=list(stream([{'role':'user','content':'刚才这次聊天的内容别存进记忆。'}],tools,'native'))
        self.assertEqual(called, [])
        self.assertEqual(tools.audit, [])
        self.assertIn('刚才这次聊天的内容别存进记忆。',str(received))
        self.assertEqual(result[0]['message']['content'],'模型回复')

    def test_clock_is_explicit_timezone(self):
        from companion_tools import time_context
        self.assertIn('+08:00',time_context()['content'])

if __name__=='__main__':unittest.main()
