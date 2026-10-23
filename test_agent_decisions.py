"""Decision contract tests: receipts, invalid output, scope and cancellation."""
import json
import unittest
from companion_tools import CompanionTools
from agent_decisions import run, TurnDecision
from experiments.benchmark_agents import FakeMemory


def actions(*items):
    return {'kind':'actions','reply':'','actions':list(items)}


class DecisionTests(unittest.TestCase):
    def test_success_comes_from_execution_not_model_reply(self):
        edits=[]
        tools=CompanionTools(FakeMemory(),on_edit=lambda:edits.append(True))
        result=list(run([],tools,lambda *_:actions({'operation':'forget_recent'})))
        self.assertEqual(edits,[True])
        self.assertIn('已清除', ''.join(result))
        self.assertTrue(tools.audit[-1]['response']['ok'])

    def test_tool_failure_cannot_produce_success_receipt(self):
        tools=CompanionTools(None)
        result=list(run([],tools,lambda *_:actions({'operation':'forget_recent'})))
        self.assertEqual(result,['这项操作没能完成。'])
        self.assertFalse(tools.audit[-1]['response']['ok'])

    def test_invalid_output_retried_before_any_execution(self):
        tools=CompanionTools(FakeMemory());decisions=iter([
            {'kind':'actions','reply':'已经删除','actions':[{'operation':'forget_recent'}]},
            {'kind':'clarify','reply':'你希望清除哪部分？','actions':[]}])
        result=list(run([],tools,lambda *_:next(decisions)))
        self.assertEqual(tools.audit,[])
        self.assertEqual(result,['你希望清除哪部分？'])

    def test_invalid_output_retry_is_bounded(self):
        tools=CompanionTools(FakeMemory())
        with self.assertRaises(RuntimeError):list(run([],tools,lambda *_:{}))
        self.assertEqual(tools.audit,[])

    def test_search_then_delete_with_real_id(self):
        memory=FakeMemory();memory.rows=[{'id':'home','memory':'用户住在北京'}]
        tools=CompanionTools(memory)
        def decide(history, scoped):
            if not scoped.audit:return actions({'operation':'search','query':'住址'})
            self.assertIn('home',history[-1]['content'])
            return actions({'operation':'forget','memory_id':'home'})
        self.assertIn('已删除',''.join(run([],tools,decide)))
        self.assertEqual(memory.rows,[])

    def test_unknown_id_rejected(self):
        tools=CompanionTools(FakeMemory())
        self.assertEqual(list(run([],tools,lambda *_:actions({'operation':'forget','memory_id':'invented'}))),['这项操作没能完成。'])

    def test_batch_stops_after_failure_and_reports_partial_success(self):
        tools=CompanionTools(FakeMemory())
        result=list(run([],tools,lambda *_:actions(
            {'operation':'settings','key':'minutes','value':'10'},
            {'operation':'settings','key':'volume','value':'2'},
            {'operation':'settings','key':'minutes','value':'99'})))
        self.assertEqual(tools.settings['minutes'],10)
        self.assertEqual(len(tools.audit),2)
        self.assertEqual(result[-1],'这项操作没能完成。')

    def test_cancelled_turn_does_not_mutate(self):
        tools=CompanionTools(FakeMemory(),cancelled=lambda:True)
        self.assertEqual(list(run([],tools,lambda *_:actions({'operation':'forget_recent'}))),[])
        self.assertEqual(tools.audit,[])

    def test_search_and_forget_cannot_share_unverified_decision(self):
        with self.assertRaises(ValueError):TurnDecision.model_validate(actions({'operation':'search'}, {'operation':'forget','memory_id':'guess'}))


if __name__=='__main__':unittest.main()
