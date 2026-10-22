"""Offline failure-path checks; no network or model startup."""
import json
import unittest
from unittest.mock import patch
import httpx
import llm_backend as llm

class FallbackTests(unittest.TestCase):
    def run_case(self, cloud, stream=False, local_error=False):
        seen=[]
        def local(request):
            seen.append(json.loads(request.content))
            self.assertNotIn('authorization', request.headers)
            if local_error:
                return httpx.Response(503)
            if stream:
                return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"本地回复"}}]}\n\ndata: [DONE]\n\n')
            return httpx.Response(200, json={'choices':[{'message':{'content':'本地回复'}}]})
        with httpx.Client(transport=httpx.MockTransport(cloud)) as c, httpx.Client(transport=httpx.MockTransport(local)) as l, patch.object(llm,'_cloud_client',c), patch.object(llm,'_client',l), patch.object(llm,'ensure_server',return_value='http://local'), patch.object(llm,'cloud_settings',return_value=('https://api.deepseek.com','flash','test-key')), patch.dict(llm.CURRENT_CONFIG,{'llm_backend':'deepseek'}):
            result=llm.chat(model='',messages=[{'role':'user','content':'你好'}],stream=stream)
            result=list(result) if stream else result
        return result, seen

    def test_http_errors(self):
        for status in (401,429,502):
            for stream in (False,True):
                with self.subTest(status=status,stream=stream):
                    result,seen=self.run_case(lambda request:httpx.Response(status),stream)
                    self.assertEqual(len(seen),1)
    def test_timeout(self):
        def timeout(request):raise httpx.ReadTimeout('test',request=request)
        result,seen=self.run_case(timeout)
        self.assertEqual(result['message']['content'],'本地回复')
    def test_empty_nonstream(self):
        self.run_case(lambda request:httpx.Response(200,json={'choices':[{'message':{'content':''}}]}))
    def test_partial_stream_continues(self):
        result,seen=self.run_case(lambda request:httpx.Response(200,text='data: {"choices":[{"delta":{"content":"开头"}}]}\n\n'),True)
        self.assertEqual([c['message']['content'] for c in result],['开头','本地回复'])
        self.assertEqual(seen[0]['messages'][-2],{'role':'assistant','content':'开头'})
    def test_empty_stream(self):
        result,seen=self.run_case(lambda request:httpx.Response(200,text='data: [DONE]\n\n'),True)
        self.assertEqual(len(seen),1)
    def test_both_fail(self):
        with self.assertRaises(httpx.HTTPStatusError):
            self.run_case(lambda request:httpx.Response(502),local_error=True)
    def test_success_does_not_fallback(self):
        result,seen=self.run_case(lambda request:httpx.Response(200,json={'choices':[{'message':{'content':'云端回复'}}]}))
        self.assertFalse(seen)
    def test_cancel_does_not_fallback(self):
        def stream(*args,**kwargs):
            yield {'message':{'content':'开头'}}
            raise AssertionError('must not resume after cancellation')
        with patch.object(llm,'_request',side_effect=stream) as call, patch.dict(llm.CURRENT_CONFIG,{'llm_backend':'deepseek'}):
            result=llm.chat(model='',messages=[],stream=True)
            next(result);result.close()
            self.assertEqual(call.call_count,1)

if __name__=='__main__':unittest.main()
