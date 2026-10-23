"""Three replaceable agent loops with one stream contract: yield plain text.

Framework imports are lazy. Tools/business rules live in companion_tools.
No persistent framework history/checkpointer: ChatEngine owns the session.
"""
import asyncio
import copy
import json
import queue
import threading
from companion_tools import POLICY, time_context
import llm_backend

MAX_REQUESTS = 4


def record_usage(tools, prompt=0, completion=0, cached=0):
    if hasattr(tools, 'metrics'):
        tools.metrics.setdefault('usage', []).append({'input': prompt, 'output': completion, 'cached': cached})


def count_request(tools):
    def hook(request):
        if tools.cancelled():
            raise asyncio.CancelledError()
        if hasattr(tools, 'metrics'):
            tools.metrics['requests'] = tools.metrics.get('requests', 0) + 1
        if hasattr(tools, 'budget'):
            tools.budget(request)
    return hook


async def async_count_request(tools, request):
    count_request(tools)(request)



def stream(messages, tools, backend='native'):
    """Bridge cancellable framework execution into the synchronous TTS consumer."""
    if backend not in ('native', 'langchain', 'pydantic'):
        raise ValueError('Unknown agent_backend')
    from config import CURRENT_CONFIG
    history = [m for m in messages if m['role'] == 'system'] + [time_context(), {'role': 'system', 'content': POLICY}, tools.context()] + [m for m in messages if m['role'] != 'system']
    if CURRENT_CONFIG.get('llm_backend') == 'llama.cpp':
        yield from _local(history)
        return
    tools = copy.copy(tools)  # Per-run cancellation; shared settings/pending/audit containers.
    q = queue.Queue(maxsize=32)
    cancelled = threading.Event()
    previous_cancelled = tools.cancelled
    tools.cancelled = lambda: cancelled.is_set() or previous_cancelled()
    audit_start = len(tools.audit)
    def emit(value):
        while not cancelled.is_set():
            try:
                q.put(value, timeout=.1)
                return
            except queue.Full:
                pass
        raise asyncio.CancelledError()
    def worker():
        try:
            settings = llm_backend.cloud_settings()
            from importlib import import_module
            adapter = import_module('agent_backends.' + backend)
            from agent_decisions import run
            for text in run(history, tools, lambda messages, scoped: adapter.decide(messages, scoped, settings), MAX_REQUESTS):
                emit(text)
            emit(None)
        except (Exception, asyncio.CancelledError) as exc:
            if not cancelled.is_set():
                emit(exc)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    emitted = ''
    try:
        while True:
            if previous_cancelled():
                return
            try:
                item = q.get(timeout=.1)
            except queue.Empty:
                continue
            if item is None:
                if emitted:
                    return
                item = RuntimeError('Empty agent reply')
            if isinstance(item, BaseException):
                if getattr(tools, 'strict_errors', False):
                    raise item
                print(f'[AGENT] {backend} failed ({type(item).__name__}); local text fallback', flush=True)
                extra = [{'role': 'system', 'content': '本地降级模式不能调用工具，不得声称已执行未成功的操作。已执行结果：' +
                          json.dumps(tools.audit[audit_start:], ensure_ascii=False)}]
                if emitted:
                    extra += [{'role': 'assistant', 'content': emitted},
                              {'role': 'user', 'content': '连接中断，只继续剩余话语，不重复。'}]
                yield from _local(history + extra)
                return
            emitted += item
            yield {'message': {'content': item}}
    finally:
        cancelled.set()
        thread.join(timeout=.2)
        # Each run owns a new tools cancellation closure; late execution stays blocked.


def _local(messages):
    messages = list(messages) + [{'role': 'system', 'content': '本地降级模式：工具不可用，不要声称改动了记忆或设置；可以正常陪聊。'}]
    result = llm_backend._request('llama.cpp', messages, True, {'num_predict': 256})
    try:
        yield from result
    finally:
        result.close()
