"""Shared LLM adapter for conversation, warmup and summaries.
Only a server started by this process is stopped on exit.
"""
import atexit
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import httpx
from config import CURRENT_CONFIG

_ROOT = Path(__file__).resolve().parent
_lock = threading.Lock()
_process = None
_client = httpx.Client(trust_env=False, timeout=120)
_cloud_client = httpx.Client(timeout=httpx.Timeout(12, connect=5))


def cloud_settings():
    values = {}
    path = _ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('\"\'')
    def get(key, default=''):
        return os.environ.get(key, values.get(key, default))
    key = get('CLOUD_LLM_API_KEY')
    if not key:
        raise ValueError('Fill CLOUD_LLM_API_KEY in .env before using DeepSeek')
    url = get('CLOUD_LLM_BASE_URL', 'https://api.deepseek.com').rstrip('/')
    if url not in ('https://api.deepseek.com', 'https://api.deepseek.com/v1'):
        raise ValueError('DeepSeek backend requires the official HTTPS API endpoint')
    return url, get('CLOUD_LLM_MODEL', 'deepseek-flash'), key


def ensure_server():
    global _process
    url = CURRENT_CONFIG.get('llama_url', 'http://127.0.0.1:18080').rstrip('/')
    with _lock:
        try:
            if _client.get(url + '/health', timeout=1).is_success:
                return url
        except httpx.HTTPError:
            pass
        if not CURRENT_CONFIG.get('llama_autostart', False):
            raise RuntimeError(f'llama.cpp is unavailable: {url}')
        if url != 'http://127.0.0.1:18080':
            raise ValueError('Managed llama.cpp requires http://127.0.0.1:18080')
        if _process is None or _process.poll() is not None:
            binary = _ROOT / CURRENT_CONFIG['llama_binary']
            model = _ROOT / CURRENT_CONFIG['llama_model']
            if not binary.is_file() or not model.is_file():
                raise FileNotFoundError('Missing llama.cpp executable or GGUF model; see README')
            log_path = _ROOT / 'cache/llama-server.log'
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open('ab') as log:
                _process = subprocess.Popen([
                    str(binary), '-m', str(model), '-t', '4', '-tb', '4',
                    '-c', str(CURRENT_CONFIG.get('llama_context', 4096)),
                    '-np', '1', '-ngl', '0', '-fa', 'off',
                    '--host', '127.0.0.1', '--port', '18080',
                ], cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(120):
            if _process.poll() is not None:
                raise RuntimeError('llama.cpp exited; inspect cache/llama-server.log')
            try:
                if _client.get(url + '/health', timeout=1).is_success:
                    return url
            except httpx.HTTPError:
                pass
            time.sleep(.25)
        raise TimeoutError('llama.cpp startup timed out; inspect cache/llama-server.log')


def chat(*, model, messages, stream=False, options=None, keep_alive=None):
    backend = CURRENT_CONFIG.get('llm_backend', 'deepseek')
    if backend not in ('llama.cpp', 'deepseek'):
        raise ValueError(f'Unknown LLM backend: {backend}')
    def request(selected, history):
        return _request(selected, history, stream, options)
    if backend == 'llama.cpp':
        return request(backend, messages)
    if not stream:
        try:
            return request('deepseek', messages)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            print(f'[LLM] cloud failed ({type(exc).__name__}); falling back to local Qwen', flush=True)
            return request('llama.cpp', messages)

    def fallback_stream():
        emitted = ''
        try:
            yield_stream = request('deepseek', messages)
            try:
                for chunk in yield_stream:
                    emitted += chunk['message']['content']
                    yield chunk
            finally:
                yield_stream.close()
            return
        except (httpx.HTTPError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            print(f'[LLM] cloud failed ({type(exc).__name__}); falling back to local Qwen', flush=True)
        history = list(messages)
        if emitted:
            history += [{'role': 'assistant', 'content': emitted},
                        {'role': 'user', 'content': '连接中断，请紧接上一段继续，只输出剩余内容，不要重复已经说过的话。'}]
        local = request('llama.cpp', history)
        try:
            yield from local
        finally:
            local.close()
    return fallback_stream()


def _request(backend, messages, stream, options=None):
    options = dict(options or {})
    headers = {}
    client = _client
    if backend == 'deepseek':
        url, cloud_model, key = cloud_settings()
        endpoint = url + '/chat/completions'
        headers = {'Authorization': 'Bearer ' + key}
        client = _cloud_client
    else:
        endpoint = ensure_server() + '/v1/chat/completions'
    body = dict(messages=messages, stream=stream, cache_prompt=True,
                temperature=options.get('temperature', .7),
                top_k=options.get('top_k', 40), top_p=options.get('top_p', .9),
                min_p=0, max_tokens=options.get('num_predict', 256))
    if 'seed' in options:
        body['seed'] = options['seed']
    if backend == 'deepseek':
        body = dict(model=cloud_model, messages=messages, stream=stream,
                    max_tokens=options.get('num_predict', 256),
                    thinking={'type': 'disabled'})
    if not stream:
        response = client.post(endpoint, json=body, headers=headers)
        response.raise_for_status()
        message = response.json()['choices'][0]['message']
        if not message.get('content'):
            raise RuntimeError('Empty model response')
        return {'message': message}

    def chunks():
        complete = False
        received = False
        with client.stream('POST', endpoint, json=body, headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    complete = True
                    break
                event = json.loads(data)
                if 'error' in event:
                    raise RuntimeError(str(event['error']))
                for choice in event.get('choices', []):
                    if choice.get('finish_reason') is not None:
                        complete = True
                    content = choice.get('delta', {}).get('content')
                    if content:
                        received = True
                        yield {'message': {'content': content}}
        if not complete or not received:
            raise RuntimeError('Incomplete or empty model stream')
    return chunks()


def shutdown():
    global _process
    with _lock:
        if _process is not None:
            if _process.poll() is None:
                _process.terminate()
                try:
                    _process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _process.kill()
                    _process.wait()
            _process = None


atexit.register(shutdown)
