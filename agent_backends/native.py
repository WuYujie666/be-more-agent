"""Direct SDK baseline: forced structured decision, shared application executor."""
import json
import httpx
from agent_decisions import TurnDecision
from agent_adapters import count_request, record_usage


def decide(messages, tools, settings):
    from openai import OpenAI
    url, model, key = settings
    schema = {'type': 'function', 'function': {'name': 'turn_decision',
              'description': 'Return the semantic decision for this turn.',
              'parameters': TurnDecision.model_json_schema()}}
    with OpenAI(api_key=key, base_url=url, max_retries=0,
                http_client=httpx.Client(timeout=12, trust_env=not url.startswith('http://127.0.0.1'),
                                         event_hooks={'request': [count_request(tools)]})) as client:
        result = client.chat.completions.create(model=model, messages=messages, tools=[schema],
                    tool_choice={'type': 'function', 'function': {'name': 'turn_decision'}},
                    parallel_tool_calls=False, max_tokens=600,
                    extra_body={'thinking': {'type': 'disabled'}})
        if result.usage:
            details = result.usage.prompt_tokens_details
            record_usage(tools, result.usage.prompt_tokens, result.usage.completion_tokens,
                         details.cached_tokens if details else 0)
        calls = result.choices[0].message.tool_calls or []
        if len(calls) != 1 or calls[0].function.name != 'turn_decision':
            raise ValueError('Expected one structured decision')
        return json.loads(calls[0].function.arguments)
