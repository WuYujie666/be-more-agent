"""LangChain structured model adapter; execution and receipts stay shared."""
import httpx
from agent_decisions import TurnDecision
from agent_adapters import count_request, record_usage


def decide(messages, tools, settings):
    from langchain_openai import ChatOpenAI
    url, model, key = settings
    with httpx.Client(timeout=12, trust_env=not url.startswith('http://127.0.0.1'),
                      event_hooks={'request': [count_request(tools)]}) as client:
        model = ChatOpenAI(model=model, api_key=key, base_url=url, http_client=client,
                           timeout=12, max_retries=0, max_tokens=600, http_socket_options=(),
                           extra_body={'thinking': {'type': 'disabled'}})
        result = model.with_structured_output(TurnDecision, method='function_calling', include_raw=True).invoke(messages)
        usage = result['raw'].usage_metadata or {}
        record_usage(tools, usage.get('input_tokens', 0), usage.get('output_tokens', 0),
                     usage.get('input_token_details', {}).get('cache_read', 0))
        if result['parsing_error']:
            raise ValueError('Invalid structured decision') from result['parsing_error']
        return result['parsed']
