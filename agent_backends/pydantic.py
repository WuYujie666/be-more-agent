"""Pydantic AI typed output adapter; no async bridge needed for complete decisions."""
import httpx
from agent_decisions import TurnDecision
from agent_adapters import record_usage, async_count_request


def decide(messages, tools, settings):
    import asyncio
    from openai import AsyncOpenAI
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
    from pydantic_ai.usage import UsageLimits
    url, model, key = settings
    history = []
    for message in messages:
        if message['role'] == 'user':
            history.append(ModelRequest(parts=[UserPromptPart(message['content'])]))
        elif message['role'] == 'assistant':
            history.append(ModelResponse(parts=[TextPart(message['content'])]))
    instructions = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
    async def request():
        async with httpx.AsyncClient(timeout=12, trust_env=not url.startswith('http://127.0.0.1'),
                                     event_hooks={'request': [lambda r: async_count_request(tools, r)]}) as transport:
            async with AsyncOpenAI(api_key=key, base_url=url, http_client=transport, max_retries=0) as client:
                agent = Agent(OpenAIChatModel(model, provider=OpenAIProvider(openai_client=client)),
                              instructions=instructions, output_type=TurnDecision, retries=0,
                              model_settings={'max_tokens': 600, 'parallel_tool_calls': False,
                                              'extra_body': {'thinking': {'type': 'disabled'}}})
                result = await agent.run(message_history=history, usage_limits=UsageLimits(request_limit=1))
                usage = result.usage
                record_usage(tools, usage.input_tokens, usage.output_tokens, usage.cache_read_tokens)
                return result.output
    return asyncio.run(request())
