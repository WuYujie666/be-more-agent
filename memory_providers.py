"""Small Mem0 provider adjustments for this Pi and the existing DeepSeek API."""
from pathlib import Path
import httpx
from openai import OpenAI, APIError
from fastembed import TextEmbedding
from mem0.embeddings.base import EmbeddingBase
from mem0.embeddings.fastembed import FastEmbedEmbedding
from mem0.llms.deepseek import DeepSeekLLM

ROOT = Path(__file__).resolve().parent


class PiEmbedding(FastEmbedEmbedding):
    def __init__(self, config=None):
        EmbeddingBase.__init__(self, config)
        self.dense_model = TextEmbedding(
            model_name=self.config.model, threads=2,
            cache_dir=str(ROOT / 'cache/memory-models'),
            local_files_only=True, providers=['CPUExecutionProvider'])
        self.config.embedding_dims = self.dense_model.embedding_size


class FlashMemoryLLM(DeepSeekLLM):
    def __init__(self, config=None):
        super().__init__(config)
        self.client = self.client.with_options(timeout=30, max_retries=0)

    def generate_response(self, *args, **kwargs):
        from llm_backend import ensure_server
        kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
        try:
            result = super().generate_response(*args, **kwargs)
            if not result:
                raise ValueError('Empty memory response')
            return result
        except (APIError, ValueError) as exc:
            print(f'[MEMORY] cloud failed ({type(exc).__name__}); trying local Qwen', flush=True)
        original = self.client
        kwargs.pop('extra_body', None)
        with httpx.Client(trust_env=False, timeout=120) as transport:
            with OpenAI(base_url=ensure_server() + '/v1', api_key='local',
                        http_client=transport, max_retries=0) as local:
                self.client = local
                try:
                    return super().generate_response(*args, **kwargs)
                finally:
                    self.client = original


from mem0.vector_stores.qdrant import Qdrant


class PiQdrant(Qdrant):
    def _get_bm25_encoder(self):
        # This first Chinese configuration uses semantic retrieval only. Avoid
        # lazily downloading an English BM25 tokenizer during a voice turn.
        return None
