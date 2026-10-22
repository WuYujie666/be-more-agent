"""Small Mem0 provider adjustments for this Pi and the existing DeepSeek API."""
from pathlib import Path
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
        kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
        return super().generate_response(*args, **kwargs)


from mem0.vector_stores.qdrant import Qdrant


class PiQdrant(Qdrant):
    def _get_bm25_encoder(self):
        # This first Chinese configuration uses semantic retrieval only. Avoid
        # lazily downloading an English BM25 tokenizer during a voice turn.
        return None
