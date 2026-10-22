"""Download the small Chinese embedding model once; normal runs are offline."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from fastembed import TextEmbedding
model = TextEmbedding('BAAI/bge-small-zh-v1.5', threads=2,
                      cache_dir=str(ROOT / 'cache/memory-models'),
                      providers=['CPUExecutionProvider'])
assert len(next(model.embed(['中文记忆测试']))) == 512
print('Chinese embedding model ready. Normal memory queries run locally.')
