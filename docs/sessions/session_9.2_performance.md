# Session 9.2: Performance Optimization

## 🎯 Learning Objectives

By the end of this session, you will:
- Identify every batching knob in the pipeline and what it controls
- Reason about VRAM, throughput, and latency tradeoffs
- Apply caching at the ZenML and application levels
- Understand quantization choices across training, serving, and evaluation
- Profile a workload before optimizing it

---

## 🏗️ Architecture Overview

Performance in this project is governed by a few recurring patterns:

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Performance levers                            │
│                                                                        │
│  1. Batching        ── embed=10, load=4, generate=24, eval=5           │
│  2. Parallelism     ── ThreadPoolExecutor across queries/collections   │
│  3. Caching         ── ZenML step cache + cached_property              │
│  4. Quantization    ── 4-bit training, 8-bit serving, vLLM eval        │
│  5. Device choice   ── RAG_MODEL_DEVICE=cpu vs cuda                    │
│  6. Retrieval size  ── k and k//3 per collection                       │
└──────────────────────────────────────────────────────────────────────┘
```

**Rule: measure before optimizing.** Every claim below is a hypothesis until you profile the actual workload.

---

## 📁 Batching Knobs in the Code

### 1. Embedding batch - `chunk_and_embed` (batch = 10)

```python
# steps/feature_engineering/rag.py
        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
```

**Tradeoff**: larger batches raise GPU utilization and throughput but raise peak activation memory. On a 16 GB card with a large embedding model, drop this to 4 or use CPU.

### 2. Qdrant load batch - `load_to_vector_db` (batch = 4)

```python
# steps/feature_engineering/load_to_vector_db.py
        for documents_batch in utils.misc.batch(documents, size=4):
            document_class.bulk_insert(documents_batch)
```

**Tradeoff**: point upserts are network-bound. Small batches keep request payloads modest and avoid timeouts; larger batches reduce round-trips. Four is conservative.

### 3. LLM generation batch - `generate` (batch = 24)

```python
# llm_engineering/application/dataset/generation.py
            batches = utils.misc.batch(langchain_category_prompts, size=24)
            for batch in batches:
                batched_dataset_samples = chain.batch(batch, stop=None)
```

**Tradeoff**: `chain.batch` issues concurrent LLM requests. Higher batch = higher throughput until you hit OpenAI rate limits (HTTP 429). If you see rate-limit errors, lower this first.

### 4. Evaluation threads and batch

```python
# llm_engineering/model/evaluation/evaluate.py
def evaluate_answers(model_id: str, num_threads: int = 10, batch_size: int = 5) -> Dataset:
    ...
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
```

**Tradeoff**: 10 concurrent judge calls balances throughput and rate limits; batch size 5 groups samples per thread. Tune against your OpenAI tier.

### 5. Parallel I/O - `query_data_warehouse`

```python
# steps/feature_engineering/query_data_warehouse.py
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }
```

Three Mongo queries run concurrently; wall-clock is the slowest query, not the sum.

### 6. Parallel retrieval - `ContextRetriever.search`

```python
# llm_engineering/application/rag/retriever.py
        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in n_generated_queries]
```

Each query variant searches in parallel, and within a variant the three collections are searched serially. Increasing `expand_to_n_queries` multiplies parallel work but also latency and cost.

---

## 🔬 Deep Dive: The Batch-Size Tradeoff

```
Throughput
    ▲
    │            ╭───────  GPU utilization saturates
    │        ╭───╯
    │    ╭───╯
    │╭───╯
    └────────────────────────► batch size
        │           │
     latency     memory
     floor       ceiling (OOM)
```

For GPU embedding/training, increasing batch size improves throughput until the GPU saturates; beyond that, only memory and latency grow. For API calls, increasing concurrency improves throughput until rate limits trigger retries, which can make it *slower*.

**On the RTX 5000 (16 GB, Turing)**:
- **No bfloat16** - `is_bfloat16_supported()` returns `False`, so training uses fp16. fp16 is fine but has a narrower dynamic range; keep the learning rate reasonable.
- **Gradient checkpointing** (not enabled in this code) trades compute for memory and would help fit larger effective batches.
- **`RAG_MODEL_DEVICE=cpu`** keeps embedding/reranking off the GPU so training has the whole card. Switch to `cuda` only when not training.

---

## 📁 Caching Layers

### 1. ZenML step cache

Every pipeline run has `enable_cache` (default `True`; `--no-cache` disables). Unchanged steps reuse outputs, which is the biggest saving during iterative development.

```python
    pipeline_args = {"enable_cache": not no_cache}
```

### 2. `cached_property` for model metadata

```python
# embeddings.py
    @cached_property
    def embedding_size(self) -> int:
        dummy_embedding = self._model.encode("")
        return dummy_embedding.shape[0]
```

The model is probed once; repeated collection sizing calls are free.

### 3. Singleton model loading

`EmbeddingModelSingleton` and `CrossEncoderModelSingleton` load once per process. This is the difference between a 5-second and a 50-second startup across a batch of operations.

### 4. Application-level semantic caching (not implemented)

A production RAG service often caches `query → answer` with a TTL. This project does not, so repeated identical queries re-run the full path. Adding a `TTLCache` in front of `rag()` is the highest-value next optimization (see the exercise).

---

## 📁 Quantization Across the Lifecycle

| Stage | Technique | Where | Effect |
|-------|-----------|-------|--------|
| Training | 4-bit NF4 base + LoRA | `load_in_4bit=True` | ~4× smaller base weights |
| Training optimizer | 8-bit AdamW | `optim="adamw_8bit"` | smaller optimizer state |
| Serving | 8-bit bitsandbytes | `HF_MODEL_QUANTIZE="bitsandbytes"` | fits endpoint GPU |
| Evaluation | vLLM (fp16/tp) | `vllm.LLM` | high-throughput generation |
| Embeddings | fp32/fp16 CPU | `RAG_MODEL_DEVICE=cpu` | small model, no GPU contention |

Quantization reduces memory at some quality cost. Match the technique to the constraint: 4-bit for fitting 8B training, 8-bit for serving with a quality margin.

---

## 🛠️ Hands-On: Profile Before Optimizing

### Step 1: Time the embedding step

```python
import time
from llm_engineering.application.networks import EmbeddingModelSingleton

model = EmbeddingModelSingleton()
texts = ["Sample text " * 50] * 64

for bs in (1, 10, 50):
    start = time.perf_counter()
    for i in range(0, len(texts), bs):
        model(texts[i:i+bs], to_list=True)
    print(f"batch={bs:3d}  {time.perf_counter()-start:.3f}s")
```

### Step 2: Measure peak VRAM (GPU)

```python
import torch
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
torch.cuda.reset_peak_memory_stats()
m.encode([""] * 256)
print(torch.cuda.max_memory_allocated() / 1024**2, "MB")
```

### Step 3: Time retrieval

```python
import time
from llm_engineering.application.rag.retriever import ContextRetriever

r = ContextRetriever(mock=False)
start = time.perf_counter()
_ = r.search("How does RAG work?", k=3)
print(f"retrieval: {time.perf_counter()-start:.2f}s")
```

### Step 4: Break down latency

Use the Opik trace (Session 7.2) to see where the seconds go: SelfQuery, QueryExpansion, search, rerank, or generation.

---

## 📝 Exercise: Add Semantic Answer Caching

### Task

Cache RAG answers to eliminate repeated work.

```python
from cachetools import TTLCache, cached

# in the API module
_rag_cache = TTLCache(maxsize=1000, ttl=3600)

@cached(cache=_rag_cache)
def rag(query: str) -> str:
    ...
```

1. Add `cachetools` to dependencies.
2. Wrap `rag()`.
3. Measure the latency of a repeated query before and after.
4. Discuss invalidation: when the corpus changes, cached answers go stale. Add a version key (for example a collection hash) to the cache key.

**Goal**: Experience the biggest single latency win in a RAG service - and its correctness cost.

---

## 🐛 Common Pitfalls

- **Premature optimization**: raising batch sizes without measuring often causes OOM or rate-limit retries that are slower. Measure first.
- **GPU contention**: embedding and reranking on GPU during training starves the trainer. Keep `RAG_MODEL_DEVICE=cpu` while training.
- **Cache staleness in ZenML**: a "finished quickly" pipeline run may have skipped steps. Use `--no-cache` after data changes.
- **bfloat16 assumption**: Turing does not support it; forcing bf16 fails or falls back silently. Use `is_bfloat16_supported()`.
- **Unbounded `k`**: large `k` increases reranking and generation cost superlinearly. Start small.

---

## 🎓 Knowledge Check

1. **What are the four batching values in the pipeline?**
   - Answer: embed=10, load-to-vector=4, LLM generate=24, evaluation batch=5 with 10 threads.

2. **What is the first knob to lower on OpenAI 429 errors?**
   - Answer: the generation batch size (24).

3. **Why is `embedding_size` a `cached_property`?**
   - Answer: It requires a model probe; caching avoids repeating it.

4. **Why keep `RAG_MODEL_DEVICE=cpu` during training?**
   - Answer: To avoid VRAM contention with the training job.

5. **What precision does the RTX 5000 use for training, and why?**
   - Answer: fp16, because Turing lacks bfloat16 support.

6. **What is the highest-value unimplemented optimization for the RAG service?**
   - Answer: Semantic answer caching with a TTL and a corpus-version key.

---

## 🔗 Next Session

**Session 9.3**: Security Best Practices

We harden secrets, the API, and the local stack.

---

## 📚 Additional Resources

- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [vLLM](https://docs.vllm.ai/)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes)
- [cachetools](https://cachetools.readthedocs.io/)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 2.3, 5.1, 6.2, 8.3

**Outcome**: You can identify the performance levers in the project, profile a workload, and apply the right optimization at the right layer.
