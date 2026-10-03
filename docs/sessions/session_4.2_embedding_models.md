# Session 4.2: Embedding Models & Cross-Encoders

## 🎯 Learning Objectives

By the end of this session, you will:
- Explain the difference between bi-encoders (embeddings) and cross-encoders (rerankers)
- Read the internals of `EmbeddingModelSingleton` and `CrossEncoderModelSingleton`
- Understand why the embedding dimension drives Qdrant collection sizing
- Compare embedding models on the project's own data
- Choose the right device and batch size on a 16 GB GPU

---

## 🏗️ Architecture Overview

### Two Models, Two Jobs

```
┌────────────────────────────────────────────────────────────────────┐
│  Retrieval stage 1: BI-ENCODER (fast, approximate)                  │
│                                                                      │
│   query ──► [encoder] ──► q_vec ─┐                                   │
│                                  ├──► cosine(q_vec, d_vec) → top-N   │
│   doc   ──► [encoder] ──► d_vec ─┘        (vectors precomputed)      │
│                                                                      │
│   Encodes query and document INDEPENDENTLY → all doc vectors can be  │
│   indexed and compared with a matrix multiply. Scales to millions.   │
├────────────────────────────────────────────────────────────────────┤
│  Retrieval stage 2: CROSS-ENCODER (slow, accurate)                  │
│                                                                      │
│   [query] [SEP] [doc] ──► [transformer] ──► relevance score          │
│                                                                      │
│   Scores one (query, doc) pair at a time. No precomputation → used   │
│   only to rerank a small candidate set (top-N → top-k).              │
└────────────────────────────────────────────────────────────────────┘
```

| Property | Bi-Encoder (`EmbeddingModelSingleton`) | Cross-Encoder (`CrossEncoderModelSingleton`) |
|----------|----------------------------------------|----------------------------------------------|
| Input | one text | a `(query, doc)` pair |
| Output | a vector | a scalar score |
| Precompute docs | Yes | No |
| Scale | Millions | Tens to hundreds per query |
| Used in project | Indexing + first-pass search | `Reranker.generate` |
| Speed | Fast | Slow (linear in candidate count) |
| Default model | `all-MiniLM-L6-v2` | `cross-encoder/ms-marco-MiniLM-L-4-v2` |

---

## 📁 Key Files Explained

### 1. `EmbeddingModelSingleton` - The Bi-Encoder

```python
# llm_engineering/application/networks/embeddings.py
class EmbeddingModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = settings.TEXT_EMBEDDING_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model_id = model_id
        self._device = device

        self._model = SentenceTransformer(
            self._model_id,
            device=self._device,
            cache_folder=str(cache_dir) if cache_dir else None,
        )
        self._model.eval()
```

**Public surface**:

```python
    @property
    def model_id(self) -> str:
        return self._model_id

    @cached_property
    def embedding_size(self) -> int:
        dummy_embedding = self._model.encode("")
        return dummy_embedding.shape[0]

    @property
    def max_input_length(self) -> int:
        return self._model.max_seq_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._model.tokenizer

    def __call__(self, input_text: str | list[str], to_list: bool = True):
        try:
            embeddings = self._model.encode(input_text)
        except Exception:
            logger.error(f"Error generating embeddings for {self._model_id=} and {input_text=}")
            return [] if to_list else np.array([])

        if to_list:
            embeddings = embeddings.tolist()
        return embeddings
```

**Key Concepts**:
- **`SentenceTransformer.encode` pipeline**: tokenize → transformer forward pass → **mean pooling** over token embeddings → optional L2 normalization. The default `all-MiniLM-L6-v2` produces **384-dim** vectors, and cosine similarity is used downstream.
- **`embedding_size` probes the model with `""`**: a cheap, single call that returns the true output dimension. `_create_collection` uses it as `VectorParams(size=...)`, so the index can never disagree with the model.
- **`max_seq_length`** (256 for MiniLM) is the token budget; this is why chunking uses `SentenceTransformersTokenTextSplitter` with `tokens_per_chunk=embedding_model.max_input_length` (Session 2.2).
- **CPU by default**: `RAG_MODEL_DEVICE=cpu`. MiniLM is small enough that CPU is fine for indexing, and it avoids competing with training for VRAM.
- **Error tolerance**: a bad input returns an empty result rather than raising, so a single malformed chunk cannot kill a batch.

**Dimension cheat-sheet**:

| Model | Dim | Max tokens | Notes |
|-------|-----|-----------|-------|
| `all-MiniLM-L6-v2` (default) | 384 | 256 | Fast, small, good baseline |
| `all-mpnet-base-v2` | 768 | 384 | Higher quality, ~2× cost |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | Strong retrieval, heavier |
| `intfloat/e5-large-v2` | 1024 | 512 | Needs `query:`/`passage:` prefixes |

> Changing `TEXT_EMBEDDING_MODEL_ID` changes the vector dimension. Existing Qdrant collections sized for 384 dims must be recreated; the stored `metadata.embedding_size` reveals the mismatch.

---

### 2. `CrossEncoderModelSingleton` - The Reranker

```python
class CrossEncoderModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = settings.RERANKING_CROSS_ENCODER_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
    ) -> None:
        self._model_id = model_id
        self._device = device

        self._model = CrossEncoder(
            model_name=self._model_id,
            device=self._device,
        )
        self._model.model.eval()

    def __call__(self, pairs: list[tuple[str, str]], to_list: bool = True):
        scores = self._model.predict(pairs)
        return scores.tolist() if to_list else scores
```

**Key Concepts**:
- **`CrossEncoder` wraps a transformer with a regression head.** It concatenates `query` and `doc` with the model's separator tokens, runs a forward pass, and outputs one score per pair.
- **`predict` accepts a list of pairs**, so batching still applies: the `Reranker` sends all candidates at once. This is the "batch reranking" speedup: one call for N pairs instead of N calls.
- **`use_vector_index` in Qdrant is irrelevant here**; the cross-encoder never touches the vector store.
- Default model `ms-marco-MiniLM-L-4-v2` is trained for passage reranking (MS MARCO) and scores how well a passage answers a query.

---

### 3. How the Reranker Uses the Cross-Encoder

```python
# llm_engineering/application/rag/reranking.py
class Reranker(RAGStep):
    def __init__(self, mock: bool = False) -> None:
        super().__init__(mock=mock)
        self._model = CrossEncoderModelSingleton()

    @opik.track(name="Reranker.generate")
    def generate(self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        if self._mock:
            return chunks

        query_doc_tuples = [(query.content, chunk.content) for chunk in chunks]
        scores = self._model(query_doc_tuples)

        scored_query_doc_tuples = list(zip(scores, chunks, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]
        return reranked_documents
```

**Key Concepts**:
- **`zip(scores, chunks, strict=False)`** pairs each score with its chunk; then a stable sort by score descending keeps the top `keep_top_k`.
- **The `@opik.track` decorator** wraps the call for prompt monitoring (Session 7.2), so rerank input/output is traced.
- **`mock=True`** returns candidates unchanged, letting the RAG pipeline be tested without downloading the cross-encoder.

---

### 4. Where Each Model Is Wired In

```python
# settings.py
TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
RAG_MODEL_DEVICE: str = "cpu"
```

```
EmbeddingModelSingleton  ←  embedding_data_handlers.py (indexing chunks, embedding queries)
EmbeddingModelSingleton  ←  domain/base/vector.py (_create_collection sizes the collection)
CrossEncoderModelSingleton ← reranking.py (rerank candidates)
```

---

## 🔬 Deep Dive: Similarity Mathematics

### Cosine similarity

```
cos(q, d) = (q · d) / (||q|| · ||d||)
```

Qdrant collections are created with `Distance.COSINE`. `all-MiniLM-L6-v2` normalizes embeddings, so cosine reduces to a dot product and ranking is monotonic.

### Why reranking helps

The bi-encoder compresses each text into a fixed vector **before** seeing the query. The cross-encoder sees the query and the passage **together**, so attention can align the specific query terms with passage spans. That extra interaction is why the cross-encoder is more accurate, and why it is too expensive to run over a whole corpus.

### The two-stage tradeoff

```
15 candidates  ──bi-encoder recall──►  high recall, medium precision
       │
       ▼ cross-encoder
top-3          ──►  high precision  →  better context → better generation
```

---

## 🛠️ Hands-On: Benchmark Embedding Models

### Step 1: Measure encoding throughput

```python
import time
from llm_engineering.application.networks import EmbeddingModelSingleton

model = EmbeddingModelSingleton()
texts = ["How does RAG work?"] * 64

start = time.perf_counter()
_ = model(texts, to_list=True)
elapsed = time.perf_counter() - start
print(f"{len(texts)/elapsed:.1f} texts/sec on {model._device}")
```

### Step 2: Compare candidate models

```python
from sentence_transformers import SentenceTransformer

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
for model_id in MODELS:
    m = SentenceTransformer(model_id)
    vec = m.encode("How does RAG work?")
    print(f"{model_id:50s} dim={vec.shape[0]} max_tokens={m.max_seq_length}")
```

### Step 3: Rerank a candidate set and inspect score spread

```python
from llm_engineering.application.networks import CrossEncoderModelSingleton

cross = CrossEncoderModelSingleton()
pairs = [
    ("How does RAG work?", "RAG combines retrieval with generation."),
    ("How does RAG work?", "The Eiffel Tower is in Paris."),
    ("How does RAG work?", "Retrieval-augmented generation grounds LLMs in documents."),
]
print(list(zip([p[1] for p in pairs], cross(pairs))))
```

The relevant passages should score clearly higher. A small spread signals that the candidate set is off-topic or the model is wrong for the domain.

---

## 📝 Exercise: Swap the Embedding Model End to End

### Task

Replace the embedding model and rebuild the vector index.

1. Set `TEXT_EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2` in `.env`.
2. Delete the existing Qdrant collections (or use a new `DATABASE_NAME`).
3. Re-run feature engineering.
4. Confirm the new collection dimension is **768**, not 384.
5. Re-run a RAG query and note whether retrieved chunks look more relevant.

**Goal**: Experience the ripple effect: model id → dimension → collection size → re-index. This is the most common upgrade and the most common pitfall.

---

## 🐛 Common Pitfalls

- **Dimension mismatch after a model swap**: Qdrant rejects points whose vector length differs from the collection. Always recreate collections when `embedding_size` changes.
- **`max_input_length` truncation**: if you hard-code a token splitter size instead of `embedding_model.max_input_length`, long chunks get silently truncated by the encoder, losing text.
- **Cross-encoder cost**: reranking 100 candidates is ~100 forward passes. Keep the candidate set small (the project retrieves `k // 3` per category, then reranks).
- **Device contention on 16 GB**: running the cross-encoder on GPU while training competes for VRAM. Keep `RAG_MODEL_DEVICE=cpu` during training runs.

---

## 🎓 Knowledge Check

1. **Why can a bi-encoder scale but a cross-encoder cannot?**
   - Answer: Bi-encoders precompute document vectors independently; cross-encoders must process each (query, doc) pair at query time.

2. **What determines the Qdrant collection vector size?**
   - Answer: `EmbeddingModelSingleton.embedding_size`, probed from the model.

3. **What does `max_input_length` control in the pipeline?**
   - Answer: The token splitter's `tokens_per_chunk`, guaranteeing chunks fit the encoder.

4. **How does `Reranker.generate` order documents?**
   - Answer: It scores all (query, chunk) pairs with the cross-encoder and sorts by score descending.

5. **Why is the default embedding device CPU?**
   - Answer: The model is small, and CPU avoids competing with training for VRAM.

6. **What is stored on each vector that helps detect a model change?**
   - Answer: `metadata.embedding_model_id` and `metadata.embedding_size`.

---

## 🔗 Next Session

**Session 5.1**: Supervised Fine-Tuning (SFT)

We move from retrieval to training and fine-tune Llama 3.1 8B with LoRA on the instruction dataset.

---

## 📚 Additional Resources

- [SentenceTransformers](https://www.sbert.net/)
- [Cross-Encoders in SentenceTransformers](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MS MARCO Reranking Models](https://huggingface.co/cross-encoder)
- [Qdrant Distance Metrics](https://qdrant.tech/documentation/concepts/collections/#distance-metrics)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 2.2, 2.3, 4.1

**Outcome**: You can explain, benchmark, and swap embedding and reranking models, and predict the effect on the vector index.
