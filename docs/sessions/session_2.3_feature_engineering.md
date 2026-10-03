# Session 2.3: Feature Engineering Pipeline

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand `EmbeddingModelSingleton` and the thread-safe `SingletonMeta`
- Embed chunks in batches with `EmbeddingDispatcher`
- Load embedded documents into Qdrant collection-by-collection
- Read the `feature_engineering` ZenML pipeline end to end
- Reason about batching, device placement, and idempotency

---

## 🏗️ Architecture Overview

### The Feature Engineering Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                    feature_engineering pipeline                     │
│                                                                      │
│  query_data_warehouse(author_full_names)                            │
│        │  raw docs from MongoDB (articles + posts + repositories)   │
│        ▼                                                            │
│  clean_documents(raw_documents)                                     │
│        │  CleanedDocument (Session 2.2)                             │
│        ├──────────────────────────────► load_to_vector_db ◄── cleaned
│        │                                        │                    │
│        ▼                                        │                    │
│  chunk_and_embed(cleaned_documents)             │                    │
│        │  ChunkingDispatcher → chunks            │                    │
│        │  EmbeddingDispatcher (batch=10)         │                    │
│        ▼                                        │                    │
│  load_to_vector_db(embedded_documents) ─────────┘                    │
│        │  group_by_class + batch(size=4) + bulk_insert               │
│        ▼                                                            │
│  Qdrant: embedded_articles / embedded_posts / embedded_repositories │
└────────────────────────────────────────────────────────────────────┘
```

Notice the pipeline loads **two** sets of documents into Qdrant: the cleaned documents (unindexed payload store) and the embedded chunks (searchable vectors).

---

## 📁 Key Files Explained

### 1. `llm_engineering/application/networks/base.py` - Thread-Safe Singleton Metaclass

**Purpose**: A metaclass that guarantees one instance per class across threads.

```python
# llm_engineering/application/networks/base.py
from threading import Lock
from typing import ClassVar


class SingletonMeta(type):
    """This is a thread-safe implementation of Singleton."""

    _instances: ClassVar = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]
```

**Key Concepts**:
- **Double-checked locking with an explicit `Lock`**: two threads entering `__call__` simultaneously cannot race to build two models.
- **`_instances` is keyed by class**, so different subclasses get separate instances.
- **`__init__` arguments are ignored after the first call.** This is why the project never re-constructs models with different settings at runtime; change `.env` and restart instead.
- Contrast with `MongoDatabaseConnector` (Session 1.3), which uses the simpler `__new__` singleton. The model loader needs the lock because model construction is slow and non-atomic.

---

### 2. `llm_engineering/application/networks/embeddings.py` - The Embedding Model

**Purpose**: A singleton wrapper around a `SentenceTransformer`.

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

    def __call__(
        self, input_text: str | list[str], to_list: bool = True
    ) -> NDArray[np.float32] | list[float] | list[list[float]]:
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
- **`eval()` and no gradients**: this is an inference-only model. No `torch.no_grad()` needed because `SentenceTransformer.encode` already does it.
- **`embedding_size` is a `cached_property`**: it probes the model once by encoding `""` and caches the dimension. That value sizes every Qdrant collection.
- **`max_input_length` feeds chunking** (Session 2.2): chunks are guaranteed to fit the model's `max_seq_length`.
- **`__call__` swallows errors and returns empty**; the embedding handler then produces fewer documents rather than crashing the step.
- Default model is `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 256-token max), controllable via `TEXT_EMBEDDING_MODEL_ID` and `RAG_MODEL_DEVICE`.

**Cross-encoder sibling** (used later in Session 4.1/4.2):

```python
class CrossEncoderModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = settings.RERANKING_CROSS_ENCODER_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
    ) -> None:
        self._model = CrossEncoder(model_name=self._model_id, device=self._device)
        self._model.model.eval()

    def __call__(self, pairs: list[tuple[str, str]], to_list: bool = True):
        scores = self._model.predict(pairs)
        return scores.tolist() if to_list else scores
```

---

### 3. `llm_engineering/application/preprocessing/embedding_data_handlers.py` - Embedding Handlers

**Purpose**: Turn chunks into embedded chunks and attach embedding provenance metadata.

```python
# llm_engineering/application/preprocessing/embedding_data_handlers.py
from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.chunks import ArticleChunk, Chunk, PostChunk, RepositoryChunk
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk, EmbeddedChunk, EmbeddedPostChunk, EmbeddedRepositoryChunk,
)
from llm_engineering.domain.queries import EmbeddedQuery, Query

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [data_model.content for data_model in data_model]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_model, embeddings, strict=False)
        ]
        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass
```

**Concrete mapper (article)**:

```python
class ArticleEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: ArticleChunk, embedding: list[float]) -> EmbeddedArticleChunk:
        return EmbeddedArticleChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
```

**The query handler**:

```python
class QueryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Query, embedding: list[float]) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=data_model.id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            content=data_model.content,
            embedding=embedding,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
```

**Key Concepts**:
- **`embed` vs `embed_batch`**: `embed` is a convenience wrapper; `embed_batch` is the real path and encodes a whole list in one model call (far faster than one call per chunk on GPU).
- **The chunk `id` is preserved** into the embedded chunk, so re-embedding overwrites the same Qdrant point.
- **Provenance metadata** (`embedding_model_id`, `embedding_size`, `max_input_length`) is stored on every vector. If you change the embedding model, the collection dimension check and this metadata reveal the stale vectors.
- **`QueryEmbeddingHandler` produces an `EmbeddedQuery` that is never persisted**; it is embedded on the fly at query time and used only as the search vector.
- **`strict=False`** in `zip` tolerates a shorter/longer embedding list without raising, which pairs with the error-swallowing `__call__`.

---

### 4. `EmbeddingDispatcher` - Batch Entry Point

```python
# dispatchers.py (excerpt)
class EmbeddingDispatcher:
    factory = EmbeddingHandlerFactory

    @classmethod
    def dispatch(
        cls, data_model: VectorBaseDocument | list[VectorBaseDocument]
    ) -> VectorBaseDocument | list[VectorBaseDocument]:
        is_list = isinstance(data_model, list)
        if not is_list:
            data_model = [data_model]

        if len(data_model) == 0:
            return []

        data_category = data_model[0].get_category()
        assert all(
            data_model.get_category() == data_category for data_model in data_model
        ), "Data models must be of the same category."
        handler = cls.factory.create_handler(data_category)

        embedded_chunk_model = handler.embed_batch(data_model)

        if not is_list:
            embedded_chunk_model = embedded_chunk_model[0]

        logger.info("Data embedded successfully.", data_category=data_category)

        return embedded_chunk_model
```

**Key Concepts**:
- **Accepts a single document or a list**, returning the matching shape. This keeps call sites clean.
- **The `assert` enforces homogeneous categories.** All documents in one batch must belong to the same Qdrant collection, otherwise the resulting vectors would collide across collections.
- Handles the `QUERIES` category too, so the RAG retriever can embed a user query with the same code path.

---

### 5. `steps/feature_engineering/query_data_warehouse.py` - Fetch Raw Data

**Purpose**: Pull all raw documents for the given authors out of MongoDB in parallel.

```python
# steps/feature_engineering/query_data_warehouse.py
@step
def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = []
    authors = []
    for author_full_name in author_full_names:
        first_name, last_name = utils.split_user_full_name(author_full_name)
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)

        results = fetch_all_data(user)
        user_documents = [doc for query_result in results.values() for doc in query_result]
        documents.extend(user_documents)
    ...
    return documents


def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }
        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")
                results[query_name] = []
    return results
```

**Key Concepts**:
- **Three parallel Mongo queries** (`articles`, `posts`, `repositories`) via `ThreadPoolExecutor`, fan-in through `as_completed`.
- **`get_or_create`** makes the step safe to re-run: an unknown author is created once, then reused.
- **`split_user_full_name`** handles multi-word first names (`"Paul Iusztin"` → `("Paul", "Iusztin")`; `"Maxime Labonne"` → `("Maxime", "Labonne")`).

---

### 6. `steps/feature_engineering/chunk_and_embed.py`

**Purpose**: Chunk then embed in batches.

```python
# steps/feature_engineering/rag.py
@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    metadata = {"chunking": {}, "embedding": {}, "num_documents": len(cleaned_documents)}

    embedded_chunks = []
    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])

        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)

    metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"])
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="embedded_documents", metadata=metadata)

    return embedded_chunks
```

**Key Concepts**:
- **`utils.misc.batch(chunks, 10)`**: embed 10 chunks per model call. This is the memory/speed dial. Larger batches are faster on GPU but use more VRAM.
- **Rich ZenML metadata**: chunk counts and unique authors per category are attached to the step output, visible in the ZenML dashboard.
- The step returns plain `EmbeddedChunk` objects; persistence is a separate step.

---

### 7. `steps/feature_engineering/load_to_vector_db.py` - Persist

**Purpose**: Group by concrete class and upsert in small batches.

```python
# steps/feature_engineering/load_to_vector_db.py
@step
def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> Annotated[bool, "successful"]:
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")
        for documents_batch in utils.misc.batch(documents, size=4):
            try:
                document_class.bulk_insert(documents_batch)
            except Exception:
                logger.error(f"Failed to insert documents into {document_class.get_collection_name()}")
                return False
    return True
```

**Key Concepts**:
- **`group_by_class`** fans a mixed list into `cleaned_*`, `embedded_*`, etc. Each group goes to its own Qdrant collection.
- **Batch size 4** is deliberately small; Qdrant point upserts are network-bound and `bulk_insert` already creates the collection on first use.
- **Returns a bool**, so the pipeline can assert success downstream.

---

### 8. `pipelines/feature_engineering.py` - Orchestration

```python
# pipelines/feature_engineering.py
from zenml import pipeline
from steps import feature_engineering as fe_steps


@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    return [last_step_1.invocation_id, last_step_2.invocation_id]
```

**Key Concepts**:
- **DAG shape**: `query_data_warehouse → clean_documents → {load_to_vector_db, chunk_and_embed → load_to_vector_db}`.
- **`wait_for`** wires pipeline dependencies: `end_to_end_data` passes the ETL step invocation IDs so feature engineering runs only after crawling completes.
- **Returns invocation IDs** so the parent `end_to_end_data` pipeline can chain `generate_datasets` after it.

---

## 🛠️ Hands-On: Run Feature Engineering

### Step 1: Prerequisites

```bash
docker compose up -d          # mongo + qdrant
python -m tools.run --run-feature-engineering --no-cache
```

### Step 2: Inspect the collections

```python
from llm_engineering.infrastructure.db.qdrant import connection

for c in connection.get_collections().collections:
    info = connection.get_collection(c.name)
    print(c.name, info.points_count)
```

### Step 3: Embed and search a query

```python
from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.embedded_chunks import EmbeddedArticleChunk

model = EmbeddingModelSingleton()
query_vector = model("How does RAG work?", to_list=True)

results = EmbeddedArticleChunk.search(query_vector=query_vector, limit=3)
for r in results:
    print(r.author_full_name, "|", r.content[:80])
```

### Step 4: Confirm provenance metadata

```python
print(results[0].metadata)
# {'embedding_model_id': 'sentence-transformers/all-MiniLM-L6-v2', 'embedding_size': 384, ...}
```

---

## 📝 Exercise: Tune Batching

### Task

Measure the effect of batch size on embedding throughput.

1. Wrap the embedding loop in a timer.
2. Run with `batch(chunks, 1)`, `batch(chunks, 10)`, and `batch(chunks, 50)`.
3. Record wall-clock time and (if GPU is used) peak VRAM.
4. Write down the tradeoff.

**Goal**: Understand why the pipeline uses 10 (a balance on 16 GB VRAM) and where the ceiling is.

> ⚠️ **VRAM note**: `all-MiniLM-L6-v2` is tiny (~90 MB) and runs on CPU by default (`RAG_MODEL_DEVICE=cpu`). If you point `TEXT_EMBEDDING_MODEL_ID` at a larger model and set `RAG_MODEL_DEVICE=cuda`, watch VRAM: batch size and `max_input_length` both scale activation memory.

---

## 🎓 Knowledge Check

1. **Why does `SingletonMeta` need a `Lock`?**
   - Answer: Model construction is slow; without a lock two threads could build two models at startup.

2. **What sizes the Qdrant collections?**
   - Answer: `EmbeddingModelSingleton.embedding_size`, probed once from a dummy encoding.

3. **Why do embedding handlers attach the model id to each vector's metadata?**
   - Answer: To detect stale vectors if the embedding model changes.

4. **Why is batching used in `chunk_and_embed` and `load_to_vector_db`?**
   - Answer: To bound memory per model/network call while keeping throughput high.

5. **Why does `EmbeddingDispatcher` assert a single category per batch?**
   - Answer: Because each category maps to a different Qdrant collection with different point shapes.

6. **What two things does the pipeline load into Qdrant?**
   - Answer: Cleaned documents (unindexed staging) and embedded chunks (searchable vectors).

---

## 🔗 Next Session

**Session 3.1**: Instruction Dataset Creation

We'll use the cleaned documents as seeds and generate instruction-answer pairs with GPT-4o-mini.

---

## 📚 Additional Resources

- [SentenceTransformers](https://www.sbert.net/)
- [Qdrant Upsert](https://qdrant.tech/documentation/concepts/points/#upload-points)
- [Python `concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 1.2, 1.3, 2.1, 2.2

**Outcome**: You can embed chunks, load them into Qdrant collection-by-collection, and explain every step of the feature-engineering pipeline.
