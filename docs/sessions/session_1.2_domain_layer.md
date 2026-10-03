# Session 1.2: Domain Layer - Data Modeling

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand the two base document hierarchies: `NoSQLBaseDocument` and `VectorBaseDocument`
- Model data with Pydantic v2 and `Generic[T]`
- Implement MongoDB CRUD through the ODM class methods
- Implement Qdrant vector operations through the same ODM pattern
- Know every concrete entity in `llm_engineering/domain/`

---

## 🏗️ Architecture Overview

### The Domain Layer at a Glance

The domain layer is the innermost layer. It has no knowledge of pipelines, steps, or AWS. It only knows business entities and how to persist them. Everything else depends on it.

```
┌───────────────────────────────────────────────────────────────┐
│                      domain/ (innermost)                       │
│                                                                │
│                    VectorBaseDocument (Qdrant)                 │
│                    NoSQLBaseDocument  (MongoDB)                │
│                            ▲                                   │
│             ┌──────────────┼───────────────┬───────────────┐   │
│             │              │               │               │   │
│        documents.py    chunks.py    embedded_chunks.py  dataset.py
│        (raw Mongo)     (Qdrant)     (Qdrant+vector)     (datasets)│
│             │              │               │               │   │
│   User/Post/Article   Post/Article     EmbeddedPost     Instruct  │
│   /Repository        /Repository       /Article/Repo    Preference│
└───────────────────────────────────────────────────────────────┘
```

### Why Two Base Classes?

The system stores the same conceptual data in two engines with different constraints:

| Concern | MongoDB (`NoSQLBaseDocument`) | Qdrant (`VectorBaseDocument`) |
|---------|-------------------------------|-------------------------------|
| Identity | `_id` string ↔ `id` UUID | point `id` string ↔ `id` UUID |
| Payload | full document | `payload` + optional `embedding` vector |
| Query | `find_one` / `find` | `search` / `scroll` |
| Bulk | `insert_many` | `upsert` points |
| Config | inner `Settings` class | inner `Config` class |

Both expose class methods (`find`, `bulk_find`, `save`, `bulk_insert`) so callers write the same code against either store.

---

## 📁 Key Files Explained

### 1. `llm_engineering/domain/types.py` - Data Categories

**Purpose**: A single `StrEnum` that tags every entity and collection in the system.

```python
# llm_engineering/domain/types.py
from enum import StrEnum


class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"

    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
```

**Key Concepts**:
- **`StrEnum`** members behave as `str`, so they serialize cleanly to MongoDB and Qdrant payloads.
- The values double as **collection names** (`Settings.name`, `Config.name`) and as the grouping key in `group_by_category`.
- Notice the enum spans the whole lifecycle: raw data (`POSTS`, `ARTICLES`, `REPOSITORIES`), generated prompts (`PROMPT`, `QUERIES`), and datasets (`INSTRUCT_*`, `PREFERENCE_*`).

---

### 2. `llm_engineering/domain/base/nosql.py` - MongoDB Base Document

**Purpose**: A generic, Pydantic-based ODM base for MongoDB.

```python
# llm_engineering/domain/base/nosql.py
import uuid
from abc import ABC
from typing import Generic, Type, TypeVar

from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import errors

from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings

_database = connection.get_database(settings.DATABASE_NAME)

T = TypeVar("T", bound="NoSQLBaseDocument")


class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)
```

**Identity and hashing**:

```python
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)
```

Equality is by `id`, not by all fields. This makes de-duplication trivial:

```python
documents = list({doc.id: doc for doc in documents}.values())
```

**The Mongo ⇄ Pydantic translation layer**:

```python
    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        """Convert "_id" (str object) into "id" (UUID object)."""
        if not data:
            raise ValueError("Data is empty.")

        id = data.pop("_id")
        return cls(**dict(data, id=id))

    def to_mongo(self: T, **kwargs) -> dict:
        """Convert "id" (UUID object) into "_id" (str object)."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        return parsed
```

**Why this matters**: MongoDB's primary key is `_id` (a string/ObjectId), while the Python model uses `id: UUID4`. The two methods form a clean seam so no other layer worries about the difference.

**Full CRUD surface**:

```python
    def save(self: T, **kwargs) -> T | None:
        collection = _database[self.get_collection_name()]
        try:
            collection.insert_one(self.to_mongo(**kwargs))
            return self
        except errors.WriteError:
            logger.exception("Failed to insert document.")
            return None

    @classmethod
    def get_or_create(cls: Type[T], **filter_options) -> T:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)

            new_instance = cls(**filter_options)
            new_instance = new_instance.save()
            return new_instance
        except errors.OperationFailure:
            logger.exception(f"Failed to retrieve document with filter options: {filter_options}")
            raise

    @classmethod
    def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
        collection = _database[cls.get_collection_name()]
        try:
            collection.insert_many(doc.to_mongo(**kwargs) for doc in documents)
            return True
        except (errors.WriteError, errors.BulkWriteError):
            logger.error(f"Failed to insert documents of type {cls.__name__}")
            return False

    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)
            return None
        except errors.OperationFailure:
            logger.error("Failed to retrieve document")
            return None

    @classmethod
    def bulk_find(cls: Type[T], **filter_options) -> list[T]:
        collection = _database[cls.get_collection_name()]
        try:
            instances = collection.find(filter_options)
            return [document for instance in instances if (document := cls.from_mongo(instance)) is not None]
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents")
            return []
```

**Collection name resolution**:

```python
    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define an Settings configuration class with the name of the collection."
            )
        return cls.Settings.name
```

**Key Concepts**:
- **Generic + ABC**: `Generic[T]` gives subclasses a precise return type; `ABC` prevents direct instantiation.
- **Class methods as an ODM**: There is no repository class. Persistence lives on the entity itself.
- **Fail soft**: Read helpers log and return `None`/`[]`; the caller decides what to do.
- **`get_collection_name` guard**: A subclass that forgets its `Settings.name` fails immediately with `ImproperlyConfigured`, not with a cryptic key error later.

---

### 3. `llm_engineering/domain/base/vector.py` - Qdrant Base Document

**Purpose**: The vector-store counterpart. Same ODM style, different engine.

```python
# llm_engineering/domain/base/vector.py
class VectorBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    @classmethod
    def from_record(cls: Type[T], point: Record) -> T:
        _id = UUID(point.id, version=4)
        payload = point.payload or {}

        attributes = {
            "id": _id,
            **payload,
        }
        if cls._has_class_attribute("embedding"):
            attributes["embedding"] = point.vector or None

        return cls(**attributes)

    def to_point(self: T, **kwargs) -> PointStruct:
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        payload = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        _id = str(payload.pop("id"))
        vector = payload.pop("embedding", {})
        if vector and isinstance(vector, np.ndarray):
            vector = vector.tolist()

        return PointStruct(id=_id, vector=vector, payload=payload)
```

**Why `_has_class_attribute("embedding")`**: Not every vector document carries an embedding. `CleanedDocument` (a vector collection used as a staging store) has no `embedding` field, so `from_record` must not inject one.

**Search and scroll**:

```python
    @classmethod
    def search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]:
        try:
            documents = cls._search(query_vector=query_vector, limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            logger.error(f"Failed to search documents in '{cls.get_collection_name()}'.")
            documents = []
        return documents

    @classmethod
    def _search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]:
        collection_name = cls.get_collection_name()
        records = connection.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", False),
            **kwargs,
        )
        documents = [cls.from_record(record) for record in records]
        return documents

    @classmethod
    def bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[list[T], UUID | None]:
        try:
            documents, next_offset = cls._bulk_find(limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            logger.error(f"Failed to search documents in '{cls.get_collection_name()}'.")
            documents, next_offset = [], None
        return documents, next_offset
```

`_bulk_find` wraps Qdrant's `scroll`, returning a `(documents, next_offset)` tuple so callers can paginate through a large collection.

**Collection management**:

```python
    @classmethod
    def bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> bool:
        try:
            cls._bulk_insert(documents)
        except exceptions.UnexpectedResponse:
            logger.info(
                f"Collection '{cls.get_collection_name()}' does not exist. "
                "Trying to create the collection and reinsert."
            )
            cls.create_collection()
            try:
                cls._bulk_insert(documents)
            except exceptions.UnexpectedResponse:
                logger.error(f"Failed to insert documents in '{cls.get_collection_name()}'.")
                return False
        return True

    @classmethod
    def _create_collection(cls, collection_name: str, use_vector_index: bool = True) -> bool:
        if use_vector_index is True:
            vectors_config = VectorParams(
                size=EmbeddingModelSingleton().embedding_size,
                distance=Distance.COSINE,
            )
        else:
            vectors_config = {}

        return connection.create_collection(collection_name=collection_name, vectors_config=vectors_config)
```

**Key Concepts**:
- **Just-in-time collection creation**: `bulk_insert` creates the collection on first use instead of requiring a migration step.
- **`use_vector_index`**: collections used only as string stores set `use_vector_index = False`; the vector size comes from the embedding singleton, so the collection dimension can never drift from the model dimension.

**Grouping utilities**:

```python
    @classmethod
    def group_by_class(cls, documents):        # -> dict[type, list[doc]]
    @classmethod
    def group_by_category(cls, documents):     # -> dict[DataCategory, list[doc]]
```

These are used by the feature-engineering and embedding steps to fan a mixed list of chunks out to the right collection.

---

### 4. `llm_engineering/domain/documents.py` - Raw Documents (MongoDB)

**Purpose**: The entities the crawlers write, before any cleaning.

```python
# llm_engineering/domain/documents.py
from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from .base import NoSQLBaseDocument
from .types import DataCategory


class UserDocument(NoSQLBaseDocument):
    first_name: str
    last_name: str

    class Settings:
        name = "users"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"


class Document(NoSQLBaseDocument, ABC):
    content: dict
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")


class RepositoryDocument(Document):
    name: str
    link: str

    class Settings:
        name = DataCategory.REPOSITORIES


class PostDocument(Document):
    image: Optional[str] = None
    link: str | None = None

    class Settings:
        name = DataCategory.POSTS


class ArticleDocument(Document):
    link: str

    class Settings:
        name = DataCategory.ARTICLES
```

**Key Concepts**:
- **`UserDocument` is the root entity**. Every document references `author_id` and `author_full_name`, which become the metadata used later for self-query filtering.
- **`Document` is abstract** and shared by the three platform documents.
- **`content` is a `dict`** at this stage. The crawler stores a platform-specific structure (title, subtitle, body) that cleaning will later collapse into a single string.
- **`Settings.name` uses `DataCategory` members** for the three document types, so collection names stay consistent: `posts`, `articles`, `repositories`.

---

### 5. `llm_engineering/domain/cleaned_documents.py` - Cleaned Documents

**Purpose**: The output of the cleaning step, stored as vector-collection payloads but **without** embeddings.

```python
# llm_engineering/domain/cleaned_documents.py
class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: UUID4
    author_full_name: str


class CleanedPostDocument(CleanedDocument):
    image: Optional[str] = None

    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


class CleanedArticleDocument(CleanedDocument):
    link: str

    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class CleanedRepositoryDocument(CleanedDocument):
    name: str
    link: str

    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False
```

**Key Concepts**:
- `content` is now a `str` (cleaned text), not a dict.
- **`use_vector_index = False`**: cleaned documents are a staging store in Qdrant, not a searchable vector index. This is the case that `to_point` / `from_record` must tolerate with no embedding.
- Transition `NoSQLBaseDocument → VectorBaseDocument` is deliberate: raw data lives in MongoDB, cleaned data lives in Qdrant.

---

### 6. `llm_engineering/domain/chunks.py` - Chunks

**Purpose**: Cleaned documents split into retrieval-sized pieces.

```python
# llm_engineering/domain/chunks.py
class Chunk(VectorBaseDocument, ABC):
    content: str
    platform: str
    document_id: UUID4
    author_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)


class PostChunk(Chunk):
    image: Optional[str] = None
    class Config:
        category = DataCategory.POSTS


class ArticleChunk(Chunk):
    link: str
    class Config:
        category = DataCategory.ARTICLES


class RepositoryChunk(Chunk):
    name: str
    link: str
    class Config:
        category = DataCategory.REPOSITORIES
```

**Key Concepts**:
- **`document_id` links a chunk back to its parent** cleaned document. This is the foreign key that enables citations.
- **`metadata`** is a free dict that carries chunker-specific info (for example the chunk index).
- The chunk has no `name`/`link` on the base class; platform-specific fields live on the subclasses.

---

### 7. `llm_engineering/domain/embedded_chunks.py` - Embedded Chunks

**Purpose**: Chunks plus their embedding. These are the searchable vectors used at inference time.

```python
# llm_engineering/domain/embedded_chunks.py
class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: list[float] | None
    platform: str
    document_id: UUID4
    author_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Author: {chunk.author_full_name}
            Content: {chunk.content}\n
            """
        return context


class EmbeddedPostChunk(EmbeddedChunk):
    class Config:
        name = "embedded_posts"
        category = DataCategory.POSTS
        use_vector_index = True


class EmbeddedArticleChunk(EmbeddedChunk):
    link: str
    class Config:
        name = "embedded_articles"
        category = DataCategory.ARTICLES
        use_vector_index = True


class EmbeddedRepositoryChunk(EmbeddedChunk):
    name: str
    link: str
    class Config:
        name = "embedded_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = True
```

**Key Concepts**:
- **`use_vector_index = True`**: these are the only queryable vector collections in Qdrant.
- **`to_context`** is the bridge from retrieval to generation. It renders the retrieved chunks into the prompt context that the RAG chain passes to the LLM.

---

### 8. `llm_engineering/domain/dataset.py` - Training Datasets

**Purpose**: The entities produced by dataset generation and consumed by training.

```python
# llm_engineering/domain/dataset.py
class DatasetType(Enum):
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"


class InstructDatasetSample(VectorBaseDocument):
    instruction: str
    answer: str
    class Config:
        category = DataCategory.INSTRUCT_DATASET_SAMPLES


class PreferenceDatasetSample(VectorBaseDocument):
    instruction: str
    rejected: str
    chosen: str
    class Config:
        category = DataCategory.PREFERENCE_DATASET_SAMPLES


class InstructDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[InstructDatasetSample]
    class Config:
        category = DataCategory.INSTRUCT_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]
        return Dataset.from_dict(
            {"instruction": [d["instruction"] for d in data], "output": [d["answer"] for d in data]}
        )
```

```python
class PreferenceDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[PreferenceDatasetSample]

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]
        return Dataset.from_dict({
            "prompt": [d["instruction"] for d in data],
            "rejected": [d["rejected"] for d in data],
            "chosen": [d["chosen"] for d in data],
        })
```

**Key Concepts**:
- **`to_huggingface()`** maps the internal schema to the exact column names the training stack expects: `instruction`/`output` for SFT, `prompt`/`chosen`/`rejected` for DPO.
- **`TrainTestSplit`** wraps `train` / `test` dicts keyed by `DataCategory` and can `flatten` them into a single `datasets.DatasetDict`.

---

### 9. `llm_engineering/domain/queries.py` and `prompt.py`

**Purpose**: The inference-side and generation-side value objects.

```python
# llm_engineering/domain/queries.py
class Query(VectorBaseDocument):
    content: str
    author_id: UUID4 | None = None
    author_full_name: str | None = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        category = DataCategory.QUERIES

    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))

    def replace_content(self, new_content: str) -> "Query":
        return Query(
            id=self.id,
            content=new_content,
            author_id=self.author_id,
            author_full_name=self.author_full_name,
            metadata=self.metadata,
        )


class EmbeddedQuery(Query):
    embedding: list[float]
```

`replace_content` is what query expansion uses: it keeps the original `id` and metadata while swapping in a rewritten query, so traces stay grouped.

```python
# llm_engineering/domain/prompt.py
class Prompt(VectorBaseDocument):
    template: str
    input_variables: dict
    content: str
    num_tokens: int | None = None
    class Config:
        category = DataCategory.PROMPT


class GenerateDatasetSamplesPrompt(Prompt):
    data_category: DataCategory
    document: CleanedDocument
```

---

## 🧠 Design Patterns Used

### 1. **Generic Repository (ODM)**

Instead of a separate `DocumentRepository` class, persistence is generic over `T`:

```python
T = TypeVar("T", bound="NoSQLBaseDocument")

class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None: ...
```

Callers get full type inference: `UserDocument.find(...)` returns `UserDocument | None`.

### 2. **Template Method**

The base defines the persist/insert skeleton; `to_mongo` / `to_point` are the overridable translation steps. Subclasses only declare fields and a `Config`/`Settings` class.

### 3. **Configuration via inner class**

`Settings` (Mongo) and `Config` (Qdrant) declare `name` and `category`. The base validates their presence at call time and raises `ImproperlyConfigured`.

### 4. **Singleton accessor for connections**

`connection` is created at import time in `infrastructure/db/*.py`; the domain layer just imports it. (Covered in depth in Session 1.3.)

---

## 🛠️ Hands-On: Create a Custom Document Model

### Step 1: Define a new raw document

```python
# llm_engineering/domain/documents.py
class TweetDocument(Document):
    link: str
    hashtags: list[str] = []

    class Settings:
        name = "tweets"
```

### Step 2: Define its cleaned and chunked forms

```python
# cleaned_documents.py
class CleanedTweetDocument(CleanedDocument):
    link: str

    class Config:
        name = "cleaned_tweets"
        category = DataCategory.POSTS   # reuse a category if no new enum value exists
        use_vector_index = False
```

### Step 3: Persist and read it

```python
from llm_engineering.domain.documents import TweetDocument

doc = TweetDocument(
    content={"text": "Hello world"},
    platform="twitter",
    author_id=user.id,
    author_full_name=user.full_name,
    link="https://x.com/...",
    hashtags=["#llm"],
)
saved = doc.save()                      # writes to the "tweets" collection
found = TweetDocument.find(_id=str(doc.id))
print(found == doc)                     # True (equality by id)
```

### Step 4: Inspect collections in MongoDB

```bash
# inside the container
docker compose exec mongo mongosh \
  "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017/twin" \
  --eval "db.getCollectionNames()"
```

---

## 📝 Exercise: Trace the Lifecycle of One Entity

### Task

Follow a single `ArticleDocument` from creation to retrievable chunk.

1. Create an `ArticleDocument` and `save()` it → collection `articles`.
2. Clean it into a `CleanedArticleDocument` and `bulk_insert()` → collection `cleaned_articles`.
3. Chunk it into `ArticleChunk` objects → collection `chunks` (category `articles`).
4. Embed into `EmbeddedArticleChunk` and `bulk_insert()` → collection `embedded_articles`.
5. `EmbeddedArticleChunk.search(query_vector=...)` and inspect `metadata`, `document_id`, `author_full_name`.

**Goal**: See how `id`, `document_id`, and `author_id` thread through every stage.

---

## 🎓 Knowledge Check

1. **Why does `NoSQLBaseDocument` override `model_dump`?**
   - Answer: To convert `UUID` field values into plain `str` so they can be stored in Mongo.

2. **What is the difference between `to_mongo` and `to_point`?**
   - Answer: `to_mongo` emits a Mongo document with `_id`; `to_point` emits a Qdrant `PointStruct` with a separate `vector` and `payload`.

3. **Why do `CleanedDocument`s set `use_vector_index = False`?**
   - Answer: They are a staging payload store with no embedding and are never searched by vector similarity.

4. **What does `to_context` produce, and who consumes it?**
   - Answer: A formatted string of the retrieved chunks; the RAG prompt builder consumes it during generation.

5. **How is de-duplication across query variations implemented?**
   - Answer: Documents hash by `id`, so `{doc.id: doc for doc in docs}.values()` collapses duplicates.

6. **Where does a chunk get its link back to the source?**
   - Answer: `document_id` on `Chunk`, pointing at the parent cleaned document.

---

## 🔗 Next Session

**Session 1.3**: Infrastructure Layer - Database Connections

We'll cover:
- `MongoDatabaseConnector` and `QdrantDatabaseConnector` singletons
- Connection lifecycle and lazy environment loading
- `docker-compose.yml` services
- Verifying both stores with real queries

---

## 📚 Additional Resources

- [Pydantic Generics](https://docs.pydantic.dev/latest/concepts/models/#generic-models)
- [PyMongo CRUD](https://pymongo.readthedocs.io/en/stable/tutorial.html)
- [Qdrant Points and Collections](https://qdrant.tech/documentation/concepts/points/)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Session 1.1

**Outcome**: You can design and persist domain entities against both MongoDB and Qdrant using the project's ODM base classes.
