# Session 2.2: Text Preprocessing Pipeline

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand the factory + dispatcher + handler pattern used for preprocessing
- Clean raw platform documents into normalized text
- Chunk documents with the right strategy per data category
- Explain how deterministic chunk IDs enable idempotent re-runs

---

## 🏗️ Architecture Overview

### The Preprocessing Stage

Preprocessing sits between the ETL (crawling) stage and feature engineering. It has two transforms: **clean** and **chunk**.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Preprocessing Pipeline                          │
│                                                                    │
│  Raw docs (MongoDB)                                                │
│  Post / Article / Repository                                       │
│        │                                                           │
│        ▼                                                           │
│  ┌───────────────────────────────┐                                 │
│  │  CleaningDispatcher.dispatch   │                                │
│  │    └─ CleaningHandlerFactory   │                                │
│  │        ├─ PostCleaningHandler  │                                │
│  │        ├─ ArticleCleaningHandler│                               │
│  │        └─ RepositoryCleaningHandler                             │
│  └───────────────────────────────┘                                 │
│        │  CleanedDocument (str content, no embedding)              │
│        ▼                                                           │
│  ┌───────────────────────────────┐                                 │
│  │  ChunkingDispatcher.dispatch   │                                │
│  │    └─ ChunkingHandlerFactory   │                                │
│  │        ├─ PostChunkingHandler  │  (250/25)                      │
│  │        ├─ ArticleChunkingHandler│ (1000..2000 sentence chunks)  │
│  │        └─ RepositoryChunkingHandler (1500/100)                  │
│  └───────────────────────────────┘                                 │
│        │  Chunk objects (deterministic md5 ids)                    │
│        ▼                                                           │
│  Feature engineering (Session 2.3)                                 │
└──────────────────────────────────────────────────────────────────┘
```

### The Three-Level Pattern

Every transform uses the same shape:

```
Dispatcher  ──reads category──►  Factory  ──builds──►  Handler
   (entry)                     (selection)            (logic)
```

- **Dispatcher**: the public entry point. Infers the `DataCategory` and delegates.
- **Factory**: maps `DataCategory` → concrete handler. The single place to add a new type.
- **Handler**: abstract generic base plus one concrete subclass per category that does the actual work.

---

## 📁 Key Files Explained

### 1. `llm_engineering/application/preprocessing/operations/cleaning.py`

**Purpose**: The only cleaning primitive.

```python
# llm_engineering/application/preprocessing/operations/cleaning.py
import re


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
```

**Key Concepts**:
- The first regex removes every character that is **not** a word character, whitespace, or `. , ! ?`. Markdown symbols (`#`, `*`, `` ` ``), HTML remnants, and punctuation collapse to spaces.
- The second regex collapses runs of whitespace into a single space.
- Deliberately simple: cleaning normalizes noise, it does not attempt semantic parsing. LLM-facing quality comes from chunking and prompt design.

---

### 2. `llm_engineering/application/preprocessing/cleaning_data_handlers.py`

**Purpose**: Convert each raw `Document` into its `CleanedDocument` form.

```python
# llm_engineering/application/preprocessing/cleaning_data_handlers.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument, CleanedDocument, CleanedPostDocument, CleanedRepositoryDocument,
)
from llm_engineering.domain.documents import (
    ArticleDocument, Document, PostDocument, RepositoryDocument,
)

from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass
```

**Concrete handlers**:

```python
class PostCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PostDocument) -> CleanedPostDocument:
        return CleanedPostDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            image=data_model.image if data_model.image else None,
        )


class ArticleCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ArticleDocument) -> CleanedArticleDocument:
        valid_content = [content for content in data_model.content.values() if content]

        return CleanedArticleDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(valid_content)),
            platform=data_model.platform,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> CleanedRepositoryDocument:
        return CleanedRepositoryDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )
```

**Key Concepts**:
- **`" #### ".join(content.values())`** flattens the dict of platform fields (title, subtitle, body) into one string, separated by a marker that survives cleaning (`#` becomes a space, but the intent is a separator).
- **`id=data_model.id` is preserved**. Cleaning does not mint a new identity; downstream chunks can still trace to the raw document.
- **`valid_content`** filters empty fields, which matters for articles where a subtitle may be missing.

---

### 3. `llm_engineering/application/preprocessing/operations/chunking.py`

**Purpose**: Two chunking strategies - character/token chunking and sentence-aware article chunking.

```python
# llm_engineering/application/preprocessing/operations/chunking.py
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from llm_engineering.application.networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0
    )
    text_split_by_characters = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )
    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens


def chunk_document(text: str, min_length: int, max_length: int) -> list[str]:
    """Alias for chunk_article()."""
    return chunk_article(text, min_length, max_length)


def chunk_article(text: str, min_length: int, max_length: int) -> list[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    extracts = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())
            current_chunk = sentence + " "

    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())

    return extracts
```

**Key Concepts**:
- **`chunk_text` is a two-pass split**. Pass 1 (`RecursiveCharacterTextSplitter`, separators `["\n\n"]`) splits on paragraphs by **characters**. Pass 2 (`SentenceTransformersTokenTextSplitter`) re-splits each section by **tokens** using the embedding model's own tokenizer.
- **Why two passes**: the embedding model has a hard token limit (`max_input_length`). Splitting first by paragraphs avoids cutting mid-idea; the token pass guarantees every chunk fits the model.
- **`tokens_per_chunk=embedding_model.max_input_length`** ties chunk size directly to the embedding model. If you swap the embedding model, chunk sizes adapt automatically.
- **`chunk_article`** is sentence-aware: the regex splits on sentence boundaries (with guards against false splits on abbreviations like `e.g.`). It greedily packs sentences up to `max_length` and only emits a chunk once it reaches `min_length`.
- **Sentence chunking preserves meaning** better than fixed character windows for prose articles.

---

### 4. `llm_engineering/application/preprocessing/chunking_data_handlers.py`

**Purpose**: Per-category chunking with deterministic IDs.

```python
# llm_engineering/application/preprocessing/chunking_data_handlers.py
class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    @property
    def metadata(self) -> dict:
        return {"chunk_size": 500, "chunk_overlap": 50}

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass
```

**Post handler (small chunks)**:

```python
class PostChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {"chunk_size": 250, "chunk_overlap": 25}

    def chunk(self, data_model: CleanedPostDocument) -> list[PostChunk]:
        data_models_list = []
        chunks = chunk_text(
            data_model.content,
            chunk_size=self.metadata["chunk_size"],
            chunk_overlap=self.metadata["chunk_overlap"],
        )
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = PostChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                image=data_model.image if data_model.image else None,
                metadata=self.metadata,
            )
            data_models_list.append(model)
        return data_models_list
```

**Article handler (sentence chunks)**:

```python
class ArticleChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {"min_length": 1000, "max_length": 2000}

    def chunk(self, data_model: CleanedArticleDocument) -> list[ArticleChunk]:
        ...
        chunks = chunk_article(
            cleaned_content, min_length=self.metadata["min_length"], max_length=self.metadata["max_length"]
        )
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ArticleChunk(id=UUID(chunk_id, version=4), ...)
```

**Repository handler (large code chunks)**:

```python
class RepositoryChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {"chunk_size": 1500, "chunk_overlap": 100}
```

**Chunk-size rationale**:

| Category | Strategy | Size / overlap | Why |
|----------|----------|----------------|-----|
| Posts | token | 250 / 25 | Short-form content, tight chunks improve precision |
| Articles | sentence | 1000–2000 chars | Long-form prose, keep sentences intact |
| Repositories | token | 1500 / 100 | Code needs large windows for context |

**Key Concepts**:
- **`hashlib.md5(chunk.encode()).hexdigest()`** makes the chunk ID a pure function of its text. Re-running the pipeline produces the **same IDs**, so Qdrant `upsert` is idempotent - no duplicate chunks.
- **`metadata` is attached to every chunk** and stored in the payload, so you can trace which parameters produced a chunk.
- **`document_id`** preserves the link back to the cleaned parent.

---

### 5. `llm_engineering/application/preprocessing/dispatchers.py`

**Purpose**: The factory + dispatcher facade.

```python
# llm_engineering/application/preprocessing/dispatchers.py
class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> CleaningDataHandler:
        if data_category == DataCategory.POSTS:
            return PostCleaningHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleCleaningHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryCleaningHandler()
        else:
            raise ValueError("Unsupported data type")


class CleaningDispatcher:
    factory = CleaningHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        data_category = DataCategory(data_model.get_collection_name())
        handler = cls.factory.create_handler(data_category)
        clean_model = handler.clean(data_model)

        logger.info(
            "Document cleaned successfully.",
            data_category=data_category,
            cleaned_content_len=len(clean_model.content),
        )
        return clean_model
```

**Chunking dispatcher**:

```python
class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> ChunkingDataHandler:
        if data_category == DataCategory.POSTS:
            return PostChunkingHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleChunkingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryChunkingHandler()
        else:
            raise ValueError("Unsupported data type")


class ChunkingDispatcher:
    factory = ChunkingHandlerFactory

    @classmethod
    def dispatch(cls, data_model: VectorBaseDocument) -> list[VectorBaseDocument]:
        data_category = data_model.get_category()
        handler = cls.factory.create_handler(data_category)
        chunk_models = handler.chunk(data_model)
        ...
        return chunk_models
```

**Key Concepts**:
- **Category inference differs by stage**: cleaning derives the category from the **collection name** (`DataCategory(data_model.get_collection_name())`), while chunking reads it from the vector document's **`Config.category`**.
- **O(1) extension point**: to support `TWEET`, add one enum value, one handler class, and one factory branch. Nothing else changes.
- The dispatcher logs structured context (`data_category`, `cleaned_content_len`) that shows up in log aggregation.

---

## 🔧 Pipeline Integration

### The `clean_documents` step

```python
# steps/feature_engineering/clean.py
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.preprocessing import CleaningDispatcher
from llm_engineering.domain.cleaned_documents import CleanedDocument


@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    cleaned_documents = []
    for document in documents:
        cleaned_document = CleaningDispatcher.dispatch(document)
        cleaned_documents.append(cleaned_document)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents)
    )
    return cleaned_documents
```

The `_get_metadata` helper counts documents and unique authors per category, so the ZenML dashboard shows exactly what was cleaned.

### The `chunk_and_embed` step

`steps/feature_engineering/chunk_and_embed.py` calls `ChunkingDispatcher.dispatch` then the embedding dispatcher (Session 2.3), producing `EmbeddedChunk` objects ready for Qdrant.

---

## 🛠️ Hands-On: Clean and Chunk a Real Document

### Step 1: Fetch a raw document

```python
from llm_engineering.domain.documents import ArticleDocument

doc = ArticleDocument.bulk_find(author_id="<user-uuid>")[0]
print(type(doc.content), list(doc.content.keys()))
```

### Step 2: Clean it

```python
from llm_engineering.application.preprocessing import CleaningDispatcher

cleaned = CleaningDispatcher.dispatch(doc)
print(cleaned.get_collection_name())   # cleaned_articles
print(len(cleaned.content), cleaned.content[:200])
```

### Step 3: Chunk it

```python
from llm_engineering.application.preprocessing import ChunkingDispatcher

chunks = ChunkingDispatcher.dispatch(cleaned)
print(f"{len(chunks)} chunks")
for c in chunks[:3]:
    print(c.id, "|", c.content[:80])
```

### Step 4: Prove idempotency

```python
chunks_again = ChunkingDispatcher.dispatch(cleaned)
assert [c.id for c in chunks] == [c.id for c in chunks_again]
print("Deterministic chunk IDs confirmed")
```

---

## 📝 Exercise: Add a LinkedIn-comment category

### Task

Support short "comment" documents with their own chunking rule.

1. Add `COMMENTS = "comments"` to `DataCategory`.
2. Add `CommentDocument` (Mongo) and `CleanedCommentDocument` (vector, `use_vector_index=False`).
3. Add `CommentChunk` with `Config.category = DataCategory.COMMENTS`.
4. Add `CommentCleaningHandler` and `CommentChunkingHandler` (try `chunk_size=120, chunk_overlap=10`).
5. Register both in the factories.
6. Round-trip a sample document through cleaning and chunking.

**Goal**: Internalize the three-level pattern and where each addition belongs.

---

## 🐛 Common Pitfalls

- **Empty content**: `clean_text` returns `""` for all-symbol input. Guard before embedding; `SentenceTransformersTokenTextSplitter` on an empty string yields no chunks, which is safe but silent.
- **`max_input_length` mismatch**: chunk token size is always `embedding_model.max_input_length`. Never hard-code it, or chunks may be truncated at embed time.
- **`DataCategory` coverage**: the factories raise `ValueError("Unsupported data type")` for unknown categories. New categories must be registered in **all three** factories you use.

---

## 🎓 Knowledge Check

1. **Why two splitting passes in `chunk_text`?**
   - Answer: Character/paragraph splitting keeps ideas intact; the token pass enforces the embedding model's token limit.

2. **Why is a chunk ID an md5 of its content?**
   - Answer: So re-processing produces identical IDs and `upsert` is idempotent.

3. **How does the cleaning dispatcher decide which handler to use?**
   - Answer: It converts `get_collection_name()` into a `DataCategory` and the factory maps it to a handler.

4. **What stays the same between a raw document and its cleaned form?**
   - Answer: `id`, `platform`, `author_id`, and `author_full_name`.

5. **Why does `chunk_article` need both `min_length` and `max_length`?**
   - Answer: `max_length` caps chunk size; `min_length` prevents emitting tiny, low-signal chunks.

6. **Where is the per-category chunk configuration stored?**
   - Answer: In each handler's `metadata` property, and copied onto every chunk's `metadata` field.

---

## 🔗 Next Session

**Session 2.3**: Feature Engineering Pipeline

We'll embed chunks with `EmbeddingModelSingleton`, batch-load them into Qdrant, and study the feature-engineering pipeline end to end.

---

## 📚 Additional Resources

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [SentenceTransformers Token Splitter](https://python.langchain.com/docs/integrations/document_transformers/sentence_transformers_token/)
- [Factory Pattern](https://refactoring.guru/design-patterns/factory-method)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 1.2, 1.3, 2.1

**Outcome**: You can route any document category through cleaning and category-specific chunking, and reason about chunk sizes.
