# Session 1.1: Project Overview & DDD Principles

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand the project's layered architecture
- Know how dependencies flow between layers
- Be able to navigate the codebase effectively
- Set up your development environment

---

## 📐 Architecture Overview

### The Big Picture

This project builds an **LLM Twin** - an AI that mimics your writing style and knowledge using RAG and fine-tuned LLMs.

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Flow                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Collection → Preprocessing → Features → Training → RAG     │
│      ↓            ↓             ↓           ↓         ↓      │
│   Selenium    Cleaning     Embeddings   SFT/DPO   Retrieval │
│   Git/Medium  Chunking     Vector DB    SageMaker  + LLM    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layered Architecture (DDD)

```
┌─────────────────────────────────────────────────────────────┐
│                    Dependency Flow                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  infrastructure (AWS, MongoDB, Qdrant, FastAPI)             │
│       ↓                                                      │
│  model (Training, Inference, Evaluation)                    │
│       ↓                                                      │
│  application (Crawlers, Preprocessing, RAG, Dataset Gen)    │
│       ↓                                                      │
│  domain (Core business entities: Documents, Chunks, Datasets)│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle**: Higher layers can depend on lower layers, but NEVER the reverse.

---

## 📁 Project Structure Deep Dive

### Root Level

```
LLM-Engineers-Handbook/
├── .env.example              # Template for environment variables
├── .python-version           # Python version (3.11)
├── pyproject.toml            # Dependencies and Poe the Poet tasks
├── docker-compose.yml        # Local MongoDB + Qdrant
├── Dockerfile                # Container image
├── configs/                  # ZenML YAML configs
├── llm_engineering/          # Main Python package
├── pipelines/                # ZenML pipeline definitions
├── steps/                    # ZenML step components
├── tools/                    # Utility scripts
└── code_snippets/            # Standalone examples
```

### The `llm_engineering` Package

```
llm_engineering/
├── __init__.py               # Package initialization
├── settings.py               # Centralized configuration
│
├── domain/                   # Core business entities
│   ├── __init__.py
│   ├── types.py              # Data categories (POST, ARTICLE, etc.)
│   ├── base/
│   │   ├── nosql.py          # MongoDB base document
│   │   └── vector.py         # Qdrant vector base document
│   ├── documents.py          # Raw document models
│   ├── cleaned_documents.py  # Cleaned document models
│   ├── chunks.py             # Text chunk models
│   ├── embedded_chunks.py    # Vector-enabled chunks
│   └── dataset.py            # Training dataset models
│
├── application/              # Business logic
│   ├── __init__.py
│   ├── crawlers/             # Web scraping
│   │   ├── base.py
│   │   ├── dispatcher.py
│   │   ├── github.py
│   │   ├── linkedin.py
│   │   └── medium.py
│   ├── preprocessing/        # Text processing
│   │   ├── dispatchers.py
│   │   ├── cleaning_data_handlers.py
│   │   ├── chunking_data_handlers.py
│   │   ├── embedding_data_handlers.py
│   │   └── operations/
│   │       ├── cleaning.py
│   │       ├── chunking.py
│   │       └── embedding.py
│   ├── dataset/              # Dataset generation
│   │   ├── generation.py
│   │   ├── output_parsers.py
│   │   └── constants.py
│   └── rag/                  # RAG implementation
│       ├── retriever.py
│       ├── query_expanison.py
│       ├── reranking.py
│       └── self_query.py
│
├── model/                    # ML logic
│   ├── __init__.py
│   ├── finetuning/           # SFT and DPO training
│   │   ├── finetune.py
│   │   └── sagemaker.py
│   ├── inference/            # Inference logic
│   │   ├── inference.py
│   │   └── run.py
│   └── evaluation/           # Model evaluation
│       ├── evaluate.py
│       └── sagemaker.py
│
└── infrastructure/           # External services
    ├── __init__.py
    ├── aws/                  # AWS SageMaker
    ├── db/                   # Database connections
    │   ├── mongo.py
    │   └── qdrant.py
    ├── networks/             # Embedding models
    │   ├── base.py
    │   └── embeddings.py
    └── inference_pipeline_api.py  # FastAPI app
```

---

## 🔍 Key Files Explained

### 1. `settings.py` - Centralized Configuration

**Purpose**: Single source of truth for all configuration using Pydantic.

```python
# llm_engineering/settings.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # OpenAI
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str
    
    # Hugging Face
    HUGGINGFACE_ACCESS_TOKEN: str
    
    # Comet ML & Opik
    COMET_API_KEY: str
    COMET_PROJECT: str = "llm-engineering"
    
    # MongoDB
    DATABASE_HOST: str = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
    
    # Qdrant
    USE_QDRANT_CLOUD: bool = False
    QDRANT_CLOUD_URL: str = ""
    QDRANT_APIKEY: str = ""
    
    # AWS
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str
    AWS_ARN_ROLE: str
    
    # SageMaker
    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin"
    SAGEMAKER_ENDPOINT_TRAINING: str = "twin-training"
    
    # Model IDs
    HF_MODEL_ID: str = "mlabonne/TwinLlama-3.1-8B-DPO"
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

settings = Settings()
```

**Usage**:
```python
from llm_engineering.settings import settings

# Access any setting
api_key = settings.OPENAI_API_KEY
mongo_uri = settings.DATABASE_HOST
```

---

### 2. `domain/types.py` - Data Categories

**Purpose**: Define the types of data the system handles.

```python
# llm_engineering/domain/types.py

from enum import StrEnum

class DataCategory(StrEnum):
    POST = "post"           # LinkedIn posts
    ARTICLE = "article"     # Medium articles
    REPOSITORY = "repository"  # GitHub repositories
    TWEET = "tweet"         # Tweets (not implemented)
```

**Usage**: Used for categorizing documents, chunks, and datasets throughout the system.

---

### 3. `__init__.py` - Package Structure

**Purpose**: Define what's exported from the package.

```python
# llm_engineering/__init__.py

from llm_engineering import application, domain, infrastructure

__all__ = ["application", "domain", "infrastructure"]
```

---

## 🛠️ Hands-On: Environment Setup

### Step 1: Verify Python Version

```bash
python --version
# Should show: Python 3.11.x
```

### Step 2: Activate Virtual Environment

```bash
.venv\Scripts\activate
```

### Step 3: Copy Environment Template

```bash
cp .env.example .env
```

### Step 4: Edit `.env` File

Open `.env` and fill in your credentials:

```env
# Required for LLM features
OPENAI_API_KEY=sk-...
HUGGINGFACE_ACCESS_TOKEN=hf_...
COMET_API_KEY=...

# Optional (defaults work for local)
DATABASE_HOST="mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
USE_QDRANT_CLOUD=false
```

### Step 5: Test Configuration

```python
# test_settings.py
from llm_engineering.settings import settings

print(f"OpenAI Model: {settings.OPENAI_MODEL_ID}")
print(f"MongoDB URI: {settings.DATABASE_HOST}")
print(f"Qdrant Cloud: {settings.USE_QDRANT_CLOUD}")
```

---

## 🧠 Design Patterns Used

### 1. **Singleton Pattern**

Used for database connections and model instances.

```python
# llm_engineering/application/networks/base.py

class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class EmbeddingModelSingleton(metaclass=SingletonMeta):
    def __init__(self, model_id="..."):
        self._model = SentenceTransformer(model_id)
```

**Why**: Ensure only ONE instance of expensive resources (models, DB connections).

---

### 2. **Generic Types**

Used for type-safe document operations.

```python
# llm_engineering/domain/base/nosql.py

from typing import Generic, TypeVar

T = TypeVar("T", bound="NoSQLBaseDocument")

class NoSQLBaseDocument(BaseModel, Generic[T]):
    id: UUID4
    
    @classmethod
    def find(cls, id: UUID4) -> Optional[T]:
        ...
```

**Why**: Type safety when working with different document types.

---

### 3. **Strategy Pattern**

Used for crawler dispatching.

```python
# llm_engineering/application/crawlers/dispatcher.py

class CrawlerDispatcher:
    def __init__(self):
        self._crawlers = {}
    
    def register(self, domain: str, crawler: BaseCrawler):
        self._crawlers[domain] = crawler
    
    def get_crawler(self, url: str) -> BaseCrawler:
        domain = urlparse(url).netloc
        return self._crawlers.get(domain, self._default_crawler)
```

**Why**: Easily add new crawlers without modifying existing code.

---

## 📝 Exercise: Explore the Codebase

### Task 1: Trace the Dependency Flow

1. Start at `infrastructure/inference_pipeline_api.py`
2. Find what it imports from `model/`
3. Find what `model/` imports from `application/`
4. Find what `application/` imports from `domain/`

**Goal**: Understand how layers depend on each other.

### Task 2: Identify All Singleton Classes

Find all classes using `SingletonMeta` in the codebase.

**Hint**: Search for `metaclass=SingletonMeta`

### Task 3: Map Data Categories

List all places where `DataCategory` enum is used.

**Hint**: Use grep search for `DataCategory`

---

## 🎓 Knowledge Check

1. **What is the dependency flow direction?**
   - Answer: `infrastructure → model → application → domain`

2. **Why use Singleton for database connections?**
   - Answer: Avoid creating multiple expensive connections

3. **What does `Generic[T]` provide?**
   - Answer: Type safety across different document types

4. **Where is configuration stored?**
   - Answer: In `settings.py` using Pydantic, loaded from `.env`

---

## 🔗 Next Session

**Session 1.2**: Domain Layer - Data Modeling

We'll dive deep into:
- MongoDB document design with `NoSQLBaseDocument`
- Vector database design with `VectorBaseDocument`
- How to create custom document models
- CRUD operations implementation

---

## 📚 Additional Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [Singleton Pattern](https://refactoring.guru/design-patterns/singleton)

---

**Estimated Time**: 2-3 hours

**Prerequisites**: Basic Python knowledge

**Outcome**: You'll understand the project architecture and be able to navigate the codebase effectively.
