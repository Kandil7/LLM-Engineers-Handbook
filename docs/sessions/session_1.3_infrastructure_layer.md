# Session 1.3: Infrastructure Layer - Database Connections

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how MongoDB and Qdrant connections are created as singletons
- Know how `settings` is loaded from `.env`, defaults, and the ZenML secret store
- Read and write JSON artifacts with `JsonFileManager`
- Start the local database stack with Docker Compose
- Verify both stores with real diagnostic queries

---

## 🏗️ Architecture Overview

The infrastructure layer is the outermost layer. It owns everything that talks to the outside world: databases, AWS, HTTP APIs, and tracing. The domain layer imports connections from here, but infrastructure never imports pipelines.

```
┌──────────────────────────────────────────────────────────────┐
│                  infrastructure/ (outermost)                  │
│                                                               │
│   db/                    aws/              files_io.py         │
│   ├── mongo.py           ├── deploy/       opik_utils.py       │
│   └── qdrant.py          └── roles/        inference_pipeline_api.py
│        │                      │                   │            │
│        ▼                      ▼                   ▼            │
│   MongoClient            SageMaker           FastAPI app       │
│   QdrantClient           IAM roles                             │
└──────────────────────────────────────────────────────────────┘
                 ▲                       ▲
                 │ imports               │ imports
          domain/base/*.py        model/ + application/
```

### Connection Flow

```
.env  ──►  Settings (pydantic-settings)  ──►  settings singleton
                                                    │
                     ┌──────────────────────────────┴──────────────────────┐
                     ▼                                                      ▼
          MongoDatabaseConnector()                              QdrantDatabaseConnector()
          returns MongoClient                                   returns QdrantClient
                     │                                                      │
          connection.get_database(settings.DATABASE_NAME)       connection.search(...)
                     │                                                      │
          domain/base/nosql.py                                  domain/base/vector.py
```

---

## 📁 Key Files Explained

### 1. `llm_engineering/settings.py` - Central Configuration

**Purpose**: One typed `Settings` object for the entire codebase.

```python
# llm_engineering/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from zenml.client import Client
from zenml.exceptions import EntityExistsError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Required settings even when working locally. ---
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "twin"

    # --- Required settings when deploying the code. ---
    DATABASE_HOST: str = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "twin"

    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None

    # --- Optional settings used to tweak the code. ---
    HF_MODEL_ID: str = "mlabonne/TwinLlama-3.1-8B-DPO"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096

    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"
```

**Derived property**:

```python
    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)
        return max_token_window
```

The 90% factor reserves headroom for prompt overhead, so context building never exceeds the real model window.

**ZenML secret-store integration**:

```python
    @classmethod
    def load_settings(cls) -> "Settings":
        try:
            logger.info("Loading settings from the ZenML secret store.")
            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError):
            logger.warning(
                "Failed to load settings from the ZenML secret store. "
                "Defaulting to loading the settings from the '.env' file."
            )
            settings = Settings()
        return settings

    def export(self) -> None:
        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()
        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning("Secret 'scope' already exists. Delete it manually ...")


settings = Settings.load_settings()
```

**Key Concepts**:
- **Priority order**: ZenML secret store → `.env` → typed defaults. Locally you only need `.env`.
- **`settings` is a module-level singleton**, imported as `from llm_engineering.settings import settings`.
- **Secrets never hard-coded**: `OPENAI_API_KEY`, `DATABASE_HOST`, and AWS keys are all `str | None` with safe local defaults.
- **`export()`** pushes the current settings into ZenML so remote pipeline runs (CI, cloud) read the same config without shipping `.env`.

---

### 2. `llm_engineering/infrastructure/db/mongo.py` - MongoDB Connector

**Purpose**: A lazy, process-wide `MongoClient`.

```python
# llm_engineering/infrastructure/db/mongo.py
from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from llm_engineering.settings import settings


class MongoDatabaseConnector:
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.DATABASE_HOST)
            except ConnectionFailure as e:
                logger.error(f"Couldn't connect to the database: {e!s}")
                raise

        logger.info(f"Connection to MongoDB with URI successful: {settings.DATABASE_HOST}")
        return cls._instance


connection = MongoDatabaseConnector()
```

**Key Concepts**:
- **Singleton via `__new__`**: `MongoDatabaseConnector()` always returns the same `MongoClient`. Note that `MongoClient` itself is already thread-safe and pools connections.
- **`connection` is actually a `MongoClient`**, not a connector instance. The module-level name reads like a connection, which is how `domain/base/nosql.py` uses it.
- **Lazy** because the module is only imported when first needed; the client connects on first operation.

**How the domain layer consumes it**:

```python
# llm_engineering/domain/base/nosql.py
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings

_database = connection.get_database(settings.DATABASE_NAME)
```

Every document class shares one `_database` handle and selects its collection with `get_collection_name()`.

---

### 3. `llm_engineering/infrastructure/db/qdrant.py` - Qdrant Connector

**Purpose**: The same singleton pattern, with a cloud/local switch.

```python
# llm_engineering/infrastructure/db/qdrant.py
class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __new__(cls, *args, **kwargs) -> QdrantClient:
        if cls._instance is None:
            try:
                if settings.USE_QDRANT_CLOUD:
                    cls._instance = QdrantClient(
                        url=settings.QDRANT_CLOUD_URL,
                        api_key=settings.QDRANT_APIKEY,
                    )
                    uri = settings.QDRANT_CLOUD_URL
                else:
                    cls._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT,
                    )
                    uri = f"{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"

                logger.info(f"Connection to Qdrant DB with URI successful: {uri}")
            except UnexpectedResponse:
                logger.exception("Couldn't connect to Qdrant.", ...)
                raise

        return cls._instance


connection = QdrantDatabaseConnector()
```

**Key Concepts**:
- **One flag switches environments**: `USE_QDRANT_CLOUD=false` → local Docker at `localhost:6333`; `true` → Qdrant Cloud with `QDRANT_APIKEY`.
- Local Qdrant needs no auth; the Docker network exposes ports 6333 (HTTP/REST) and 6334 (gRPC).
- The connector is reused by `VectorBaseDocument` through `connection.search`, `connection.scroll`, `connection.upsert`, `connection.create_collection`.

---

### 4. `llm_engineering/infrastructure/files_io.py` - JSON Artifacts

**Purpose**: A small, explicit reader/writer for the JSON files under `data/artifacts/`.

```python
# llm_engineering/infrastructure/files_io.py
import json
from pathlib import Path


class JsonFileManager:
    @classmethod
    def read(cls, filename: str | Path) -> list:
        file_path: Path = Path(filename)
        try:
            with file_path.open("r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path=}' does not exist.") from None
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                msg=f"File '{file_path=}' is not properly formatted as JSON.",
                doc=e.doc,
                pos=e.pos,
            ) from None

    @classmethod
    def write(cls, filename: str | Path, data: list | dict) -> Path:
        file_path: Path = Path(filename)
        file_path = file_path.resolve().absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

        return file_path
```

**Key Concepts**:
- **Error messages carry the path** (`{file_path=}`), so failures point at the exact file.
- **`write` creates parent directories** (`mkdir(parents=True, exist_ok=True)`), which is why `data/artifacts/*.json` can be written on first run.
- Used by the export steps and the training/inference scripts that bridge ZenML artifacts and Hugging Face datasets.

---

### 5. `docker-compose.yml` - Local Database Stack

**Purpose**: Bring up MongoDB and Qdrant locally with one command.

```yaml
# docker-compose.yml
services:
  mongo:
    image: mongo:latest
    container_name: "llm_engineering_mongo"
    environment:
      MONGO_INITDB_ROOT_USERNAME: "llm_engineering"
      MONGO_INITDB_ROOT_PASSWORD: "llm_engineering"
    ports:
      - 27017:27017
    volumes:
      - mongo_data:/data/db
    networks:
      - local
    restart: always

  qdrant:
    image: qdrant/qdrant:latest
    container_name: "llm_engineering_qdrant"
    ports:
      - 6333:6333
      - 6334:6334
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - local
    restart: always

volumes:
  mongo_data:
  qdrant_data:

networks:
  local:
    driver: bridge
```

**Key Concepts**:
- **Named volumes** (`mongo_data`, `qdrant_data`) persist data across `docker compose down` (without `-v`).
- MongoDB uses `mongo:latest` with root credentials matching the default `DATABASE_HOST`.
- Qdrant has no credentials locally, matching `USE_QDRANT_CLOUD=false`.

> ⚠️ **Windows note**: On Windows/WSL the class-method line `networks: local: driver: bridge` is standard; if you get a networking error, ensure Docker Desktop's WSL2 integration is enabled.

---

## 🔧 Local Setup

### Step 1: Start databases

```bash
docker compose up -d
```

### Step 2: Confirm containers

```bash
docker ps --filter "name=llm_engineering"
```

Expected: `llm_engineering_mongo` on `0.0.0.0:27017` and `llm_engineering_qdrant` on `0.0.0.0:6333-6334`.

### Step 3: Create the environment file

```bash
cp .env.example .env
```

Fill in at least `OPENAI_API_KEY`, `HUGGINGFACE_ACCESS_TOKEN`, and `COMET_API_KEY`. The database defaults already work.

---

## 🛠️ Hands-On: Verify Both Connections

### Test MongoDB

```python
# scripts/check_mongo.py
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings

db = connection.get_database(settings.DATABASE_NAME)
print("Mongo ping:", db.command("ping"))
print("Collections:", db.list_collection_names())
```

```bash
python -m scripts.check_mongo
```

### Test Qdrant

```python
# scripts/check_qdrant.py
from llm_engineering.infrastructure.db.qdrant import connection

print("Qdrant collections:", [c.name for c in connection.get_collections().collections])
```

### Test JSON artifacts

```python
from llm_engineering.infrastructure.files_io import JsonFileManager

data = JsonFileManager.read("data/artifacts/cleaned_documents.json")
print(f"Read {len(data)} records")

path = JsonFileManager.write("data/artifacts/_smoke_test.json", {"ok": True})
print("Wrote:", path)
```

---

## 📝 Exercise: Add a New Infrastructure Connector

### Task

Add a thin connector for a Redis cache using the same singleton pattern.

```python
# llm_engineering/infrastructure/db/redis.py
import redis
from llm_engineering.settings import settings


class RedisDatabaseConnector:
    _instance: "redis.Redis | None" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = redis.Redis(host="localhost", port=6379, decode_responses=True)
        return cls._instance


connection = RedisDatabaseConnector()
```

Then:
1. Add `REDIS_HOST` / `REDIS_PORT` to `Settings`.
2. Add a `redis` service to `docker-compose.yml`.
3. Write a `check_redis.py` script that sets and reads a key.

**Goal**: Practice the singleton + settings + compose triad the project already uses.

---

## 🎓 Knowledge Check

1. **Why is the connector a singleton, and where is the instance stored?**
   - Answer: To reuse one connection pool; stored on the class as `_instance`, returned by `__new__`.

2. **What is `connection` in `mongo.py` - a connector or a client?**
   - Answer: A `MongoClient`; the class constructor returns the client, not an instance of the class.

3. **How does the code switch between local and Qdrant Cloud?**
   - Answer: Via the `USE_QDRANT_CLOUD` boolean in `Settings`.

4. **Which settings source wins if ZenML has a `settings` secret?**
   - Answer: The ZenML secret store; `.env` and defaults are the fallback.

5. **What does `JsonFileManager.write` do that `json.dump` alone does not?**
   - Answer: Resolves the path and creates parent directories before writing.

6. **Does dropping the compose stack delete data?**
   - Answer: No, named volumes persist; only `docker compose down -v` removes them.

---

## 🔗 Next Session

**Session 2.1**: Web Crawling with Selenium

We move up into the application layer and build the data collection crawlers.

---

## 📚 Additional Resources

- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [PyMongo Tutorial](https://pymongo.readthedocs.io/en/stable/tutorial.html)
- [Qdrant Client](https://github.com/qdrant/qdrant-client)
- [ZenML Secrets](https://docs.zenml.io/user-guide/advanced-guide/secret-management)

---

**Estimated Time**: 2-3 hours

**Prerequisites**: Sessions 1.1, 1.2

**Outcome**: You can stand up the local database stack, verify both stores, and add new connectors using the project's conventions.
