# Session 8.1: Docker & Local Infrastructure

## 🎯 Learning Objectives

By the end of this session, you will:
- Read the production `Dockerfile` and why it installs Chrome
- Run the local MongoDB + Qdrant stack with Docker Compose
- Build and run the application image
- Use Poe the Poet tasks to drive infrastructure and pipelines
- Understand the two-layer local topology (services vs application)

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        Local development                            │
│                                                                      │
│  docker compose (services)                                          │
│    ├── mongo    :27017   (llm_engineering / llm_engineering)        │
│    └── qdrant   :6333/6334                                          │
│                                                                      │
│  application container  (optional)                                  │
│    └── llmtwin image  ──docker run --network host──►  services      │
│                                                                      │
│  ZenML server (local)  :8237                                        │
└────────────────────────────────────────────────────────────────────┘
```

Two distinct things run in Docker: **infrastructure services** (long-lived databases) and an optional **application image** (the code, run per pipeline invocation).

---

## 📁 Key Files Explained

### 1. `Dockerfile` - The Application Image

```dockerfile
# Dockerfile
FROM python:3.11-slim-bullseye AS release

ENV WORKSPACE_ROOT=/app/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.8.3
ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_NO_INTERACTION=1

# Install Google Chrome
RUN apt-get update -y && \
    apt-get install -y gnupg wget curl --no-install-recommends && \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-linux-signing-key.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/google-linux-signing-key.gpg] https://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update -y && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Install other system dependencies.
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends build-essential \
    gcc \
    python3-dev \
    libglib2.0-dev \
    libnss3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"
RUN poetry config installer.max-workers 20

WORKDIR $WORKSPACE_ROOT

COPY pyproject.toml poetry.lock $WORKSPACE_ROOT

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-cache --without dev && \
    poetry self add 'poethepoet[poetry_plugin]' && \
    rm -rf ~/.cache/pypoetry/cache/ && \
    rm -rf ~/.cache/pypoetry/artifacts/

COPY . $WORKSPACE_ROOT
```

**Key Concepts**:
- **Chrome is required** because the crawlers use Selenium (Session 2.1). The Google-signed apt repo installs `google-chrome-stable` so `chromedriver-autoinstaller` can pair a matching driver at runtime.
- **`PYTHONDONTWRITEBYTECODE` + `PYTHONUNBUFFERED`**: cleaner logs and a smaller image; output is not buffered so container logs stream in real time.
- **Layer caching**: `pyproject.toml` and `poetry.lock` are copied **before** the source, so dependency installation is cached unless the lock file changes.
- **`virtualenvs.create false`**: dependencies install into the system Python, which is fine inside a container and avoids an extra venv layer.
- **`--without dev`**: no test/lint tools in the production image.
- **`poetry self add 'poethepoet[poetry_plugin]'`**: Poe tasks are available inside the image, so `poetry poe ...` works in the container.
- **No build cache cleanup beyond pip**: `rm -rf ~/.cache/pypoetry/...` trims the image.

**Multi-arch note**: The CD workflow builds with `--platform linux/amd64` (Session 8.2) because SageMaker containers are amd64. On Apple Silicon you must build for amd64 explicitly.

---

### 2. `docker-compose.yml` - Local Services

```yaml
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
    expose:
      - 6333
      - 6334
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
- **No application service** in Compose: the databases run as services; the app runs via `docker run` (or locally) and connects over the host network.
- **Credentials** match the default `DATABASE_HOST` in `settings.py` (`llm_engineering:llm_engineering`).
- **Qdrant exposes 6333 (REST) and 6334 (gRPC)**; the client uses 6333.
- **`restart: always`** keeps databases up across reboots.

### 3. Poe tasks - Docker Orchestration

```toml
# pyproject.toml (excerpt)
build-docker-image = "docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile ."
run-docker-end-to-end-data-pipeline = "docker run --rm --network host --shm-size=2g --env-file .env llmtwin poetry poe --no-cache --run-end-to-end-data"
bash-docker-container = "docker run --rm -it --network host --env-file .env llmtwin bash"

local-docker-infrastructure-up = "docker compose up -d"
local-docker-infrastructure-down = "docker compose stop"
```

**Key Concepts**:
- **`--network host`** lets the container reach `mongo`/`qdrant` on localhost exactly as the local code expects. (On Docker Desktop for Windows, use the host's reachable address or run services in the same network.)
- **`--shm-size=2g`** raises shared memory, needed by Chrome/headless Chromium to avoid crashes.
- **`--env-file .env`** injects configuration and secrets without baking them into the image.
- **`poetry poe --no-cache --run-end-to-end-data`** runs the whole data pipeline inside the container.

---

## 🛠️ Hands-On: Build and Run Locally

### Step 1: Start the databases

```bash
docker compose up -d
docker ps --filter name=llm_engineering
```

### Step 2: Verify connectivity from Python

```bash
python -c "from llm_engineering.infrastructure.db.mongo import connection; print(connection.get_database('twin').command('ping'))"
python -c "from llm_engineering.infrastructure.db.qdrant import connection; print([c.name for c in connection.get_collections().collections])"
```

### Step 3: Build the image

```bash
docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile .
```

### Step 4: Open a shell in the image

```bash
docker run --rm -it --network host --env-file .env llmtwin bash
# inside:
poetry poe --no-cache --run-feature-engineering
```

### Step 5: Tear down services

```bash
docker compose stop       # keeps volumes
# docker compose down -v  # also deletes data (dangerous)
```

---

## 📝 Exercise: Add a Redis Service

### Task

Extend the local stack with a cache.

1. Add a `redis` service to `docker-compose.yml` (image `redis:7`, port 6379).
2. Add `REDIS_HOST` / `REDIS_PORT` to `Settings`.
3. Add a Poe task `local-infrastructure-status = "docker compose ps"`.
4. Verify a Python `redis` client can `PING`.

**Goal**: Practice extending both Compose and the settings layer consistently.

---

## 🐛 Common Pitfalls

- **Chrome missing at runtime**: a crawler fails with "chromedriver not found". The Dockerfile installs Chrome; if you run outside Docker, install Chrome locally.
- **Network mismatch on Windows**: `--network host` behaves differently on Docker Desktop. Prefer running the app on the host and only databases in Compose.
- **`down -v` deletes data**: only use it when you intend to wipe MongoDB and Qdrant.
- **Image size**: installing Chrome and build tools makes the image large; that is the cost of crawling in-container.
- **Architecture**: building on ARM without `--platform linux/amd64` produces an image SageMaker cannot run.

---

## 🎓 Knowledge Check

1. **Why does the Dockerfile install Google Chrome?**
   - Answer: The Selenium crawlers need a browser and a matching driver.

2. **Why are `pyproject.toml` and `poetry.lock` copied before the source?**
   - Answer: To cache the dependency-install layer.

3. **What does `--shm-size=2g` prevent?**
   - Answer: Headless-Chrome crashes from a too-small shared memory segment.

4. **Which service ports does Compose expose for Qdrant?**
   - Answer: 6333 (REST) and 6334 (gRPC).

5. **What is `--without dev` for?**
   - Answer: Excluding dev-only tools (ruff, pytest, pre-commit) from the production image.

6. **How does the app container reach the databases?**
   - Answer: Over the host network (`--network host`) using localhost addresses.

---

## 🔗 Next Session

**Session 8.2**: CI/CD with GitHub Actions

We automate lint, tests, secret scanning, and image publication.

---

## 📚 Additional Resources

- [Docker Compose](https://docs.docker.com/compose/)
- [Poetry](https://python-poetry.org/docs/)
- [Poe the Poet](https://poethepoet.natn.io/)
- [Chrome for Linux](https://www.google.com/chrome/)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 1.3, 2.1

**Outcome**: You can build the application image, run the local service stack, and drive pipelines through Poe tasks.
