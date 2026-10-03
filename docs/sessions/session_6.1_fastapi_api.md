# Session 6.1: FastAPI REST API

## 🎯 Learning Objectives

By the end of this session, you will:
- Read the FastAPI app that exposes the LLM as an HTTP service
- Understand the `Inference` and `DeploymentStrategy` domain abstractions
- Use `InferenceExecutor` to build a prompt and call SageMaker
- Add request validation and error handling with Pydantic
- Run the API locally with uvicorn

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI process                                │
│                                                                        │
│   POST /rag  { "query": "..." }                                        │
│        │                                                               │
│        ▼                                                               │
│   rag(query)  ──► ContextRetriever.search  (Session 6.2)              │
│        │                                                               │
│        ▼                                                               │
│   call_llm_service(query, context)                                     │
│        │                                                               │
│        ▼                                                               │
│   LLMInferenceSagemakerEndpoint  (implements Inference ABC)           │
│        │  boto3 sagemaker-runtime invoke_endpoint                      │
│        ▼                                                               │
│   InferenceExecutor.execute()  ──►  answer  (HTTP 200)                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Domain Abstractions

```
domain/inference.py
├── DeploymentStrategy (ABC)   ← implemented by SagemakerHuggingfaceStrategy (Session 5.3)
└── Inference (ABC)            ← implemented by LLMInferenceSagemakerEndpoint
        set_payload(inputs, parameters)
        inference() -> dict
```

The `Inference` ABC lets the API depend on an interface, not boto3. A future local-vLLM executor only has to implement the same two methods.

---

## 📁 Key Files Explained

### 1. `domain/inference.py` - The Contract

```python
# llm_engineering/domain/inference.py
from abc import ABC, abstractmethod


class DeploymentStrategy(ABC):
    @abstractmethod
    def deploy(self, model, endpoint_name: str, endpoint_config_name: str) -> None:
        pass


class Inference(ABC):
    """An abstract class for performing inference."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def set_payload(self, inputs, parameters=None):
        pass

    @abstractmethod
    def inference(self):
        pass
```

**Key Concepts**:
- **`Inference` has two abstract methods**: `set_payload` (build the request) and `inference` (execute it). The API only programs against these.
- **`DeploymentStrategy`** is implemented in infrastructure (Session 5.3), keeping the deploy mechanism swappable.

---

### 2. `model/inference/inference.py` - SageMaker Client

```python
# llm_engineering/model/inference/inference.py
class LLMInferenceSagemakerEndpoint(Inference):
    def __init__(
        self,
        endpoint_name: str,
        default_payload: Optional[Dict[str, Any]] = None,
        inference_component_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.client = boto3.client(
            "sagemaker-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.endpoint_name = endpoint_name
        self.payload = default_payload if default_payload else self._default_payload()
        self.inference_component_name = inference_component_name

    def _default_payload(self) -> Dict[str, Any]:
        return {
            "inputs": "How is the weather?",
            "parameters": {
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "top_p": settings.TOP_P_INFERENCE,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "return_full_text": False,
            },
        }

    def set_payload(self, inputs: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        self.payload["inputs"] = inputs
        if parameters:
            self.payload["parameters"].update(parameters)

    def inference(self) -> Dict[str, Any]:
        try:
            invoke_args = {
                "EndpointName": self.endpoint_name,
                "ContentType": "application/json",
                "Body": json.dumps(self.payload),
            }
            if self.inference_component_name not in ["None", None]:
                invoke_args["InferenceComponentName"] = self.inference_component_name
            response = self.client.invoke_endpoint(**invoke_args)
            response_body = response["Body"].read().decode("utf8")
            return json.loads(response_body)
        except Exception:
            logger.exception("SageMaker inference failed.")
            raise
```

**Key Concepts**:
- **`sagemaker-runtime` client** is distinct from the `sagemaker` control-plane client used for deployment.
- **Payload shape matches the TGI container**: `{"inputs": ..., "parameters": {...}}`.
- **`return_full_text=False`** makes the container return only the generated continuation, not the prompt + continuation.
- **`InferenceComponentName`** is added only when set, which is required for component-based endpoints.
- **Error handling re-raises** after logging, so the API layer can translate it to an HTTP error.

---

### 3. `model/inference/run.py` - Prompt Builder and Executor

```python
# llm_engineering/model/inference/run.py
class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        context: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}
            """
        else:
            self.prompt = prompt

    def execute(self) -> str:
        self.llm.set_payload(
            inputs=self.prompt.format(query=self.query, context=self.context),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
            },
        )
        answer = self.llm.inference()[0]["generated_text"]
        return answer
```

**Key Concepts**:
- **`InferenceExecutor` is the template-method glue**: it formats the prompt, sets the payload, calls the model, and extracts `[0]["generated_text"]`.
- **`repetition_penalty=1.1`** discourages loops, which is important for a small `max_new_tokens` budget.
- **`temperature=0.01`** (from settings) makes the endpoint nearly deterministic, appropriate for a factual assistant.
- **`prompt` is overridable**, so callers can supply a different template without subclassing.

---

### 4. `infrastructure/inference_pipeline_api.py` - The FastAPI App

```python
# llm_engineering/infrastructure/inference_pipeline_api.py
import opik
from fastapi import FastAPI, HTTPException
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

configure_opik()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()
    return answer


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )
    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Key Concepts**:
- **`configure_opik()` runs at import time** so every endpoint call is traced.
- **`QueryRequest` / `QueryResponse`** are the request/response contracts. FastAPI validates the body automatically and returns **422** on a missing `query`.
- **`response_model=QueryResponse`** documents and enforces the output shape in the OpenAPI schema.
- **`@opik.track` on `call_llm_service` and `rag`** creates nested spans under the request trace.
- **Error handling**: any exception becomes a `500` with the exception text. In production, log the detail and return a generic message to avoid leaking internals.
- **The endpoint is `async` but `rag()` is synchronous and blocks the event loop.** For a single-user local service this is fine. Under load, move the work to a thread pool (`await run_in_threadpool(rag, request.query)`) or make the whole chain async.

---

### 5. `tools/ml_service.py` - The Runner

```python
# tools/ml_service.py
from llm_engineering.infrastructure.inference_pipeline_api import app  # noqa

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tools.ml_service:app", host="0.0.0.0", port=8000, reload=True)
```

**Key Concepts**:
- **`reload=True`** hot-reloads on code changes - development only.
- **`host="0.0.0.0"`** binds all interfaces, which is convenient in Docker and risky on an open network.
- The string form `"tools.ml_service:app"` is required for reload workers to re-import the app.

---

### 6. `infrastructure/opik_utils.py` - Tracing Setup

```python
# llm_engineering/infrastructure/opik_utils.py
def configure_opik() -> None:
    if settings.COMET_API_KEY and settings.COMET_PROJECT:
        try:
            client = OpikConfigurator(api_key=settings.COMET_API_KEY)
            default_workspace = client._get_default_workspace()
        except Exception:
            logger.warning("Default workspace not found. Setting workspace to None and enabling interactive mode.")
            default_workspace = None

        os.environ["OPIK_PROJECT_NAME"] = settings.COMET_PROJECT
        opik.configure(api_key=settings.COMET_API_KEY, workspace=default_workspace, use_local=False, force=True)
        logger.info("Opik configured successfully.")
    else:
        logger.warning("COMET_API_KEY and COMET_PROJECT are not set. ...")
```

**Key Concepts**:
- **Best-effort configuration**: if the key or workspace is missing, it warns and continues rather than crashing the API.
- `OPIK_PROJECT_NAME` routes traces into the configured project (Session 7.2).

---

## 🛠️ Hands-On: Run and Call the API

### Step 1: Prerequisites

- Qdrant running with embedded chunks (`docker compose up -d`, feature engineering done).
- A SageMaker endpoint created (Session 5.3), or a local stand-in.

### Step 2: Start the service

```bash
python -m tools.ml_service
# Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Call the endpoint

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"My name is Paul Iusztin. Write a LinkedIn post about RAG.\"}"
```

Expected:

```json
{ "answer": "RAG (Retrieval-Augmented Generation) is ..." }
```

### Step 4: Inspect the OpenAPI schema

Open `http://localhost:8000/docs` for the interactive Swagger UI. The `QueryRequest` and `QueryResponse` models are documented automatically.

### Step 5: Validate error handling

```bash
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d "{}"
# 422 Unprocessable Entity (missing 'query')
```

---

## 📝 Exercise: Add a Health and a Streaming Endpoint

### Task 1: Health check

```python
@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Task 2: Streaming (sketch)

Add `POST /rag/stream` using `StreamingResponse` from `fastapi.responses`:

```python
from fastapi.responses import StreamingResponse

@app.post("/rag/stream")
async def rag_stream(request: QueryRequest):
    def token_gen():
        answer = rag(query=request.query)
        for token in answer.split():
            yield token + " "
    return StreamingResponse(token_gen(), media_type="text/plain")
```

**Goal**: Understand how the current design (blocking sync function) makes true token streaming hard, and where you would need an async inference client.

---

## 🐛 Common Pitfalls

- **Blocking the event loop**: `rag()` is synchronous. One slow SageMaker call blocks other requests. Use a thread pool under concurrency.
- **500 leaks internals**: `detail=str(e)` exposes stack-derived messages. Log them server-side instead.
- **Missing env**: `LLMInferenceSagemakerEndpoint` needs AWS credentials; without them boto3 fails at request time.
- **`response_model` mismatch**: returning a key other than `answer` will fail validation.
- **CORS**: browser clients need `CORSMiddleware`; it is not configured by default.

---

## 🎓 Knowledge Check

1. **What two methods must any `Inference` implementation provide?**
   - Answer: `set_payload` and `inference`.

2. **What request shape does the SageMaker TGI container expect?**
   - Answer: `{"inputs": "...", "parameters": {...}}`.

3. **What does `return_full_text=False` do?**
   - Answer: Returns only the generated continuation, not the prompt prefix.

4. **Why are `QueryRequest` and `QueryResponse` Pydantic models?**
   - Answer: They validate input/output and generate the OpenAPI schema.

5. **What is the main scalability weakness of this app?**
   - Answer: A synchronous blocking call inside an async endpoint, which stalls the event loop.

6. **How is tracing turned on?**
   - Answer: `configure_opik()` at import plus `@opik.track` decorators.

---

## 🔗 Next Session

**Session 6.2**: RAG Inference Flow

We trace a query from retrieval through context building to generation, and inspect the Opik trace metadata.

---

## 📚 Additional Resources

- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)
- [Uvicorn](https://www.uvicorn.org/)
- [Opik Tracing](https://www.comet.com/docs/opik/)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 4.1, 4.2, 5.3

**Outcome**: You can run, call, validate, and extend the inference API, and you understand its domain abstractions.
