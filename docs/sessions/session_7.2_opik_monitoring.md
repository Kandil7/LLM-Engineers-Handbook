# Session 7.2: Prompt Monitoring with Opik

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how Opik traces the RAG path with `@opik.track`
- Read the trace hierarchy produced by one query
- Use `opik_context.update_current_trace` for tags and metadata
- Interpret token counts and latency per span
- Add a custom traced span

---

## 🏗️ Architecture Overview

```
rag(query)                                    @opik.track → root trace
├── ContextRetriever.search                   @opik.track
│   ├── SelfQuery.generate                    @opik.track
│   ├── QueryExpansion.generate               @opik.track
│   ├── (parallel Qdrant searches)            not traced (fast)
│   └── Reranker.generate                     @opik.track
└── call_llm_service(query, context)          @opik.track
    └── InferenceExecutor / SageMaker call    (HTTP span)

opik_context.update_current_trace(tags=["rag"], metadata={...tokens...})
```

Every `@opik.track`-decorated function becomes a span nested under its caller. Opik (by Comet ML) collects inputs, outputs, timing, and any metadata you attach.

---

## 📁 Key Files Explained

### 1. `infrastructure/opik_utils.py` - Configuration

```python
# llm_engineering/infrastructure/opik_utils.py
import os

import opik
from loguru import logger
from opik.configurator.configure import OpikConfigurator

from llm_engineering import settings


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
        logger.warning("COMET_API_KEY and COMET_PROJECT are not set. Set them to enable prompt monitoring with Opik.")
```

**Key Concepts**:
- **`configure_opik()` must run before any traced call**; the API calls it at import time.
- **`OPIK_PROJECT_NAME` set from `COMET_PROJECT`** routes traces into the same project name used by training experiments.
- **`use_local=False`** sends traces to the hosted Opik service. Set it to `True` for a fully local Opik server.
- **Best-effort**: if configuration fails, it warns and the application keeps running untraced.

---

### 2. The Traced Call Chain

**Root span - `rag`**:

```python
# llm_engineering/infrastructure/inference_pipeline_api.py
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
```

**Key Concepts**:
- **`opik_context.update_current_trace`** enriches the root trace with a `rag` tag and model/token metadata. This is what makes a trace searchable and comparable.
- **Token counts are computed with the generator's tokenizer** (`misc.compute_num_tokens` uses `AutoTokenizer.from_pretrained(settings.HF_MODEL_ID)`), so they reflect what the endpoint processes.
- **The same key names** (`query_tokens`, `context_tokens`, `answer_tokens`) appear consistently, enabling dashboards that aggregate over traces.

**Retrieval spans**:

```python
# llm_engineering/application/rag/retriever.py
    @opik.track(name="ContextRetriever.search")
    def search(self, query: str, k: int = 3, expand_to_n_queries: int = 3) -> list:
```

```python
# llm_engineering/application/rag/query_expanison.py
    @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
```

```python
# llm_engineering/application/rag/self_query.py
    @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
```

```python
# llm_engineering/application/rag/reranking.py
    @opik.track(name="Reranker.generate")
    def generate(self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
```

**Key Concepts**:
- **`name=` overrides the span name** for readability (for example `SelfQuery.generate` instead of `SelfQuery.generate.<module>`).
- Because `ContextRetriever.search` is traced and it calls the three sub-steps, Opik builds a tree: `search` contains `SelfQuery`, `QueryExpansion`, and `Reranker`.
- **Inputs and outputs are captured automatically**. You can see the extracted author, the generated query variants, and the reranked chunks in the trace.

**LLM span**:

```python
@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()
    return answer
```

**Key Concepts**:
- The LLM call is a sibling span under `rag`, separate from retrieval. This split lets you attribute latency to retrieval versus generation.
- **No explicit token usage is set on this span** because the SageMaker invocation returns text, not usage stats. The `rag` trace computes tokens after the fact.

---

## 🔬 Deep Dive: What a Good Trace Shows

```
rag (trace)
  tags: ["rag"]
  metadata: model_id, embedding_model_id, temperature,
            query_tokens, context_tokens, answer_tokens
  │
  ├─ ContextRetriever.search            ~120 ms
  │    input:  query string
  │    output: list of EmbeddedChunk
  │    ├─ SelfQuery.generate            ~300 ms   (LLM call)
  │    │    output: author_id + author_full_name
  │    ├─ QueryExpansion.generate       ~400 ms   (LLM call)
  │    │    output: 3 variant queries
  │    └─ Reranker.generate             ~80 ms    (cross-encoder)
  │         output: top-3 chunks
  │
  └─ call_llm_service                   ~1.5 s
       input:  prompt (query + context)
       output: answer
```

**What to look for**:
- **Latency split**: two LLM calls (SelfQuery, QueryExpansion) plus the generation call dominate latency. Retrieval-only logic is fast.
- **Empty retrieval**: if `context_tokens` is 0, the model answered without grounding. Investigate the author filter and collection contents.
- **Author extraction**: confirm `SelfQuery` returned the right name and not `none`.
- **Variant quality**: read the expanded queries; poor variants correlate with poor recall.

**Why the two extra LLM calls matter**: SelfQuery and QueryExpansion add latency but improve precision and recall. If latency is critical, cache them or disable expansion for simple queries. The trace is how you find that tradeoff.

---

## 🛠️ Hands-On: Produce and Inspect a Trace

### Step 1: Configure

```env
COMET_API_KEY=your-key
COMET_PROJECT=twin
```

### Step 2: Focus on generation via the retrieval harness

```bash
python -m tools.rag
```

This calls `ContextRetriever.search` directly. Even without the API, the `search`, `SelfQuery`, `QueryExpansion`, and `Reranker` spans are recorded.

### Step 3: Run the full RAG

```bash
python -m tools.ml_service
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" \
  -d "{\"query\": \"My name is Paul Iusztin. Draft a post about RAG.\"}"
```

### Step 4: Inspect in Opik

Open the Opik project for `twin`. You should see the `rag` trace with the tag and metadata, the retrieval subtree, and the LLM span. Check:
- `query_tokens`, `context_tokens`, `answer_tokens`.
- The SelfQuery output (author id/name).
- The three expanded queries.

---

## 📝 Exercise: Add a Custom Span for Retrieval Filtering

### Task

Trace the Qdrant filtering decision explicitly.

```python
import opik
from opik import opik_context

@opik.track(name="build_qdrant_filter")
def build_query_filter(author_id):
    opik_context.update_current_span(metadata={"has_author_filter": author_id is not None})
    ...
```

1. Wrap `_search_data_category`'s filter construction in this traced helper.
2. Re-run a query with and without an author name.
3. Confirm the span shows `has_author_filter: true/false`.

**Goal**: Practice creating custom spans and span-level metadata, so any decision inside the pipeline becomes observable.

---

## 🐛 Common Pitfalls

- **Opik not configured**: without a key, traces are silently dropped and only a warning is logged. Verify the log line "Opik configured successfully."
- **Decorating nested functions too much**: every traced call adds overhead. Trace meaningful units (retrieval stages, LLM calls), not tight loops.
- **PII in traces**: queries and context are stored. Avoid sending sensitive data through the API, or use local Opik (`use_local=True`).
- **Consistent metadata keys**: renaming `context_tokens` breaks dashboards that aggregate on it. Keep names stable.

---

## 🎓 Knowledge Check

1. **What decorator creates a trace or span?**
   - Answer: `@opik.track`.

2. **How do you attach tags and metadata to the whole request?**
   - Answer: `opik_context.update_current_trace(tags=..., metadata=...)`.

3. **Why is `call_llm_service` a separate span from retrieval?**
   - Answer: To attribute latency to generation versus retrieval.

4. **What does the `Reranker.generate` span capture?**
   - Answer: Its inputs (query, chunks) and output (top-k chunks).

5. **Where do Opik traces go by default?**
   - Answer: The hosted Opik service routed by `OPIK_PROJECT_NAME`, unless `use_local=True`.

6. **What does `context_tokens = 0` indicate?**
   - Answer: Retrieval returned no usable context, so the answer was ungrounded.

---

## 🔗 Next Session

**Session 7.3**: Model Evaluation

We score SFT, DPO, and Instruct models with an LLM-as-a-judge pipeline.

---

## 📚 Additional Resources

- [Opik Documentation](https://www.comet.com/docs/opik/)
- [Opik Tracing Quickstart](https://www.comet.com/docs/opik/tracing)
- [Comet ML](https://www.comet.com/)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 6.1, 6.2, 7.1

**Outcome**: You can read and extend Opik traces, and diagnose retrieval and generation issues from span data.
