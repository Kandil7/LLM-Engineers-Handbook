# Session 6.2: RAG Inference Flow

## 🎯 Learning Objectives

By the end of this session, you will:
- Trace a user query through the complete RAG inference path
- Understand how retrieved chunks become LLM context via `to_context`
- Read the content-creator prompt used for generation
- Inspect the token accounting and Opik trace metadata
- Run retrieval-only and full-generation harnesses

---

## 🏗️ Architecture Overview

```
User query: "My name is Paul Iusztin. Could you draft a LinkedIn post about RAG?"
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ContextRetriever.search(query, k=3)                                   │
│                                                                       │
│  1. SelfQuery        → author_id = Paul's user id                     │
│  2. QueryExpansion   → 3 query variants (temperature=0)               │
│  3. ThreadPool search → per variant, per collection (k//3 each)       │
│  4. dedupe (hash by id)                                               │
│  5. Reranker (cross-encoder) → keep_top_k = k                         │
└──────────────────────────────────────────────────────────────────────┘
        │  list[EmbeddedChunk]
        ▼
EmbeddedChunk.to_context(documents)
        │  a single formatted string
        ▼
call_llm_service(query, context)
        │  InferenceExecutor builds the content-creator prompt
        ▼
LLMInferenceSagemakerEndpoint.inference()
        │
        ▼
answer  →  Response + Opik trace metadata
```

---

## 📁 Key Files Explained

### 1. `retriever.py` - Multi-Stage Retrieval

```python
# llm_engineering/application/rag/retriever.py
class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)

    @opik.track(name="ContextRetriever.search")
    def search(self, query: str, k: int = 3, expand_to_n_queries: int = 3) -> list:
        query_model = Query.from_str(query)

        query_model = self._metadata_extractor.generate(query_model)
        logger.info(f"Successfully extracted the author_full_name = {query_model.author_full_name} from the query.")

        n_generated_queries = self._query_expander.generate(query_model, expand_to_n=expand_to_n_queries)
        logger.info(f"Successfully generated {len(n_generated_queries)} search queries.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in n_generated_queries]

            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(f"{len(n_k_documents)} documents retrieved successfully")

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        return k_documents
```

**Key Concepts**:
- **`Query.from_str`** normalizes the raw string into a `Query` value object.
- **`SelfQuery`** optionally extracts an author and attaches `author_id`/`author_full_name` for metadata filtering.
- **`QueryExpansion` generates `expand_to_n_queries` variants** (including the original), increasing recall.
- **`ThreadPoolExecutor`** runs each variant's search concurrently, then `flatten` + `set` de-duplicate by identity (`__hash__` is by `id`).
- **Reranking only happens when candidates exist**, and `keep_top_k=k` returns exactly the final context size.

### The per-variant search

```python
    def _search(self, query: Query, k: int = 3) -> list[EmbeddedChunk]:
        assert k >= 3, "k should be >= 3"

        def _search_data_category(data_category_odm: type[EmbeddedChunk], embedded_query: EmbeddedQuery):
            if embedded_query.author_id:
                query_filter = Filter(
                    must=[FieldCondition(key="author_id", match=MatchValue(value=str(embedded_query.author_id)))]
                )
            else:
                query_filter = None

            return data_category_odm.search(
                query_vector=embedded_query.embedding,
                limit=k // 3,
                query_filter=query_filter,
            )

        embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

        post_chunks = _search_data_category(EmbeddedPostChunk, embedded_query)
        articles_chunks = _search_data_category(EmbeddedArticleChunk, embedded_query)
        repositories_chunks = _search_data_category(EmbeddedRepositoryChunk, embedded_query)

        return post_chunks + articles_chunks + repositories_chunks
```

**Key Concepts**:
- **Queries are embedded on the fly** with the same `EmbeddingDispatcher` used for indexing, producing an `EmbeddedQuery` (never persisted).
- **`k // 3` per collection** distributes the candidate budget evenly across posts, articles, and repositories, so no single source dominates.
- **Metadata filtering** builds a Qdrant `Filter` on `author_id` when SelfQuery found a user, restricting results to that person's content. This is the self-query feature in action.

---

### 2. `EmbeddedChunk.to_context` - Context Rendering

```python
# llm_engineering/domain/embedded_chunks.py
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
```

**Key Concepts**:
- Each chunk is rendered with its **type, platform, author, and content**. Grouping content blocks like this makes it easy for the model to attribute a fact to a source.
- The **1-indexed numbering** appears in traces, so you can see which chunk informed the answer.
- `chunk.__class__.__name__` (for example `EmbeddedArticleChunk`) tells the model whether the source was a post, article, or repository.

---

### 3. `InferenceExecutor` - The Generation Prompt

```python
# llm_engineering/model/inference/run.py
class InferenceExecutor:
    def __init__(self, llm, query, context=None, prompt=None) -> None:
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
- **"Use the provided context as the primary source of information"** instructs grounding. This is what makes the output a RAG result rather than a free-form generation.
- **The exact `{query}` / `{context}` placeholders** are resolved with `str.format`. Curly braces in retrieved content would break `format`; the current pipeline does not sanitize them, which is a known fragility to watch.
- **Generation parameters**: `max_new_tokens` (150 by default) bounds cost and latency; `repetition_penalty=1.1` reduces loops; `temperature=0.01` keeps it faithful.

---

### 4. The Full RAG Function

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
- **The flow is three lines**: retrieve, render, generate. All the complexity is hidden behind the retriever and executor.
- **`k=3`** is the final context size after reranking. Larger `k` improves recall but costs tokens and can add noise.
- **`opik_context.update_current_trace`** attaches the model ids, temperature, and **token counts for query, context, and answer**. This is the observability payload you inspect in Session 7.2.
- **`misc.compute_num_tokens`** tokenizes with the `HF_MODEL_ID` tokenizer (`AutoTokenizer.from_pretrained`), so the counts reflect the generator's actual tokenizer.

---

### 5. `tools/rag.py` - Retrieval-Only Harness

```python
# tools/rag.py
from langchain.globals import set_verbose
from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.infrastructure.opik_utils import configure_opik

if __name__ == "__main__":
    configure_opik()
    set_verbose(True)

    query = """
        My name is Paul Iusztin.

        Could you draft a LinkedIn post discussing RAG systems?
        I'm particularly interested in:
            - how RAG works
            - how it is integrated with vector DBs and large language models (LLMs).
        """

    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=9)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document}")
```

**Key Concepts**:
- **No LLM call**: this harness exercises only retrieval, which is ideal for debugging retrieval quality without incurring generation cost.
- **`set_verbose(True)`** makes LangChain log its chains, useful while debugging prompt construction.
- **`k=9`** retrieves a wider candidate set for manual inspection.

---

## 🔬 Deep Dive: Token Budget Management

```
query tokens  ─┐
context tokens ─┼──► must fit inside  MAX_INPUT_LENGTH  (settings)
prompt tokens  ─┘                      + MAX_NEW_TOKENS_INFERENCE
```

| Setting | Default | Role |
|---------|---------|------|
| `MAX_INPUT_LENGTH` | 2048 | Max prompt length the endpoint accepts |
| `MAX_TOTAL_TOKENS` | 4096 | Input + output ceiling |
| `MAX_NEW_TOKENS_INFERENCE` | 150 | Generated tokens per answer |

If `k` is large and chunks are long, the context can exceed `MAX_INPUT_LENGTH`, and the endpoint truncates from the front (or errors). The Opik trace's `context_tokens` is the number to watch. Raise `MAX_INPUT_LENGTH` or reduce `k` when you approach the limit.

---

## 🛠️ Hands-On: Trace One Query End to End

### Step 1: Retrieval only

```bash
python -m tools.rag
```

Check the top-ranked documents. Are they on topic and (if you named an author) filtered to that author?

### Step 2: Full RAG via the API

```bash
python -m tools.ml_service
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" \
  -d "{\"query\": \"My name is Paul Iusztin. Draft a LinkedIn post about RAG.\"}"
```

### Step 3: Inspect the Opik trace

Open the Opik project (`COMET_PROJECT`) and find the `rag` trace. Verify:
- A `SelfQuery.generate` span with the extracted author.
- A `QueryExpansion.generate` span with the variants.
- A `ContextRetriever.search` span with the candidate count.
- A `Reranker.generate` span.
- The `rag` trace metadata with `query_tokens`, `context_tokens`, `answer_tokens`.

### Step 4: Compare `k` values

Run with `k=1`, `k=3`, and `k=9` and compare answer quality and `context_tokens`.

---

## 📝 Exercise: Add Source Citations

### Task

Return the source categories alongside the answer.

1. Change `rag()` to also return the list of distinct `platform` values from the retrieved chunks.
2. Extend `QueryResponse` with `sources: list[str]`.
3. Verify the API returns citations.

```python
def rag(query: str) -> tuple[str, list[str]]:
    ...
    sources = sorted({doc.platform for doc in documents})
    return answer, sources
```

**Goal**: Turn retrieval provenance into a user-visible feature. This is the first step toward trustworthy, auditable RAG.

---

## 🐛 Common Pitfalls

- **Empty retrieval**: if collections are empty or the author filter matches nothing, `to_context([])` returns `""` and the model answers from parametric memory. Check that feature engineering ran and the author name is correct.
- **`str.format` on braces**: retrieved content containing `{` or `}` will raise in `prompt.format(...)`. Escape retrieval text or switch to a safer template.
- **Context overflow**: with large `k`, watch `context_tokens` against `MAX_INPUT_LENGTH`.
- **Author filter too strict**: SelfQuery filters on `author_id`; a wrong extraction yields zero results. The SelfQuery prompt returns `none` when no name is present, which correctly disables filtering.
- **Reranker cold-start**: the first call downloads the cross-encoder, adding latency to the first request.

---

## 🎓 Knowledge Check

1. **What three stages make up `ContextRetriever.search`?**
   - Answer: self-query metadata extraction, query expansion, and parallel multi-collection search followed by reranking.

2. **How many candidates does each collection return per query variant?**
   - Answer: `k // 3`.

3. **How are duplicates removed across variants?**
   - Answer: By converting to a `set`, since documents hash by `id`.

4. **What does `to_context` include per chunk?**
   - Answer: index, type, platform, author, and content.

5. **Which prompt makes the generation grounded?**
   - Answer: The content-creator prompt instructing the model to use the context as the primary source.

6. **Which metadata does the `rag` trace record?**
   - Answer: model id, embedding model id, temperature, and query/context/answer token counts.

---

## 🔗 Next Session

**Session 7.1**: Experiment Tracking with Comet ML

We connect training metrics and RAG traces into dashboards.

---

## 📚 Additional Resources

- [Qdrant Filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [Opik Tracing](https://www.comet.com/docs/opik/tracing)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 4.1, 4.2, 6.1

**Outcome**: You can trace and debug a full RAG query, and reason about the token budget and retrieval quality.
