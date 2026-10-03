# Session 4.1: Advanced RAG Architecture

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand multi-stage retrieval pipelines
- Implement query expansion techniques
- Master reranking with cross-encoders
- Build metadata filtering with self-query

---

## 🏗️ RAG Architecture Overview

### The Complete RAG Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Advanced RAG Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query: "How does RAG work?"                           │
│       ↓                                                      │
│  ┌──────────────────────────────────────────────┐           │
│  │ 1. Self-Query (Extract Metadata)             │           │
│  │    → Extract: author="Paul Iusztin"          │           │
│  └──────────────────────────────────────────────┘           │
│       ↓                                                      │
│  ┌──────────────────────────────────────────────┐           │
│  │ 2. Query Expansion (Generate Variations)     │           │
│  │    → "What is Retrieval-Augmented Generation?"            │
│  │    → "Explain RAG architecture"                            │
│  │    → "How does retrieval + generation work?"              │
│  └──────────────────────────────────────────────┘           │
│       ↓                                                      │
│  ┌──────────────────────────────────────────────┐           │
│  │ 3. Parallel Vector Search (3 queries × 5 docs)            │
│  │    → 15 documents retrieved                                │
│  └──────────────────────────────────────────────┘           │
│       ↓                                                      │
│  ┌──────────────────────────────────────────────┐           │
│  │ 4. Reranking (Cross-Encoder)                 │           │
│  │    → Score each (query, doc) pair           │           │
│  │    → Keep top 3                             │           │
│  └──────────────────────────────────────────────┘           │
│       ↓                                                      │
│  Context: [Doc1, Doc2, Doc3]                                │
│       ↓                                                      │
│  LLM Generation: "RAG works by..."                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Key Components

### 1. Main Retriever

**File**: `llm_engineering/application/rag/retriever.py`

```python
# llm_engineering/application/rag/retriever.py

from concurrent.futures import ThreadPoolExecutor
from llm_engineering.application.rag.query_expanison import QueryExpansion
from llm_engineering.application.rag.reranking import Reranker
from llm_engineering.application.rag.self_query import SelfQuery
from llm_engineering.domain.embedded_chunks import EmbeddedChunk

class ContextRetriever:
    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3
    ) -> list[EmbeddedChunk]:
        """
        Multi-stage retrieval with query expansion and reranking.
        
        Args:
            query: User query
            k: Final number of documents to return
            expand_to_n_queries: Number of query variations
        
        Returns:
            Top-k most relevant document chunks
        """
        
        # Step 1: Extract metadata (author name)
        query_obj = SelfQuery().generate(query)
        
        # Step 2: Generate query variations
        n_queries = QueryExpansion().generate(
            query_obj.content,
            expand_to_n=expand_to_n_queries
        )
        
        # Step 3: Parallel search for each query
        with ThreadPoolExecutor() as executor:
            search_futures = [
                executor.submit(self._search_single_query, q, query_obj, k)
                for q in n_queries
            ]
            
            # Collect all results
            n_k_documents = []
            for future in search_futures:
                n_k_documents.extend(future.result())
        
        # Remove duplicates
        n_k_documents = list({doc.id: doc for doc in n_k_documents}.values())
        
        # Step 4: Rerank and return top-k
        if len(n_k_documents) > k:
            k_documents = Reranker().rerank(
                query=query_obj.content,
                chunks=n_k_documents,
                keep_top_k=k
            )
        else:
            k_documents = n_k_documents[:k]
        
        return k_documents
    
    def _search_single_query(self, query: str, query_obj, k: int):
        """Search for a single query variation"""
        return EmbeddedChunk.search(
            query_vector=query,
            limit=k,
            query_filter=query_obj.to_filter()  # Metadata filtering
        )
```

**Key Concepts**:
- **Multi-Stage Pipeline**: Sequential processing stages
- **Parallel Execution**: ThreadPoolExecutor for speed
- **Deduplication**: Remove duplicate documents

---

### 2. Query Expansion

**File**: `llm_engineering/application/rag/query_expanison.py`

```python
# llm_engineering/application/rag/query_expanison.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from llm_engineering.settings import settings

class QueryExpansion:
    def __init__(self):
        # Use GPT-4o-mini for query generation
        self._llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.7,  # Higher temperature for diversity
            max_tokens=100
        )
        
        # Prompt template
        self._prompt = ChatPromptTemplate.from_template("""
Generate {expand_to_n} different versions of the given user question.
Each version should ask about the same topic but in a different way.

Original question: {question}

Provide each variation on a new line, separated by '#next-question#'.""")
        
        # Create chain
        self._chain = self._prompt | self._llm
    
    def generate(self, query: str, expand_to_n: int = 3) -> list[str]:
        """Generate multiple query variations"""
        
        # Invoke LLM
        response = self._chain.invoke({
            "question": query,
            "expand_to_n": expand_to_n
        })
        
        # Parse response
        content = response.content.strip()
        variations = content.split("#next-question#")
        
        # Clean and filter
        variations = [v.strip() for v in variations if v.strip()]
        
        return variations[:expand_to_n]
```

**Example Output**:
```
Input: "How does RAG work?"

Output:
- "What is Retrieval-Augmented Generation?"
- "Explain the RAG architecture"
- "How does retrieval + generation work?"
```

**Why It Works**:
- Different phrasings match different document styles
- Increases recall (finding relevant documents)
- Compensates for poor initial query formulation

---

### 3. Reranking with Cross-Encoders

**File**: `llm_engineering/application/rag/reranking.py`

```python
# llm_engineering/application/rag/reranking.py

from sentence_transformers import CrossEncoder
from llm_engineering.application.networks.base import SingletonMeta

class CrossEncoderModelSingleton(metaclass=SingletonMeta):
    """Singleton for cross-encoder model"""
    
    def __init__(self, model_id="cross-encoder/ms-marco-MiniLM-L-4-v2"):
        # Cross-encoder scores (query, doc) pairs
        self._model = CrossEncoder(model_id, device="cpu")
    
    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score each (query, document) pair"""
        return self._model.predict(pairs)

class Reranker:
    def __init__(self):
        self._model = CrossEncoderModelSingleton()
    
    def rerank(
        self,
        query: str,
        chunks: list,
        keep_top_k: int
    ) -> list:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: User query
            chunks: List of document chunks
            keep_top_k: Number of documents to keep
        
        Returns:
            Reranked top-k documents
        """
        
        # Create (query, doc) pairs
        query_doc_tuples = [
            (query, chunk.content)
            for chunk in chunks
        ]
        
        # Score each pair
        scores = self._model(query_doc_tuples)
        
        # Sort by score (descending)
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k
        return [chunk for _, chunk in scored_chunks[:keep_top_k]]
```

**How Cross-Encoders Work**:

```
┌─────────────────────────────────────────┐
│  Cross-Encoder Architecture             │
├─────────────────────────────────────────┤
│                                          │
│  Query: "How does RAG work?"            │
│  Doc: "RAG combines retrieval..."       │
│         ↓                                │
│  [Query] [SEP] [Doc] → BERT → Score     │
│         ↓                                │
│  0.87 (high relevance)                  │
│                                          │
└─────────────────────────────────────────┘
```

**vs Bi-Encoders**:
- **Bi-Encoder** (Embedding Model): Encode query and doc separately → Fast for search
- **Cross-Encoder**: Process query and doc together → Accurate but slow

**Best Practice**: Use bi-encoder for initial retrieval, cross-encoder for reranking.

---

### 4. Self-Query (Metadata Extraction)

**File**: `llm_engineering/application/rag/self_query.py`

```python
# llm_engineering/application/rag/self_query.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm_engineering.settings import settings

class SelfQueryResult(BaseModel):
    """Structured output for metadata extraction"""
    
    author_full_name: str | None = Field(
        default=None,
        description="Full name of the author mentioned in the query"
    )
    
    content: str = Field(
        ...,
        description="The core query content without metadata"
    )
    
    def to_filter(self) -> dict | None:
        """Convert to Qdrant filter"""
        if self.author_full_name:
            return {
                "must": [
                    {
                        "key": "author_full_name",
                        "match": {
                            "value": self.author_full_name
                        }
                    }
                ]
            }
        return None

class SelfQuery:
    def __init__(self):
        self._llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,  # Deterministic extraction
            max_tokens=200
        )
        
        self._prompt = ChatPromptTemplate.from_template("""
Extract metadata from the user query.

Query: {query}

Extract:
1. Author name (if mentioned)
2. Core question (without metadata)

Return as JSON:
{{
    "author_full_name": "John Doe" or null,
    "content": "What did John write about RAG?"
}}""")
        
        # Use structured output
        self._chain = self._prompt | self._llm.with_structured_output(SelfQueryResult)
    
    def generate(self, query: str) -> SelfQueryResult:
        """Extract metadata from query"""
        return self._chain.invoke({"query": query})
```

**Example**:
```python
query = "My name is Paul Iusztin. What did I write about RAG?"
result = SelfQuery().generate(query)

print(result.author_full_name)  # "Paul Iusztin"
print(result.content)  # "What did I write about RAG?"

filter = result.to_filter()
# {"must": [{"key": "author_full_name", "match": {"value": "Paul Iusztin"}}]}
```

**Usage in Search**:
```python
# Filter vector search to only Paul's documents
EmbeddedChunk.search(
    query_vector=...,
    query_filter=filter,  # Metadata filtering
    limit=10
)
```

---

## 🔬 Deep Dive: How Reranking Works

### Bi-Encoder vs Cross-Encoder

```python
# Bi-Encoder (Fast, for retrieval)
query_embedding = model.encode(query)
doc_embedding = model.encode(document)
similarity = cosine(query_embedding, doc_embedding)

# Cross-Encoder (Accurate, for reranking)
score = model.predict([(query, document)])
```

### Performance Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Bi-Encoder | Fast (1000s/sec) | Good | Initial retrieval |
| Cross-Encoder | Slow (10s/sec) | Excellent | Reranking top-100 |

### Why Reranking Matters

```
Initial Retrieval (Bi-Encoder):
- Retrieves 15 documents based on embedding similarity
- May miss semantic nuances
- Fast enough for large databases

Reranking (Cross-Encoder):
- Scores each (query, doc) pair jointly
- Captures semantic relationships
- Selects truly relevant documents
```

---

## 🛠️ Hands-On: Build Your RAG Pipeline

### Step 1: Test Query Expansion

```python
from llm_engineering.application.rag.query_expanison import QueryExpansion

expander = QueryExpansion()
variations = expander.generate(
    "How does RAG work?",
    expand_to_n=3
)

for v in variations:
    print(f"- {v}")
```

**Expected Output**:
```
- What is Retrieval-Augmented Generation?
- Explain the RAG architecture
- How does retrieval + generation work?
```

---

### Step 2: Test Self-Query

```python
from llm_engineering.application.rag.self_query import SelfQuery

query = "My name is Paul Iusztin. Write about RAG."
result = SelfQuery().generate(query)

print(f"Author: {result.author_full_name}")
print(f"Content: {result.content}")
print(f"Filter: {result.to_filter()}")
```

---

### Step 3: Test Reranking

```python
from llm_engineering.application.rag.reranking import Reranker
from llm_engineering.domain.embedded_chunks import EmbeddedChunk

# Create mock chunks
chunks = [
    EmbeddedChunk(content="RAG combines retrieval with generation"),
    EmbeddedChunk(content="The weather is nice today"),
    EmbeddedChunk(content="Retrieval-augmented generation improves LLMs"),
]

reranker = Reranker()
reranked = reranker.rerank(
    query="How does RAG work?",
    chunks=chunks,
    keep_top_k=2
)

for chunk in reranked:
    print(chunk.content)
```

**Expected Output**:
```
RAG combines retrieval with generation
Retrieval-augmented generation improves LLMs
```

---

### Step 4: Full RAG Search

```python
from llm_engineering.application.rag.retriever import ContextRetriever

retriever = ContextRetriever()
documents = retriever.search(
    query="My name is Paul Iusztin. How does RAG work?",
    k=3,
    expand_to_n_queries=3
)

print(f"Retrieved {len(documents)} documents:")
for doc in documents:
    print(f"- {doc.content[:100]}...")
```

---

## 📊 Performance Optimization

### Parallel Search

```python
# Sequential (Slow)
results = []
for query in queries:
    results.extend(search(query))

# Parallel (Fast)
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(search, q) for q in queries]
    results = [f.result() for f in futures]
```

**Speedup**: 3x faster for 3 queries

---

### Batch Reranking

```python
# Individual (Slow)
scores = [model.predict([(query, doc)]) for doc in docs]

# Batch (Fast)
pairs = [(query, doc) for doc in docs]
scores = model.predict(pairs)
```

**Speedup**: 10x faster for 100 documents

---

## 🎯 Advanced Techniques

### 1. **HyDE (Hypothetical Document Embeddings)**

```python
class HyDE:
    def __init__(self):
        self.llm = ChatOpenAI(...)
    
    def generate_hypothetical_doc(self, query: str) -> str:
        """Generate a hypothetical answer"""
        prompt = f"Write a detailed answer for: {query}"
        return self.llm.invoke(prompt).content
    
    def search(self, query: str):
        # Generate hypothetical document
        hypo_doc = self.generate_hypothetical_doc(query)
        
        # Embed hypothetical document
        hypo_embedding = embed(hypo_doc)
        
        # Search with hypothetical embedding
        return search_vector(hypo_embedding)
```

**Why**: Query embeddings ≠ Document embeddings. Hypothetical docs bridge the gap.

---

### 2. **Semantic Caching**

```python
from cachetools import TTLCache, cached

class CachedRetriever(ContextRetriever):
    @cached(cache=TTLCache(maxsize=1000, ttl=3600))
    def search(self, query: str, k: int = 3):
        return super().search(query, k)
```

**Benefit**: Cache frequent queries (1 hour TTL)

---

### 3. **Query Rewriting**

```python
class QueryRewriter:
    def rewrite(self, query: str) -> str:
        """Rewrite query for better retrieval"""
        # Fix typos
        # Expand abbreviations
        # Add context
        return rewritten_query
```

---

## 📝 Exercise: Implement HyDE

### Task

Implement Hypothetical Document Embeddings for better retrieval.

### Template

```python
from llm_engineering.application.rag.retriever import ContextRetriever

class HyDERetriever(ContextRetriever):
    def search(self, query: str, k: int = 3):
        # Step 1: Generate hypothetical document
        hypo_doc = self._generate_hypothetical_doc(query)
        
        # Step 2: Embed hypothetical document
        hypo_embedding = embed(hypo_doc)
        
        # Step 3: Search with hypothetical embedding
        return search_vector(hypo_embedding, limit=k)
    
    def _generate_hypothetical_doc(self, query: str) -> str:
        # Use LLM to generate answer
        pass
```

---

## 🎓 Knowledge Check

1. **Why use query expansion?**
   - Answer: Increases recall by matching different phrasings

2. **What's the difference between bi-encoder and cross-encoder?**
   - Answer: Bi-encoder encodes separately (fast), cross-encoder processes together (accururate)

3. **Why rerank after retrieval?**
   - Answer: Cross-encoder captures semantic nuances bi-encoder misses

4. **What does self-query do?**
   - Answer: Extracts metadata for filtering

---

## 🔗 Next Session

**Session 4.2**: Embedding Models & Cross-Encoders

We'll dive deep into:
- Sentence Transformer architecture
- Embedding model comparison
- Cross-encoder training
- Performance optimization

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 1.1-1.3, 2.1-2.3

**Outcome**: You'll understand and implement advanced RAG techniques.
