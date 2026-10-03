# LLM Engineer's Handbook - Complete Learning Curriculum

## 📚 Curriculum Overview

This curriculum will teach you how to build a production-ready LLM system from scratch, following the **LLM Engineer's Handbook** by Paul Iusztin and Maxime Labonne.

### 🎯 Learning Outcomes

By completing this curriculum, you will:
- ✅ Master Domain-Driven Design (DDD) for ML systems
- ✅ Build end-to-end RAG (Retrieval-Augmented Generation) pipelines
- ✅ Fine-tune LLMs with LoRA, SFT, and DPO
- ✅ Deploy models to AWS SageMaker
- ✅ Orchestrate ML pipelines with ZenML
- ✅ Monitor prompts and track experiments
- ✅ Implement production-ready APIs with FastAPI

---

## 📖 Session-by-Session Learning Path

### **Module 1: Foundations & Architecture (Week 1-2)**

#### Session 1.1: Project Overview & DDD Principles
- **Goal**: Understand the layered architecture
- **Files to Study**: 
  - `llm_engineering/__init__.py`
  - `llm_engineering/settings.py`
  - `llm_engineering/domain/types.py`
- **Key Concept**: Dependency flow: `infrastructure → model → application → domain`
- **Hands-On**: Set up environment, configure `.env` file

#### Session 1.2: Domain Layer - Data Modeling
- **Goal**: Master Pydantic and database design
- **Files to Study**:
  - `llm_engineering/domain/base/nosql.py` - MongoDB CRUD
  - `llm_engineering/domain/base/vector.py` - Qdrant vector search
  - `llm_engineering/domain/documents.py` - Document models
  - `llm_engineering/domain/chunks.py` - Chunk models
- **Key Concept**: Generic types, singleton pattern, strategy pattern
- **Hands-On**: Create custom document models

#### Session 1.3: Infrastructure Layer - Database Connections
- **Goal**: Understand database connection management
- **Files to Study**:
  - `llm_engineering/infrastructure/db/mongo.py`
  - `llm_engineering/infrastructure/db/qdrant.py`
- **Key Concept**: Singleton pattern for connections
- **Hands-On**: Test MongoDB and Qdrant connections

---

### **Module 2: Data Engineering Pipelines (Week 3-5)**

#### Session 2.1: Web Crawling with Selenium
- **Goal**: Build scalable web crawlers
- **Files to Study**:
  - `llm_engineering/application/crawlers/base.py`
  - `llm_engineering/application/crawlers/dispatcher.py`
  - `llm_engineering/application/crawlers/github.py`
  - `llm_engineering/application/crawlers/medium.py`
- **Key Concept**: Crawler dispatcher pattern, Selenium automation
- **Hands-On**: Add custom website crawler

#### Session 2.2: Text Preprocessing Pipeline
- **Goal**: Implement text cleaning and chunking
- **Files to Study**:
  - `llm_engineering/application/preprocessing/dispatchers.py`
  - `llm_engineering/application/preprocessing/operations/cleaning.py`
  - `llm_engineering/application/preprocessing/operations/chunking.py`
- **Key Concept**: Handler pattern, LangChain text splitters
- **Hands-On**: Create custom cleaning rules

#### Session 2.3: Feature Engineering Pipeline
- **Goal**: Generate embeddings and load to vector DB
- **Files to Study**:
  - `llm_engineering/application/preprocessing/embedding_data_handlers.py`
  - `llm_engineering/application/networks/embeddings.py`
  - `pipelines/feature_engineering.py`
- **Key Concept**: Sentence transformers, vector indexing
- **Hands-On**: Experiment with different embedding models

---

### **Module 3: Dataset Generation (Week 6-7)**

#### Session 3.1: Instruction Dataset Creation
- **Goal**: Generate instruction-answer pairs with LLMs
- **Files to Study**:
  - `llm_engineering/application/dataset/generation.py`
  - `llm_engineering/application/dataset/output_parsers.py`
  - `pipelines/generate_datasets.py`
- **Key Concept**: Prompt engineering, structured output parsing
- **Hands-On**: Create custom prompt templates

#### Session 3.2: Preference Dataset for DPO
- **Goal**: Generate chosen/rejected pairs for alignment
- **Files to Study**:
  - `llm_engineering/application/dataset/generation.py` (PreferenceDatasetGenerator)
  - `llm_engineering/domain/dataset.py`
- **Key Concept**: Direct Preference Optimization theory
- **Hands-On**: Implement quality filters

---

### **Module 4: Advanced RAG System (Week 8-9)**

#### Session 4.1: RAG Architecture Deep Dive
- **Goal**: Implement multi-stage retrieval
- **Files to Study**:
  - `llm_engineering/application/rag/retriever.py`
  - `llm_engineering/application/rag/query_expanison.py`
  - `llm_engineering/application/rag/reranking.py`
  - `llm_engineering/application/rag/self_query.py`
- **Key Concept**: Query expansion, reranking, metadata filtering
- **Hands-On**: Implement HyDE (Hypothetical Document Embeddings)

#### Session 4.2: Embedding Models & Cross-Encoders
- **Goal**: Master embedding generation and reranking
- **Files to Study**:
  - `llm_engineering/application/networks/embeddings.py`
  - `llm_engineering/application/networks/base.py`
- **Key Concept**: Cosine similarity, cross-encoder scoring
- **Hands-On**: Compare embedding models

---

### **Module 5: LLM Training & Fine-Tuning (Week 10-12)**

#### Session 5.1: Supervised Fine-Tuning (SFT)
- **Goal**: Fine-tune Llama with LoRA
- **Files to Study**:
  - `llm_engineering/model/finetuning/finetune.py`
  - `llm_engineering/model/finetuning/sagemaker.py`
- **Key Concept**: LoRA, Unsloth optimization, packing
- **Hands-On**: Train on custom dataset

#### Session 5.2: Direct Preference Optimization (DPO)
- **Goal**: Align model with human preferences
- **Files to Study**:
  - `llm_engineering/model/finetuning/finetune.py` (DPO section)
- **Key Concept**: DPO loss, KL penalty
- **Hands-On**: Compare SFT vs DPO outputs

#### Session 5.3: AWS SageMaker Deployment
- **Goal**: Deploy models to production
- **Files to Study**:
  - `llm_engineering/infrastructure/aws/deploy/huggingface/run.py`
  - `llm_engineering/infrastructure/aws/deploy/huggingface/sagemaker_huggingface.py`
- **Key Concept**: SageMaker endpoints, IAM roles
- **Hands-On**: Deploy custom model

---

### **Module 6: Inference & APIs (Week 13-14)**

#### Session 6.1: FastAPI REST API
- **Goal**: Build production inference API
- **Files to Study**:
  - `llm_engineering/infrastructure/inference_pipeline_api.py`
  - `llm_engineering/model/inference/inference.py`
- **Key Concept**: Async endpoints, request validation
- **Hands-On**: Add streaming responses

#### Session 6.2: RAG Inference Flow
- **Goal**: Integrate retrieval with generation
- **Files to Study**:
  - `tools/rag.py`
  - `llm_engineering/infrastructure/opik_utils.py`
- **Key Concept**: End-to-end RAG pipeline, tracing
- **Hands-On**: Implement citation tracking

---

### **Module 7: Monitoring & Evaluation (Week 15-16)**

#### Session 7.1: Experiment Tracking with Comet ML
- **Goal**: Track and compare experiments
- **Files to Study**: Training scripts with `report_to="comet_ml"`
- **Key Concept**: Metrics logging, dashboard creation
- **Hands-On**: Set up Comet ML dashboard

#### Session 7.2: Prompt Monitoring with Opik
- **Goal**: Monitor LLM prompts and responses
- **Files to Study**:
  - `llm_engineering/infrastructure/opik_utils.py`
- **Key Concept**: Trace metadata, prompt analysis
- **Hands-On**: Create custom traces

#### Session 7.3: Model Evaluation
- **Goal**: Evaluate LLM outputs automatically
- **Files to Study**:
  - `llm_engineering/model/evaluation/evaluate.py`
- **Key Concept**: LLM-as-a-judge, pairwise comparison
- **Hands-On**: Create evaluation metrics

---

### **Module 8: Production Deployment (Week 17-18)**

#### Session 8.1: Docker & Local Infrastructure
- **Goal**: Containerize applications
- **Files to Study**:
  - `Dockerfile`
  - `docker-compose.yml`
- **Key Concept**: Multi-stage builds, service orchestration
- **Hands-On**: Build custom Docker image

#### Session 8.2: CI/CD with GitHub Actions
- **Goal**: Automate testing and deployment
- **Files to Study**:
  - `.github/workflows/ci.yaml`
  - `.github/workflows/cd.yaml`
- **Key Concept**: Pipeline automation, deployment gates
- **Hands-On**: Add custom workflows

#### Session 8.3: ZenML Orchestration
- **Goal**: Orchestrate ML pipelines
- **Files to Study**:
  - `pipelines/*.py` (all pipeline files)
  - `steps/*.py` (all step files)
- **Key Concept**: DAG execution, artifact passing
- **Hands-On**: Create custom pipeline

---

### **Module 9: Advanced Topics (Week 19-20)**

#### Session 9.1: Data Warehouse Operations
- **Goal**: Backup and restore data
- **Files to Study**:
  - `tools/data_warehouse.py`
- **Key Concept**: Data export/import, versioning
- **Hands-On**: Implement incremental backups

#### Session 9.2: Performance Optimization
- **Goal**: Optimize inference latency
- **Key Concepts**: Batching, caching, quantization
- **Hands-On**: Profile and optimize code

#### Session 9.3: Security Best Practices
- **Goal**: Secure API endpoints
- **Key Concepts**: Authentication, rate limiting, input validation
- **Hands-On**: Add JWT authentication

---

## 🎯 Capstone Projects

### Project 1: Build Your LLM Twin
- Collect your writing samples
- Generate custom instruction dataset
- Fine-tune Llama 3.1 8B
- Deploy RAG system

### Project 2: Production RAG System
- Implement advanced RAG features
- Add monitoring and evaluation
- Deploy to AWS with auto-scaling

### Project 3: Multi-Tenant LLM Platform
- Support multiple users
- Implement billing and quotas
- Create admin dashboard

---

## 📚 Quick Reference

### Key Commands

```bash
# Activate environment
.venv\Scripts\activate

# Start ZenML server
zenml up --port 8237

# Run ETL pipeline
python -m tools.run --run-etl --no-cache

# Run feature engineering
python -m tools.run --run-feature-engineering --no-cache

# Start inference API
python -m tools.ml_service

# Test RAG
python -m tools.rag
```

### Access Dashboards

- **ZenML**: http://localhost:8237 (username: `default`, password: empty)
- **Qdrant**: http://localhost:6333/dashboard
- **MongoDB**: Use MongoDB Compass with `mongodb://llm_engineering:llm_engineering@127.0.0.1:27017`

---

## 🔧 Tools & Technologies

| Category | Tools |
|----------|-------|
| **Core** | Python 3.11, Pydantic, FastAPI |
| **ML/LLM** | Transformers, Unsloth, Sentence Transformers, LangChain |
| **Databases** | MongoDB, Qdrant |
| **Cloud** | AWS SageMaker, Docker, GitHub Actions |
| **Monitoring** | Comet ML, Opik |
| **Orchestration** | ZenML |

---

## 📈 Progress Tracking

Use this checklist to track your progress:

- [ ] Module 1: Foundations complete
- [ ] Module 2: Data Engineering complete
- [ ] Module 3: Dataset Generation complete
- [ ] Module 4: RAG System complete
- [ ] Module 5: LLM Training complete
- [ ] Module 6: Inference & APIs complete
- [ ] Module 7: Monitoring & Evaluation complete
- [ ] Module 8: Production Deployment complete
- [ ] Module 9: Advanced Topics complete
- [ ] Capstone Project complete

---

**Next Steps**: Start with **Session 1.1** and work through each session sequentially. Each session builds on previous knowledge.

For detailed code examples and explanations, refer to the specific files mentioned in each session.
