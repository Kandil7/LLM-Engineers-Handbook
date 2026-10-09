# Book Map: LLM Engineer's Handbook → Sessions → Code

This document maps every chapter of **LLM Engineer's Handbook** (Packt, Iusztin and Labonne, 2024; 523 pages, 11 chapters + Appendix) to its session walkthroughs and the repository code it documents.

Use it to navigate by the book you have open, rather than by the derived session numbering.

## Chapter-to-session map

| Book chapter | Pages | Session walkthrough(s) | Primary repository code | Status |
|--------------|-------|------------------------|-------------------------|--------|
| **1. Understanding the LLM Twin Concept and Its Architecture** | 30–52 | `session_1.1_project_overview.md` | `llm_engineering/`, FTI architecture, `pipelines/` | Covered |
| **2. Tooling and Installation** | 54–82 | `session_1.3_infrastructure_layer.md`, `GETTING_STARTED.md` | `pyproject.toml`, `Dockerfile`, `docker-compose.yml` | Covered |
| **3. Data Engineering** | 84–125 | `session_2.1_web_crawling.md`, `session_1.2_domain_layer.md`, `session_1.3_infrastructure_layer.md` | `llm_engineering/application/crawlers/`, `llm_engineering/domain/documents.py`, `tools/data_warehouse.py` | Covered |
| **4. RAG Feature Pipeline** | 128–203 | `session_2.2_text_preprocessing.md`, `session_2.3_feature_engineering.md`, `session_4.2_embedding_models.md`, `session_4.3_streaming_cdc.md` | `llm_engineering/application/preprocessing/`, `llm_engineering/application/networks/`, `pipelines/feature_engineering.py`, `steps/feature_engineering/` | Covered |
| **5. Supervised Fine-Tuning** | 206–255 | `session_3.1_instruction_dataset.md`, `session_5.4_data_curation.md`, `session_5.1_sft.md` | `llm_engineering/application/dataset/generation.py`, `llm_engineering/model/finetuning/finetune.py` | Covered |
| **6. Fine-Tuning with Preference Alignment** | 258–287 | `session_3.2_preference_dataset.md`, `session_5.2_dpo.md` | `PreferenceDatasetGenerator`, `DPOTrainer` in `finetune.py` | Covered |
| **7. Evaluating LLMs** | 290–316 | `session_7.3_model_evaluation.md`, `session_7.4_rag_evaluation.md` | `llm_engineering/model/evaluation/evaluate.py` | Covered |
| **8. Inference Optimization** | 318–343 | `session_8.4_inference_optimization.md` | vLLM (evaluation), TGI DLC + bitsandbytes (deployment) | Covered |
| **9. RAG Inference Pipeline** | 346–380 | `session_4.1_advanced_rag.md`, `session_6.2_rag_inference_flow.md`, `session_4.2_embedding_models.md` | `llm_engineering/application/rag/` | Covered |
| **10. Inference Pipeline Deployment** | 384–428 | `session_5.3_sagemaker_deployment.md`, `session_6.1_fastapi_api.md`, `session_10.1_deployment_topologies.md` | `llm_engineering/infrastructure/aws/deploy/`, `llm_engineering/model/inference/`, `llm_engineering/infrastructure/inference_pipeline_api.py` | Covered |
| **11. MLOps and LLMOps** | 430–488 | `session_7.1_comet_ml.md`, `session_7.2_opik_monitoring.md`, `session_8.1_docker.md`, `session_8.2_cicd.md`, `session_8.3_zenml.md`, `session_9.1_data_warehouse.md`, `session_11.1_ct_pipeline_alerting.md` | `.github/workflows/`, `pipelines/`, `llm_engineering/infrastructure/opik_utils.py` | Covered |
| **Appendix: MLOps Principles** | 490–503 | `appendix_mlops_principles.md` | `tests/`, CI, ZenML artifacts, seeds | Covered |

## Arranged as a study path

Read the book in order; each chapter's session goes deeper.

```
Ch 1  ─► session_1.1
Ch 2  ─► GETTING_STARTED, session_1.3
Ch 3  ─► session_2.1, session_1.2, session_1.3
Ch 4  ─► session_2.2, session_2.3, session_4.2, session_4.3
Ch 5  ─► session_3.1, session_5.4, session_5.1
Ch 6  ─► session_3.2, session_5.2
Ch 7  ─► session_7.3, session_7.4
Ch 8  ─► session_8.4
Ch 9  ─► session_4.1, session_6.2
Ch 10 ─► session_5.3, session_6.1, session_10.1
Ch 11 ─► session_7.1, session_7.2, session_8.1, session_8.2, session_8.3, session_9.1, session_11.1
App.  ─► appendix_mlops_principles
```

## Sessions that are not book chapters

These are cross-cutting or supporting sessions:

| Session | Purpose |
|---------|---------|
| `session_4.2_embedding_models.md` | Bi-encoder vs cross-encoder deep dive (Ch 4 and Ch 9) |
| `session_8.1_docker.md` | Containerization (Ch 11 infra) |
| `session_8.2_cicd.md` | GitHub Actions CI/CD (Ch 11) |
| `session_8.3_zenml.md` | ZenML orchestration (Ch 2 and Ch 11) |
| `session_9.1_data_warehouse.md` | Backup and restore (Ch 11 operations) |
| `session_9.2_performance.md` | Batching, caching, quantization tuning (Ch 4 and Ch 8) |
| `session_9.3_security.md` | Secret management and API hardening (cross-cutting) |
| `session_4.3_streaming_cdc.md` | Batch vs streaming and CDC (Ch 4) |
| `session_10.1_deployment_topologies.md` | Deployment types and autoscaling (Ch 10) |

## Book terminology notes

- **ODM vs OVM**: the MongoDB mapper (`NoSQLBaseDocument`) is the **ODM** (object-document mapping); the Qdrant mapper (`VectorBaseDocument`) is the **OVM** (object-vector mapping). See `session_1.2_domain_layer.md`.
- **FTI architecture**: the feature/training/inference pipeline design from Chapter 1 underpins every pipeline in `pipelines/`.
- **Repository URL**: the book references `PacktPublishing/LLM-Engineering`; the current upstream is `PacktPublishing/LLM-Engineers-Handbook`.
- **Embedding model**: the book's prose uses `all-mpnet-base-v2` while `settings.py` defaults to `all-MiniLM-L6-v2`; both are configurable via `TEXT_EMBEDDING_MODEL_ID`.

## Coverage status

All 11 chapters and the Appendix now have dedicated session coverage. The previously missing topics that were added:

- Chapter 8 inference optimization (`session_8.4_inference_optimization.md`).
- Chapter 7 RAG evaluation, Ragas, and ARES (`session_7.4_rag_evaluation.md`).
- Chapter 10 deployment taxonomy and autoscaling (`session_10.1_deployment_topologies.md`).
- Chapter 11 CT pipeline, cloud deployment, and alerting (`session_11.1_ct_pipeline_alerting.md`).
- Chapter 5 data curation (`session_5.4_data_curation.md`).
- Chapter 4 streaming and CDC (`session_4.3_streaming_cdc.md`).
- Appendix MLOps principles (`appendix_mlops_principles.md`).
