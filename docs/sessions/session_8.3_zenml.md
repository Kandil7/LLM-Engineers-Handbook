# Session 8.3: ZenML Orchestration

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how `@pipeline` and `@step` compose a DAG
- Read the stack concept and switch between local and AWS stacks
- Control runs through YAML configs and Poe tasks
- Use artifacts, caching, and step invocation IDs for chaining
- Read the full `end_to_end_data` pipeline

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ZenML stack                                    │
│                                                                        │
│  orchestrator        → where steps run (local / SageMaker)            │
│  artifact store      → where outputs live (local / S3)                │
│  container registry  → image for step containers (local / ECR)        │
│  secrets manager     → settings + credentials                         │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │  end_to_end_data (pipeline)                                        │
 │                                                                    │
 │  for each author:                                                  │
 │    digital_data_etl ──► invocation_id ──────────┐                  │
 │                                                  │ wait_for         │
 │  feature_engineering(author_full_names, wait_for) ◄┘                │
 │    query_data_warehouse → clean → load_to_vector_db                │
 │                        → chunk_and_embed → load_to_vector_db       │
 │                                    │ invocation_ids                 │
 │  generate_datasets(test_split_size, ..., wait_for) ◄───────────────┘
 └────────────────────────────────────────────────────────────────────┘
```

Pipelines are Python functions decorated with `@pipeline`; they call `@step` functions. ZenML records every run, artifact, and parameter.

---

## 📁 Key Files Explained

### 1. The Step/Pipeline Primitives

```python
# pipelines/digital_data_etl.py
from zenml import pipeline
from steps.etl import crawl_links, get_or_create_user


@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    user = get_or_create_user(user_full_name)
    last_step = crawl_links(user=user, links=links)
    return last_step.invocation_id
```

```python
# steps/feature_engineering/clean.py
from typing_extensions import Annotated
from zenml import get_step_context, step


@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    ...
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))
    return cleaned_documents
```

**Key Concepts**:
- **`@pipeline`** defines a DAG. Calling a `@step` inside it adds a node; passing one step's return value to another creates an edge.
- **`Annotated[type, "name"]`** labels step inputs/outputs. The label appears in the ZenML dashboard and names the artifact.
- **`last_step.invocation_id`** is a handle to a specific step run, used to express `wait_for` dependencies without passing data.
- **`get_step_context().add_output_metadata(...)`** attaches structured metadata to a step output for observability.

### 2. Named Artifacts

```python
# steps/generate_datasets/generate_intruction_dataset.py
@step
def generate_intruction_dataset(
    prompts,
    test_split_size,
    mock: Annotated[bool, "mock_generation"] = False,
) -> Annotated[
    InstructTrainTestSplit,
    ArtifactConfig(name="instruct_datasets", tags=["dataset", "instruct", "cleaned"]),
]:
    ...
```

**Key Concepts**:
- **`ArtifactConfig(name=...)`** gives the output a stable name (`instruct_datasets`), so `export_artifact_to_json` and training steps can reference it by name rather than by run.
- **`tags=[...]`** make artifacts searchable in the ZenML UI.

### 3. The Full Data Pipeline

```python
# pipelines/end_to_end_data.py
@pipeline
def end_to_end_data(
    author_links: list[dict[str, str | list[str]]],
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
) -> None:
    wait_for_ids = []
    for author_data in author_links:
        last_step_invocation_id = digital_data_etl(
            user_full_name=author_data["user_full_name"], links=author_data["links"]
        )
        wait_for_ids.append(last_step_invocation_id)

    author_full_names = [author_data["user_full_name"] for author_data in author_links]
    wait_for_ids = feature_engineering(author_full_names=author_full_names, wait_for=wait_for_ids)

    generate_datasets(
        test_split_size=test_split_size,
        push_to_huggingface=push_to_huggingface,
        dataset_id=dataset_id,
        mock=mock,
        wait_for=wait_for_ids,
    )
```

**Key Concepts**:
- **A pipeline calling another pipeline** nests runs: `digital_data_etl` and `feature_engineering` become child pipelines of `end_to_end_data`.
- **`wait_for`** encodes ordering without data dependencies. `digital_data_etl` returns its `crawl_links` invocation id; `feature_engineering` waits on those before querying the warehouse.
- **`feature_engineering` returns two invocation ids** (the two `load_to_vector_db` calls), which `generate_datasets` waits on, ensuring both cleaned and embedded documents are persisted before dataset generation.

### 4. Feature Engineering DAG

```python
# pipelines/feature_engineering.py
@pipeline
def feature_engineering(author_full_names: list[str], wait_for=None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)

    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    return [last_step_1.invocation_id, last_step_2.invocation_id]
```

The DAG fans out after `clean_documents`: one branch stores cleaned documents, the other chunks+embeds then stores. Both branches feed the returned ids.

### 5. Stack Configuration

Local default stack (MongoDB, Qdrant, local artifact store, local orchestrator):

```toml
# pyproject.toml
local-docker-infrastructure-up = "docker compose up -d"
local-zenml-server-down = "poetry run zenml logout --local"
local-infrastructure-up = [
    "local-docker-infrastructure-up",
    "local-zenml-server-down",
    "local-zenml-server-up",
]
set-local-stack = "poetry run zenml stack set default"
```

AWS stack (SageMaker orchestrator, S3 artifact store, ECR registry):

```toml
set-aws-stack = "poetry run zenml stack set aws-stack"
set-asynchronous-runs = "poetry run zenml orchestrator update aws-stack --synchronous=False"
export-settings-to-zenml = "poetry run python -m tools.run --export-settings"
```

**Key Concepts**:
- **`zenml stack set <name>`** switches the active stack; the same pipeline code runs locally or on SageMaker with no code change.
- **`set-asynchronous-runs`** flips the SageMaker orchestrator to async: the pipeline submits and returns, rather than blocking the CLI. This is what makes large training runs practical.
- **`export-settings-to-zenml`** pushes `Settings` into the ZenML secret store (`settings.export()` in Session 1.3), so remote steps read the same config.

---

### 6. YAML Configs

```yaml
# configs/feature_engineering.yaml
settings:
  docker:
    parent_image: 992382797823.dkr.ecr.eu-central-1.amazonaws.com/zenml-rlwlcs:latest
    skip_build: True
  orchestrator.sagemaker:
    synchronous: false

parameters:
  author_full_names:
    - Maxime Labonne
    - Paul Iusztin
```

**Key Concepts**:
- **`settings.docker.parent_image`** pins the base image for step containers (the ECR image built by CD).
- **`skip_build: True`** uses the existing image instead of rebuilding per run.
- **`settings.orchestrator.sagemaker.synchronous: false`** enables async submission for SageMaker.
- **`parameters`** are the pipeline's runtime arguments. Editing YAML changes a run without touching code.

```yaml
# configs/end_to_end_data.yaml (shape)
settings:
  docker: { parent_image: ..., skip_build: True }
  orchestrator.sagemaker: { synchronous: false }
parameters:
  author_links:
    - user_full_name: Paul Iusztin
      links: [ ...medium, ...substack... ]
    - user_full_name: Maxime Labonne
      links: [ ... ]
  test_split_size: 0.1
  push_to_huggingface: false
  dataset_id: pauliusztin/llmtwin
  mock: false
```

**Key Concepts**:
- **`author_links` is a list of dicts**, each with a `user_full_name` and `links`. This drives one ETL run per author and one combined feature-engineering run.
- **`dataset_id`** is the Hugging Face repo the dataset is pushed to (when `push_to_huggingface: true`).

---

### 7. The CLI Entry Point

```python
# tools/run.py (excerpt)
    if run_end_to_end_data:
        pipeline_args["config_path"] = root_dir / "configs" / "end_to_end_data.yaml"
        pipeline_args["run_name"] = f"end_to_end_data_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        end_to_end_data.with_options(**pipeline_args)(**run_args_end_to_end)
```

**Key Concepts**:
- **`--no-cache` maps to `enable_cache=False`** in `pipeline_args`. By default ZenML caches steps whose inputs are unchanged.
- **Config paths are resolved from the repo root**, so the CLI can run from anywhere.
- **Run names embed a timestamp**, making runs easy to find.
- **`assert` guards** ensure at least one action flag is passed.

Example commands:

```bash
python -m tools.run --run-etl --no-cache
python -m tools.run --run-feature-engineering --no-cache
python -m tools.run --run-end-to-end-data --no-cache
python -m tools.run --export-settings
```

---

## 🔬 Deep Dive: Caching and Artifacts

```
Step input unchanged?  ──yes──►  reuse cached output (skip execution)
       │no
       ▼
Run step  ──►  store output as an artifact  ──►  record in the metadata store
```

- **Caching is content-addressed**: a step reruns only if its inputs, parameters, or code change. During development, `--no-cache` forces fresh execution.
- **Artifacts are typed**: `list`, `dict`, `TrainTestSplit`, etc. ZenML serializes them to the artifact store (local or S3).
- **`Client().get_artifact_version(name_id_or_prefix=...)`** retrieves an artifact by name in `export_artifact_to_json`, decoupling export from the producing run.

**Why artifacts matter**: training and evaluation read datasets by name from the artifact store, so a rerun of dataset generation automatically flows into training without passing files by hand.

---

## 🛠️ Hands-On: Run a Local Pipeline

### Step 1: Bring up the local stack

```bash
poetry poe local-docker-infrastructure-up
poetry poe local-zenml-server-up
poetry poe set-local-stack
```

### Step 2: Export settings and run ETL

```bash
poetry poe export-settings-to-zenml
python -m tools.run --run-etl --no-cache
```

### Step 3: Inspect the dashboard

Open `http://localhost:8237`. Click the run and inspect:
- The DAG of steps.
- Each step's inputs/outputs and attached metadata (for example `num_documents`, `authors`).
- Artifact versions.

### Step 4: Run the end-to-end data pipeline

```bash
python -m tools.run --run-end-to-end-data --no-cache
```

Observe the nested child pipelines and the `wait_for` ordering.

### Step 5: Export an artifact

```bash
python -m tools.run --run-export-artifact-to-json
# writes output/<artifact_name>.json
```

---

## 📝 Exercise: Create a Custom Pipeline

### Task

Add a pipeline that computes simple corpus statistics.

1. Add a step `steps/stats/count_documents.py`:

```python
from typing_extensions import Annotated
from zenml import step

@step
def count_documents(documents: Annotated[list, "documents"]) -> Annotated[dict, "stats"]:
    stats = {}
    for doc in documents:
        key = doc.get_collection_name()
        stats[key] = stats.get(key, 0) + 1
    return stats
```

2. Add a pipeline `pipelines/stats.py` that queries the warehouse, then counts.
3. Register it in `pipelines/__init__.py`.
4. Add a Poe task and run it.

**Goal**: Practice the pipeline/step/artifact pattern end to end, including export registration.

---

## 🐛 Common Pitfalls

- **Wrong stack active**: running on the `aws-stack` by accident incurs cloud cost. Check with `zenml stack set --help` / the dashboard before big runs.
- **Cache confusion**: an unexpectedly fast run usually means steps were cached. Use `--no-cache` when you need certainty.
- **`wait_for` mismatches**: passing the wrong invocation id (for example the ETL step instead of the last ETL step) changes ordering. The code returns `last_step.invocation_id` deliberately.
- **Signature drift**: `query_feature_store` is called with `after=wait_for` but the step declares no `after` parameter; this is an existing inconsistency to be aware of when running that path.
- **Secrets not exported**: remote runs fail to read settings if `export-settings-to-zenml` was never run. `Settings.load_settings()` then falls back to `.env` defaults, which may be wrong in the cloud.

---

## 🎓 Knowledge Check

1. **What is the difference between `@pipeline` and `@step`?**
   - Answer: A pipeline defines a DAG; a step is one node that runs a single unit of work.

2. **How are two pipelines chained without passing data?**
   - Answer: Via `wait_for` using step `invocation_id`s.

3. **What does `Annotated[type, "name"]` do on a step output?**
   - Answer: Names the artifact so it is identifiable in the UI and addressable by name.

4. **What does `skip_build: True` in a config mean?**
   - Answer: Use the pinned `parent_image` instead of rebuilding a container per run.

5. **What does `synchronous: false` change?**
   - Answer: The SageMaker orchestrator submits runs asynchronously instead of blocking.

6. **How does `enable_cache` behave by default?**
   - Answer: It reuses outputs for steps whose inputs/parameters/code are unchanged; `--no-cache` disables it.

---

## 🔗 Next Session

**Session 9.1**: Data Warehouse Operations

We back up and restore the MongoDB warehouse as JSON.

---

## 📚 Additional Resources

- [ZenML Pipelines and Steps](https://docs.zenml.io/user-guide/starter-guide)
- [ZenML Stacks](https://docs.zenml.io/user-guide/production-guide/understand-stacks)
- [ZenML Caching](https://docs.zenml.io/user-guide/advanced-guide/caching)
- [ZenML Artifacts](https://docs.zenml.io/user-guide/starter-guide/manage-artifacts)

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Sessions 1.3, 8.1, 8.2

**Outcome**: You can read and extend ZenML pipelines, manage stacks and configs, and reason about artifacts and caching.
