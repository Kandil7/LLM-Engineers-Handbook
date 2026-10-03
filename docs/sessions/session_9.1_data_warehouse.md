# Session 9.1: Data Warehouse Operations

## 🎯 Learning Objectives

By the end of this session, you will:
- Back up the MongoDB warehouse to JSON and restore it
- Use `NoSQLBaseDocument.to_mongo` / `from_mongo` for a lossless round trip
- Export ZenML artifacts to JSON
- Reason about incremental backups and versioning
- Understand the difference between raw data and derived artifacts

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Two backup surfaces                            │
│                                                                        │
│  1. Raw data warehouse (MongoDB)                                       │
│     tools.data_warehouse --export-raw-data                             │
│        ArticleDocument / PostDocument / RepositoryDocument / UserDoc   │
│        → data/data_warehouse_raw_data/*.json                           │
│     tools.data_warehouse --import-raw-data                             │
│        *.json → MongoDB (bulk_insert)                                  │
│                                                                        │
│  2. Derived artifacts (ZenML)                                          │
│     export_artifact_to_json pipeline                                   │
│        serialize_artifact → to_json → output/*.json                    │
└──────────────────────────────────────────────────────────────────────┘
```

Raw data is the source of truth; feature stores and datasets are derived and can be rebuilt.

---

## 📁 Key Files Explained

### 1. `tools/data_warehouse.py` - Export and Import

```python
# tools/data_warehouse.py
@click.command()
@click.option("--export-raw-data", is_flag=True, default=False, help="Whether to export your data warehouse to a JSON file.")
@click.option("--import-raw-data", is_flag=True, default=False, help="Whether to import a JSON file into your data warehouse.")
@click.option("--data-dir", default=Path("data/data_warehouse_raw_data"), type=Path,
              help="Path to the directory containing data warehouse raw data JSON files.")
def main(export_raw_data, import_raw_data, data_dir: Path) -> None:
    assert export_raw_data or import_raw_data, "Specify at least one operation."

    if export_raw_data:
        __export(data_dir)
    if import_raw_data:
        __import(data_dir)
```

**Key Concepts**:
- **One CLI, two flags**. `--export-raw-data` and `--import-raw-data` can be combined.
- **`--data-dir` defaults to `data/data_warehouse_raw_data`**, the directory already present in the repo.
- **The `assert`** guards against running with no operation.

### Export

```python
def __export(data_dir: Path) -> None:
    logger.info(f"Exporting data warehouse to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    __export_data_category(data_dir, ArticleDocument)
    __export_data_category(data_dir, PostDocument)
    __export_data_category(data_dir, RepositoryDocument)
    __export_data_category(data_dir, UserDocument)


def __export_data_category(data_dir: Path, category_class: type[NoSQLBaseDocument]) -> None:
    data = category_class.bulk_find()
    serialized_data = [d.to_mongo() for d in data]
    export_file = data_dir / f"{category_class.__name__}.json"

    logger.info(f"Exporting {len(serialized_data)} items of {category_class.__name__} to {export_file}...")
    with export_file.open("w") as f:
        json.dump(serialized_data, f)
```

**Key Concepts**:
- **`bulk_find()` with no filters returns the whole collection.**
- **`to_mongo()` converts each document** so `_id` is restored as the Mongo key and UUIDs become strings. Using `to_mongo` (not `model_dump`) is what makes the export re-importable.
- **One file per document class**, named after the class (`ArticleDocument.json`, etc.), which the importer uses to recover the type.

### Import

```python
def __import_data_category(file: Path, category_class: type[NoSQLBaseDocument]) -> None:
    with file.open("r") as f:
        data = json.load(f)

    logger.info(f"Importing {len(data)} items of {category_class.__name__} from {file}...")
    if len(data) > 0:
        deserialized_data = [category_class.from_mongo(d) for d in data]
        category_class.bulk_insert(deserialized_data)
```

**Key Concepts**:
- **The filename stem maps to a class** via the `data_category_classes` dict. Unknown files are skipped with a warning.
- **`from_mongo`** reverses the export: `_id` → `id` (UUID), strings → typed fields, with Pydantic validation.
- **`bulk_insert`** writes in one batch; empty files are no-ops.
- **Idempotency caveat**: `bulk_insert` uses `insert_many`, so importing the same file twice **duplicates** documents (same `_id` would raise a duplicate-key error, but new UUIDs would not). For a true restore, drop the target collections first or import into a fresh database.

---

### 2. `infrastructure/files_io.py` - Safe JSON I/O

```python
class JsonFileManager:
    @classmethod
    def read(cls, filename: str | Path) -> list:
        file_path = Path(filename)
        try:
            with file_path.open("r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path=}' does not exist.") from None
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(msg=f"File '{file_path=}' is not properly formatted as JSON.", doc=e.doc, pos=e.pos) from None

    @classmethod
    def write(cls, filename: str | Path, data: list | dict) -> Path:
        file_path = Path(filename).resolve().absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as file:
            json.dump(data, file, indent=4)
        return file_path
```

`data_warehouse.py` uses plain `json.dump`/`json.load` for consistency with the raw format, while artifact export uses `JsonFileManager` (which creates parent directories automatically).

---

### 3. `export_artifact_to_json.py` - ZenML Artifact Export

```python
# pipelines/export_artifact_to_json.py
@pipeline
def export_artifact_to_json(artifact_names: list[str], output_dir: Path = Path("output")) -> None:
    for artifact_name in artifact_names:
        artifact = Client().get_artifact_version(name_id_or_prefix=artifact_name)

        data = export_steps.serialize_artifact(artifact=artifact, artifact_name=artifact_name)

        export_steps.to_json(data=data, to_file=output_dir / f"{artifact_name}.json")
```

```python
# steps/export/serialize_artifact.py
def _serialize_artifact(arfifact):
    if isinstance(arfifact, list):
        return [_serialize_artifact(item) for item in arfifact]
    elif isinstance(arfifact, dict):
        return {key: _serialize_artifact(value) for key, value in arfifact.items()}
    if isinstance(arfifact, BaseModel):
        return arfifact.model_dump()
    else:
        return arfifact
```

**Key Concepts**:
- **`Client().get_artifact_version(...)`** fetches an artifact by name from the ZenML metadata store - the bridge from a pipeline run to a file.
- **`_serialize_artifact` recursively converts Pydantic models** (documents, datasets) to plain dicts; primitives pass through.
- **The output file is named after the artifact**, so `instruct_datasets.json` lands in `output/`.

Run it:

```bash
python -m tools.run --run-export-artifact-to-json
```

---

## 🛠️ Hands-On: Back Up and Restore

### Step 1: Export the warehouse

```bash
python -m tools.data_warehouse --export-raw-data
ls data/data_warehouse_raw_data/
# ArticleDocument.json  PostDocument.json  RepositoryDocument.json  UserDocument.json
```

### Step 2: Inspect the format

```python
import json
data = json.load(open("data/data_warehouse_raw_data/ArticleDocument.json"))
print(len(data), "articles")
print(list(data[0].keys()))   # includes "_id"
```

### Step 3: Restore

```bash
python -m tools.data_warehouse --import-raw-data
```

Verify:

```python
from llm_engineering.domain.documents import ArticleDocument
docs = ArticleDocument.bulk_find()
print(len(docs))
```

### Step 4: Export a derived artifact

```bash
python -m tools.run --run-export-artifact-to-json
ls output/
```

---

## 📝 Exercise: Incremental Backup by Timestamp

### Task

Add a filtered, incremental export.

1. Extend `__export` to accept a `since: str | None` option.
2. When `since` is set, filter with `bulk_find(id=...)` style... but `id` is not a timestamp, so instead add a `created_at` field to documents and filter on it.
3. Write exports to `data/backups/<timestamp>/`.
4. Document how you would detect which documents are new.

**Goal**: Understand that incremental backups need a monotonic marker (a timestamp or version), which the current schema lacks.

> Note: the raw documents have no `created_at` field today, so true incremental backup requires a schema change first. This is a real gap, not a coding mistake.

---

## 🐛 Common Pitfalls

- **Re-import duplicates**: `bulk_insert` does not upsert. Back up into a fresh DB or clear collections before restoring.
- **Schema drift**: exporting with an old model and importing with a new one fails validation when required fields changed. Version your exports alongside the code.
- **Large collections in memory**: `bulk_find()` loads everything at once. For big warehouses, paginate (Qdrant's `bulk_find` already supports `limit`/`offset`; Mongo's does not in this code).
- **Raw vs derived confusion**: only `data_warehouse_raw_data` is the source of truth. Qdrant collections and datasets can be rebuilt from it.

---

## 🎓 Knowledge Check

1. **Why use `to_mongo` / `from_mongo` for backup instead of `model_dump`?**
   - Answer: They handle the `_id`/`id` and UUID↔string conversions, making the export re-importable.

2. **How does the importer know which class to use for a file?**
   - Answer: By the filename stem, mapped through a class dictionary.

3. **What is the idempotency risk of importing twice?**
   - Answer: `bulk_insert` can create duplicates; it is not an upsert.

4. **What does `export_artifact_to_json` retrieve and how?**
   - Answer: ZenML artifacts by name via `Client().get_artifact_version`.

5. **Which directory holds the raw source of truth?**
   - Answer: `data/data_warehouse_raw_data/`.

6. **What is missing for incremental backups?**
   - Answer: A monotonic field such as `created_at` on documents.

---

## 🔗 Next Session

**Session 9.2**: Performance Optimization

We profile and tune batching, caching, and quantization across the pipeline.

---

## 📚 Additional Resources

- [PyMongo: Insert and Find](https://pymongo.readthedocs.io/en/stable/tutorial.html)
- [ZenML Artifacts](https://docs.zenml.io/user-guide/starter-guide/manage-artifacts)
- [MongoDB Backup Methods](https://www.mongodb.com/docs/manual/core/backups/)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 1.2, 1.3, 8.3

**Outcome**: You can back up and restore the warehouse, export artifacts, and reason about data provenance and versioning.
