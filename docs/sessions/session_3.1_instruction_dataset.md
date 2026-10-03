# Session 3.1: Instruction Dataset Creation

## 🎯 Learning Objectives

By the end of this session, you will:
- Generate instruction-answer pairs from cleaned documents with GPT-4o-mini
- Understand the `DatasetGenerator` abstract base and its template isolation
- Control prompt size with `tiktoken` and `OPENAI_MAX_TOKEN_WINDOW`
- Parse a list of structured objects with a custom Pydantic output parser
- Produce the exact `instruction`/`output` schema that SFT training expects

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    generate_datasets pipeline                       │
│                                                                      │
│  query_feature_store()                                              │
│      │  scroll cleaned_articles / cleaned_posts / cleaned_repos     │
│      ▼                                                               │
│  create_prompts(documents, dataset_type)                            │
│      │  extract_substrings()  → 1000..2000 char extracts            │
│      │  group_by_category → GenerateDatasetSamplesPrompt per doc     │
│      ▼                                                               │
│  generate_intruction_dataset(prompts)                               │
│      │  ChatOpenAI(gpt-4o-mini, temp=0.7, max_tokens=1200)          │
│      │  chain.batch(batch=24)  →  ListPydanticOutputParser           │
│      │  build_dataset → InstructDataset per category                 │
│      ▼                                                               │
│  InstructTrainTestSplit(train, test)                                │
│      │                                                               │
│      ▼  (optional)                                                   │
│  push_to_huggingface(dataset_id)                                    │
└────────────────────────────────────────────────────────────────────┘
```

The instruction dataset teaches the model **what** to answer. The preference dataset (Session 3.2) teaches it **which** of two answers is better.

---

## 📁 Key Files Explained

### 1. `llm_engineering/application/dataset/generation.py` - The Base Generator

**Purpose**: Template-method base class shared by instruction and preference generation.

```python
# llm_engineering/application/dataset/generation.py
class DatasetGenerator(ABC):
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)
    dataset_type: DatasetType | None = None

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
    prompt_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"

        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION
            else "instruction-answer triples"
        )
        input_variables = {"dataset_format": dataset_format}
        system_prompt = cls.system_prompt_template.format(**input_variables)

        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt,
        )
```

**Key Concepts**:
- **`tokenizer` is a class attribute** built once from the configured OpenAI model, so token counting matches the model that will consume the prompt.
- **`dataset_type`** is the only abstract discriminator: subclasses set `DatasetType.INSTRUCTION` or `DatasetType.PREFERENCE`.
- **`system_prompt_template`** uses a single `{dataset_format}` slot, so one template serves both generators. Avoid user-controlled content in system templates - there is no injection surface here because the format is fixed by the class.

---

### 2. Prompt Building with Token Truncation

```python
    @classmethod
    def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        documents = generation_utils.extract_substrings(documents)

        grouped_prompts = {}
        grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
        for category, category_documents in grouped_cleaned_documents.items():
            category_prompts = [cls.get_prompt(document) for document in category_documents]
            grouped_prompts[category] = category_prompts
        return grouped_prompts

    @classmethod
    def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
        assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

        data_category = document.get_category()

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )
        input_variables = {"extract": document.content}
        prompt = prompt_template.format(**input_variables)
        prompt_tokens = cls.tokenizer.encode(prompt)
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)

        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            data_category=data_category,
            document=document,
        )
        return prompt
```

**Key Concepts**:
- **`extract_substrings`** (from `dataset/utils.py`) reuses `chunk_document` to slice long cleaned content into 1000–2000 character extracts and returns a `model_copy()` per extract. Each extract becomes one generation prompt.
- **`template_format="jinja2"`** is required because the templates contain literal JSON braces (`{ "instruction": ... }`). Jinja2 uses `{{extract}}` for substitution, leaving `{`/`}` untouched.
- **Hard truncation at `OPENAI_MAX_TOKEN_WINDOW`** (128000 × 0.9 for `gpt-4o-mini`) guarantees the prompt fits even for extreme documents. `num_tokens` records the final count.
- **`GenerateDatasetSamplesPrompt` carries the source document** so the dataset sample can be traced back to its origin.

---

### 3. Generation Loop with Batching and a Custom Parser

```python
    @classmethod
    def generate(
        cls,
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> TrainTestSplit:
        def _to_langchain(prompt: GenerateDatasetSamplesPrompt) -> list[BaseMessage]:
            return [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

        if mock:
            llm = FakeListLLM(responses=[constants.get_mocked_response(cls.dataset_type)])
        else:
            assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"
            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=2000 if cls.dataset_type == DatasetType.PREFERENCE else 1200,
                temperature=0.7,
            )
        parser = ListPydanticOutputParser(pydantic_object=cls._get_dataset_sample_type())
        chain = llm | parser

        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]
            batches = utils.misc.batch(langchain_category_prompts, size=24)

            flattened_instruct_dataset_samples = []
            for batch in batches:
                try:
                    batched_dataset_samples = chain.batch(batch, stop=None)
                    for instruct_dataset_sample_batch in batched_dataset_samples:
                        flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)
                except OutputParserException:
                    logger.exception(f"Failed to parse the output JSON for a batch for category {category}")

            dataset = domain.dataset.build_dataset(
                dataset_type=cls.dataset_type, category=category, samples=flattened_instruct_dataset_samples
            )
            datasets[category] = dataset
            logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'.")

        processed_datasets = cls.post_process_datasets(datasets, test_size=test_size)
        return processed_datasets
```

**Key Concepts**:
- **`temperature=0.7`** produces varied instructions. A deterministic generator would yield near-duplicate questions across extracts.
- **`max_tokens` differs by task**: 1200 for pairs, 2000 for triples (three strings each).
- **`chain.batch(batch, ...)`** sends up to 24 prompts concurrently. This is the throughput dial: `batch` uses LangChain's parallel execution, so higher values increase rate-limit risk.
- **`OutputParserException` is caught per batch**. A single malformed JSON response drops only that batch; the rest of the dataset survives.
- **`mock=True` swaps in `FakeListLLM`** so the whole pipeline can be tested without an API key (used in CI).

---

### 4. `output_parsers.py` - List of Pydantic Objects

**Purpose**: Parse a JSON **array** of sample objects into Pydantic models.

```python
# llm_engineering/application/dataset/output_parsers.py
from langchain.output_parsers import PydanticOutputParser


class ListPydanticOutputParser(PydanticOutputParser):
    def _parse_obj(self, obj: dict | list):
        if isinstance(obj, list):
            return [super(ListPydanticOutputParser, self)._parse_obj(obj_) for obj_ in obj]
        else:
            return super(ListPydanticOutputParser, self)._parse_obj(obj)
```

**Key Concepts**:
- LangChain's `PydanticOutputParser` expects a **single** object. This subclass detects a top-level list and maps `_parse_obj` over its items.
- It reuses all of the base class's validation and format instructions, so no prompt-format string is duplicated.
- If any item fails validation, the whole batch raises `ParserOutputException`, which the generator catches.

---

### 5. `InstructionDatasetGenerator` - The Prompt and Pipeline Hooks

```python
class InstructionDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.INSTRUCTION

    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \

Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
...

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{{extract}}
"""

    @classmethod
    def post_process_datasets(cls, datasets, test_size: float) -> TrainTestSplit:
        return generation_utils.create_instruct_train_test_split(
            datasets, test_size=test_size, random_state=42
        )
```

**Prompt-design principles in this template**:
- **"Only use concepts from the context"** grounds the model and prevents hallucinated facts.
- **"Instructions must never explicitly mention a context / system / course / extract"** makes samples look like real user questions, not dataset artifacts.
- **"Answers must imitate the writing style of the context"** is what makes the trained model an *LLM Twin* - it copies the author's voice.
- **A worked example** anchors the output shape before the strict JSON schema.
- **`{{extract}}`** (Jinja2) is the only substitution.

**Key Concepts**:
- **`random_state=42`** makes the train/test split reproducible; reruns produce the same split.
- `post_process_datasets` is the extension hook: instruction generation only splits, preference generation also filters (Session 3.2).

---

### 6. `dataset/utils.py` - Splitting and Sample Typing

```python
def create_instruct_train_test_split(
    data: dict[DataCategory, InstructDataset], test_size=0.2, random_state=42
) -> InstructTrainTestSplit:
    train_data = {}
    test_data = {}
    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )
            train_samples = [InstructDatasetSample(**s) for s in train_samples_dicts]
            test_samples = [InstructDatasetSample(**s) for s in test_samples_dicts]
        else:
            train_samples, test_samples = [], []

        train_data[category] = InstructDataset(category=category, samples=train_samples)
        test_data[category] = InstructDataset(category=category, samples=test_samples)

    return InstructTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)
```

**Key Concepts**:
- **The split is per category**, so a small category is not starved by a large one.
- **Empty categories are tolerated** (the `else` branch), which matters when an author has only articles and no posts.
- **Round-trip through `model_dump` → sklearn → Pydantic** keeps the domain objects pure and avoids sklearn knowing about Pydantic.

---

### 7. The `generate_datasets` Pipeline

```python
# pipelines/generate_datasets.py
@pipeline
def generate_datasets(
    dataset_type: DatasetType = DatasetType.INSTRUCTION,
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
    wait_for: str | list[str] | None = None,
) -> None:
    cleaned_documents = cd_steps.query_feature_store(after=wait_for)
    prompts = cd_steps.create_prompts(documents=cleaned_documents, dataset_type=dataset_type)
    if dataset_type == DatasetType.INSTRUCTION:
        dataset = cd_steps.generate_intruction_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock)
    elif dataset_type == DatasetType.PREFERENCE:
        dataset = cd_steps.generate_preference_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if push_to_huggingface:
        cd_steps.push_to_huggingface(dataset=dataset, dataset_id=dataset_id)
```

**Key Concepts**:
- **One pipeline, two modes** selected by `dataset_type`.
- **`query_feature_store`** scrolls Qdrant `cleaned_*` collections with pagination (`bulk_find(limit=1)` + `next_offset`), fetching from articles, posts, and repositories in parallel.
- **`ArtifactConfig(name="instruct_datasets", tags=[...])`** names the ZenML artifact so later steps (training) can reference it by name.

> Note: `query_feature_store()` in the current code declares no parameters, yet the pipeline calls `query_feature_store(after=wait_for)`. If you hit a signature error locally, this is the mismatch to check; the pipeline passes `after` but the step does not declare it.

---

## 🛠️ Hands-On: Generate a Small Instruction Dataset

### Step 1: Fetch cleaned documents

```python
from llm_engineering.domain.cleaned_documents import CleanedArticleDocument

docs, offset = CleanedArticleDocument.bulk_find(limit=1)
while offset:
    more, offset = CleanedArticleDocument.bulk_find(limit=1, offset=offset)
    docs.extend(more)
print("Cleaned articles:", len(docs))
```

### Step 2: Build prompts

```python
from llm_engineering.application.dataset import generation
from llm_engineering.domain.dataset import DatasetType

gen = generation.get_dataset_generator(DatasetType.INSTRUCTION)
prompts = gen.get_prompts(docs[:3])
for category, category_prompts in prompts.items():
    print(category, len(category_prompts), "prompts")
    print(category_prompts[0].content[:400])
```

### Step 3: Generate with mock, then real

```python
split = gen.generate(prompts, test_size=0.2, mock=True)   # no API key needed
for category, ds in split.train.items():
    print(category, ds.num_samples)
    print(ds.samples[0].instruction if ds.samples else "(empty)")
```

### Step 4: Inspect the Hugging Face schema

```python
hf = split.train[list(split.train.keys())[0]].to_huggingface()
print(hf.column_names)   # ['instruction', 'output']
print(hf[0])
```

---

## 📝 Exercise: Tune the Instruction Prompt

### Task

Improve sample quality and measure the difference.

1. Add a constraint that answers must be **between 80 and 150 words**.
2. Add a constraint that instructions must **start with a verb** ("Explain...", "Describe...").
3. Generate 20 samples with the mock, then with the real LLM.
4. Manually inspect 5 samples: are they self-contained and style-matched?

**Goal**: Feel how prompt constraints change the dataset. Strong constraints improve training quality but can cause parse failures if the model over-fits to formatting.

---

## 🐛 Common Pitfalls

- **JSON drift**: the model sometimes wraps JSON in ```` ```json ```` fences. The base `PydanticOutputParser` prompt instructions and `temperature=0.7` reduce this, but the `OutputParserException` catch is what keeps the pipeline alive.
- **Cost**: each extract becomes one request, and prompts are sent 24 at a time. Estimate cost before a full run by counting `num_prompts × 5 pairs`.
- **Rate limits**: reduce the `batch(..., size=24)` value if you see 429 responses.

---

## 🎓 Knowledge Check

1. **Why is `template_format="jinja2"` required?**
   - Answer: The templates contain literal JSON braces; Jinja2 substitutes `{{extract}}` without treating `{`/`}` as fields.

2. **What does `extract_substrings` produce, and why 1000–2000 characters?**
   - Answer: Sentence-aware extracts of cleaned documents; the size keeps prompts focused and within token budget.

3. **What is the role of `ListPydanticOutputParser`?**
   - Answer: It extends the single-object parser to accept a JSON array of samples.

4. **Why `temperature=0.7` here rather than 0?**
   - Answer: To generate diverse instructions; determinism is not required for dataset synthesis.

5. **What two prompt rules make samples realistic?**
   - Answer: Instructions never mention the context/system, and answers imitate the source writing style.

6. **What schema does `to_huggingface()` emit for instruction data?**
   - Answer: `instruction` and `output` columns.

---

## 🔗 Next Session

**Session 3.2**: Preference Dataset for DPO

We generate chosen/rejected triples, filter low-quality samples, and prepare the DPO dataset.

---

## 📚 Additional Resources

- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [tiktoken](https://github.com/openai/tiktoken)
- [sklearn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 2.2, 2.3

**Outcome**: You can generate, parse, split, and publish an instruction dataset grounded in your own corpus.
