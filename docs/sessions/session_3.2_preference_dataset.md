# Session 3.2: Preference Dataset for DPO

## 🎯 Learning Objectives

By the end of this session, you will:
- Explain why Direct Preference Optimization (DPO) needs chosen/rejected pairs
- Generate instruction-rejected-chosen triples from your corpus
- Apply the two quality filters that make preference data usable
- Produce the exact `prompt`/`chosen`/`rejected` schema that DPO training expects

---

## 🏗️ Architecture Overview

### Instruction vs Preference Data

```
Instruction dataset (Session 3.1)          Preference dataset (this session)
┌──────────────────────────────┐           ┌──────────────────────────────┐
│ instruction                  │           │ instruction                  │
│ answer                       │           │ chosen     (good, from text) │
└──────────────────────────────┘           │ rejected   (model-generated) │
   "Here is the right answer"              └──────────────────────────────┘
                                                    ▲             ▲
                                            verbatim from   plausible but
                                            the source      imperfect
```

Instruction tuning teaches the model **to answer**. Preference tuning teaches it **which answer is better** by contrasting a high-quality reference against a plausible but weaker generation.

### Generation Flow

```
query_feature_store() ──► create_prompts(dataset_type=PREFERENCE)
        │
        ▼
PreferenceDatasetGenerator.generate()
        │  ChatOpenAI(temp=0.7, max_tokens=2000)
        │  → 5 triples per extract (instruction, rejected, chosen)
        ▼
post_process_datasets()
        │  filter_short_answers(min_length=100)
        │  filter_answer_format()  (capital start, terminal . ! ?)
        ▼
create_preference_train_test_split(random_state=42)
        │
        ▼
PreferenceTrainTestSplit
        │
        ▼  (optional)
push_to_huggingface  →  prompt / rejected / chosen
```

---

## 🧠 DPO in One Page

DPO replaces a separate reward model with a direct loss over preference pairs.

For each triple `(prompt x, chosen y_w, rejected y_l)`:

```
L_DPO = - log σ( β · [ log πθ(y_w | x) - log πref(y_w | x)
                       - ( log πθ(y_l | x) - log πref(y_l | x) ) ] )
```

- **`πθ`** is the model being trained; **`πref`** is a frozen reference (usually the SFT checkpoint).
- The inner term compares how much *more* likely the policy makes the chosen answer versus the rejected one, relative to the reference.
- **`β`** controls how far the policy may drift from the reference (a KL-style regularizer).
- The SFT/instruction stage (Session 3.1 / 5.1) is the usual reference point, which is why this pipeline produces **both** dataset types.

**Why the `chosen` answer must be verbatim**: DPO assumes `chosen` is genuinely better. A re-written "good answer" can drift from the source's style and facts. This dataset's chosen answer is an exact excerpt from the corpus, guaranteeing a high-quality, on-style target.

---

## 📁 Key Files Explained

### 1. `PreferenceDatasetGenerator` - The Prompt

```python
# llm_engineering/application/dataset/generation.py
class PreferenceDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.PREFERENCE

    prompt_template_str = """Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
1. An instruction asking about a specific topic in the context.
2. A generated answer that attempts to answer the instruction based on the context, named as 'rejected'.
3. An extracted answer that is a relevant excerpt directly from the given context, named as 'chosen'.

Instructions must be self-contained and general, without explicitly mentioning a context, system, course, or extract.

Important:
- Ensure that the extracted answer, the chosen one, is a verbatim copy from the context, including all punctuation and apostrophes.
- Do not add any ellipsis (...) or [...]  to indicate skipped text in the extracted answer.
- If the relevant text is not continuous, use two separate sentences from the context instead of skipping text.

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {
        "instruction": "...",
        "rejected": "...",
        "chosen": "..."
    },
    ...
]

Extract:
{{extract}}
"""
```

**Prompt-design principles**:
- **`rejected` is model-generated** ("attempts to answer... based on the context"): plausibly fluent but not guaranteed faithful.
- **`chosen` is extracted from the context**: verbatim, so it inherits the author's style and facts.
- **Verbatim rules** ("including all punctuation and apostrophes", "no ellipsis") exist so the chosen string is a true quotation, not a summary.
- **The discontiguous-rule** ("use two separate sentences instead of skipping text") prevents fabricated continuity.

> ⚠️ Note: some local files (for example `steps/generate_datasets/generate_preference_dataset.py`) use the misspelling `preference`. The enum is `DatasetType.PREFERENCE`. Use the enum, not a string literal.

---

### 2. Post-Processing: The Two Filters

```python
# llm_engineering/application/dataset/generation.py
    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.PreferenceDataset], test_size: float
    ) -> TrainTestSplit:
        datasets = generation_utils.filter_short_answers(datasets)
        datasets = generation_utils.filter_answer_format(datasets)

        remaining_samples = sum([dataset.num_samples for dataset in datasets.values()])
        logger.info(
            f"Filtered out short answers and answers with incorrect format. Remaining samples: {remaining_samples}"
        )

        train_test_split = generation_utils.create_preference_train_test_split(
            datasets, test_size=test_size, random_state=42
        )
        return train_test_split
```

**Filter 1 - minimum length**:

```python
def filter_short_answers(
    data: dict[DataCategory, PreferenceDataset], min_length: int = 100
) -> dict[DataCategory, PreferenceDataset]:
    def is_long_enough(example: PreferenceDatasetSample) -> bool:
        return len(example.chosen) >= min_length

    filtered_data = {}
    for category, dataset in data.items():
        filtered_dataset = PreferenceDataset(
            category=category,
            samples=list(filter(is_long_enough, dataset.samples)),
        )
        filtered_data[category] = filtered_dataset
    return filtered_data
```

**Filter 2 - answer shape**:

```python
def filter_answer_format(data: dict[DataCategory, PreferenceDataset]) -> dict[DataCategory, PreferenceDataset]:
    def is_valid_format(example: PreferenceDatasetSample) -> bool:
        chosen = example.chosen
        return len(chosen) > 0 and chosen[0].isupper() and chosen[-1] in (".", "!", "?")

    filtered_data = {}
    for category, dataset in data.items():
        filtered_dataset = PreferenceDataset(
            category=category,
            samples=list(filter(is_valid_format, dataset.samples)),
        )
        filtered_data[category] = filtered_dataset
    return filtered_data
```

**Key Concepts**:
- **Length filter (>= 100 chars on `chosen`)**: a one-line chosen answer carries almost no signal for DPO; the contrast between chosen and rejected would be trivial.
- **Format filter (capital start, terminal punctuation)**: enforces that `chosen` is a real sentence, not a fragment. It complements the model-side preprocessing in the book by checking the answer shape, not the prompt shape.
- **Both filters drop whole samples**, never edit them. Editing a verbatim quotation would break the "chosen is authentic" guarantee.
- **Filters run before the split**, so the test set is also clean.

---

### 3. The Schema and Hugging Face Export

```python
# llm_engineering/domain/dataset.py
class PreferenceDatasetSample(VectorBaseDocument):
    instruction: str
    rejected: str
    chosen: str

    class Config:
        category = DataCategory.PREFERENCE_DATASET_SAMPLES


class PreferenceDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[PreferenceDatasetSample]

    class Config:
        category = DataCategory.PREFERENCE_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]
        return Dataset.from_dict({
            "prompt": [d["instruction"] for d in data],
            "rejected": [d["rejected"] for d in data],
            "chosen": [d["chosen"] for d in data],
        })
```

**Key Concepts**:
- **Column remap on export**: internal `instruction` becomes the Hugging Face `prompt` column. This is exactly the schema `trl`'s DPO trainer expects.
- **`chosen` / `rejected` names are not arbitrary** - they are the reserved names in the TRL DPO data collator.
- The generation `max_tokens=2000` (vs 1200 for pairs) is set in the base `generate()` because triples are longer.

---

### 4. Splitting

```python
def create_preference_train_test_split(
    data: dict[DataCategory, PreferenceDataset], test_size=0.2, random_state=42
) -> PreferenceTrainTestSplit:
    train_data = {}
    test_data = {}
    for category, dataset in data.items():
        samples_dicts = [sample.model_dump() for sample in dataset.samples]
        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )
            train_samples = [PreferenceDatasetSample(**s) for s in train_samples_dicts]
            test_samples = [PreferenceDatasetSample(**s) for s in test_samples_dicts]
        else:
            train_samples, test_samples = [], []

        train_data[category] = PreferenceDataset(category=category, samples=train_samples)
        test_data[category] = PreferenceDataset(category=category, samples=test_samples)

    return PreferenceTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)
```

Mirrors the instruction splitter: per category, seed 42, empty-safe.

---

### 5. The `generate_preference_dataset` Step

```python
# steps/generate_datasets/generate_preference_dataset.py
@step
def generate_preference_dataset(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"],
    test_split_size: Annotated[float, "test_split_size"],
    mock: Annotated[bool, "mock_generation"] = False,
) -> Annotated[
    PreferenceTrainTestSplit,
    ArtifactConfig(name="preference_datasets", tags=["dataset", "preference", "cleaned"]),
]:
    dataset_generator = generation.get_dataset_generator(DatasetType.PREFERENCE)
    datasets = dataset_generator.generate(prompts, test_size=test_split_size, mock=mock)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="preference_datasets", metadata=_get_metadata_preference_dataset(datasets)
    )
    return datasets
```

The metadata records `train_num_samples_per_category` and `test_num_samples_per_category`, so the ZenML dashboard shows exactly how many triples survived filtering.

---

## 🔧 Running It

### Instruction dataset

```bash
python -m tools.run --run-generate-instruct-datasets --no-cache
```

### Preference dataset

```bash
python -m tools.run --run-generate-preference-datasets --no-cache
```

Both are driven by configs under `configs/`:
- `configs/generate_instruct_datasets.yaml`
- `configs/generate_preference_datasets.yaml`

### Mock mode

Set `mock: true` in the config (or pass `mock=True`) to use `FakeListLLM` from `constants.py`. This produces three canned triples with no API calls - useful for CI and for validating the pipeline shape.

---

## 🛠️ Hands-On: Inspect and Validate Preference Data

### Step 1: Generate with mock

```python
from llm_engineering.application.dataset import generation
from llm_engineering.domain.dataset import DatasetType
from llm_engineering.domain.cleaned_documents import CleanedArticleDocument

docs, offset = CleanedArticleDocument.bulk_find(limit=1)
gen = generation.get_dataset_generator(DatasetType.PREFERENCE)
prompts = gen.get_prompts(docs)
split = gen.generate(prompts, test_size=0.2, mock=True)
```

### Step 2: Watch the filters reject mock data

```python
for category, ds in split.train.items():
    print(category, ds.num_samples)
# The mocked "chosen" with the long repeated 'extracted' passes;
# the short "Mocked extracted answer 3" is filtered out.
```

### Step 3: Verify the verbatim property

```python
sample = next(s for ds in split.train.values() for s in ds.samples)
print("chosen in source:", sample.chosen in docs[0].content)
```

### Step 4: Export and check columns

```python
hf = next(iter(split.train.values())).to_huggingface()
print(hf.column_names)  # ['prompt', 'rejected', 'chosen']
```

---

## 📝 Exercise: Add a Third Quality Filter

### Task

Add a filter that removes triples where `rejected` is too similar to `chosen`.

```python
from difflib import SequenceMatcher

def filter_similar_rejected(
    data: dict[DataCategory, PreferenceDataset], max_similarity: float = 0.9
) -> dict[DataCategory, PreferenceDataset]:
    def is_distinct(example: PreferenceDatasetSample) -> bool:
        ratio = SequenceMatcher(None, example.chosen, example.rejected).ratio()
        return ratio < max_similarity
    ...
```

Then:
1. Call it inside `PreferenceDatasetGenerator.post_process_datasets`.
2. Run with mock data and confirm it changes the sample count.
3. Explain why near-identical pairs give DPO almost no gradient signal.

**Goal**: Understand that DPO needs a **meaningful contrast**. If chosen and rejected are nearly the same, the preference signal is weak.

---

## 🐛 Common Pitfalls

- **`chosen` not verbatim**: if the model paraphrases, DPO learns the wrong target. The verbatim rules in the prompt plus the format filter guard this, but always spot-check real runs.
- **Too few samples after filtering**: aggressive filters can empty a small category. The pipeline logs the remaining count, but a silent empty train set will fail later at training time.
- **Category mismatch**: use `DatasetType.PREFERENCE` (the enum). Typos in the dataset-type string raise `ValueError` inside `get_dataset_generator`.

---

## 🎓 Knowledge Check

1. **In DPO, which answer is model-generated and which is from the corpus?**
   - Answer: `rejected` is model-generated; `chosen` is a verbatim excerpt from the context.

2. **Why must `chosen` be verbatim?**
   - Answer: DPO assumes chosen is genuinely better; a paraphrase could drift from the source's facts and style.

3. **What does `filter_short_answers` protect against?**
   - Answer: Trivially short chosen answers that give almost no preference signal.

4. **What does `filter_answer_format` check?**
   - Answer: Chosen starts with a capital and ends with `.`, `!`, or `?`.

5. **What are the Hugging Face DPO column names, and what do they map from?**
   - Answer: `prompt`/`chosen`/`rejected`, mapped from `instruction`/`chosen`/`rejected`.

6. **Why is SFT a prerequisite for DPO in this project?**
   - Answer: The DPO reference policy is the SFT checkpoint; without it, there is no sensible `πref`.

---

## 🔗 Next Session

**Session 4.1**: Advanced RAG Architecture

We return to retrieval and build the multi-stage RAG pipeline (already documented in `session_4.1_advanced_rag.md`).

---

## 📚 Additional Resources

- [Direct Preference Optimization (Rafailov et al.)](https://arxiv.org/abs/2305.18290)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Session 3.1

**Outcome**: You can generate, filter, split, and export a DPO-ready preference dataset and explain the role of each filter.
