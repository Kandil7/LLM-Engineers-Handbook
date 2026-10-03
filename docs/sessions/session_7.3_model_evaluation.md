# Session 7.3: Model Evaluation

## 🎯 Learning Objectives

By the end of this session, you will:
- Generate test-set answers with vLLM
- Implement an LLM-as-a-judge evaluation (accuracy + style)
- Run large-scale evaluation in parallel batches
- Compare SFT, DPO, and an Instruct baseline
- Understand pointwise versus pairwise judging and their biases

---

## 🏗️ Architecture Overview

```
test split of llmtwin dataset
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ generate_answers(model_id, dataset)                               │
│   vLLM: LLM(model_id, max_model_len=2048)                         │
│   SamplingParams(temperature=0.8, top_p=0.95, min_p=0.05, 2048)   │
│   → push "{workspace}/{model}-results" to the Hub                 │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ evaluate_answers(model_id)                                        │
│   batch samples, ThreadPoolExecutor(max_workers=10)               │
│   evaluate_answer(): gpt-4o-mini judge → accuracy + style scores  │
│   → add accuracy/style columns, re-push to the Hub                │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ analyze: mean accuracy / mean style per model                     │
└──────────────────────────────────────────────────────────────────┘
```

**Models compared**:

```python
model_ids = [
    check_if_huggingface_model_exists(f"{MODEL_HUGGINGFACE_WORKSPACE}/TwinLlama-3.1-8B", "mlabonne/TwinLlama-3.1-8B"),
    check_if_huggingface_model_exists(f"{MODEL_HUGGINGFACE_WORKSPACE}/TwinLlama-3.1-8B-DPO", "mlabonne/TwinLlama-3.1-8B-DPO"),
    "meta-llama/Llama-3.1-8B-Instruct",
]
```

SFT vs DPO answers the alignment question. The Instruct baseline anchors absolute quality.

---

## 📁 Key Files Explained

### 1. `evaluate.py` - Generation with vLLM

```python
# llm_engineering/model/evaluation/evaluate.py
def generate_answers(model_id: str, dataset_name: str):
    def format(sample):
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n".format(
            sample["instruction"]
        )

    dataset = load_dataset(dataset_name, split="test")
    if IS_DUMMY:
        dataset = dataset.select(range(10))
    dataset = dataset.map(lambda sample: {"prompt": format(sample)})

    llm = LLM(model=model_id, max_model_len=2048)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_p=0.05, max_tokens=2048)
    outputs = llm.generate(dataset["prompt"], sampling_params)

    answers = [output.outputs[0].text for output in outputs]
    dataset = dataset.add_column("answers", answers)

    dataset.push_to_hub(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results")
    gc.collect()
    return dataset
```

**Key Concepts**:
- **vLLM** gives high-throughput batched generation via PagedAttention. It is the standard tool for evaluating many prompts quickly.
- **The `format` prompt matches the Alpaca inference format** used elsewhere, so the model sees the same interface it was trained and served with.
- **`split="test"`** uses the held-out split created during dataset generation (Session 3.1) - evaluation never touches training data.
- **`gc.collect()`** frees GPU memory before loading the next model, which matters when evaluating three models sequentially.
- **Results are pushed to the Hub** under `{workspace}/{model}-results`, so evaluation is versioned and shareable.

**Sampling parameters**: `temperature=0.8`, `top_p=0.95`, `min_p=0.05` produce varied but coherent generations - appropriate when you want to judge typical model output, not the single most likely answer.

---

### 2. The Judge Prompt

```python
def evaluate_answer(instruction: str, answer: str, client: OpenAI) -> dict:
    prompt = f"""You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:
1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

Accuracy scale:
1 (Poor): Contains factual errors or misleading information
2 (Good): Mostly accurate with minor errors or omissions
3 (Excellent): Highly accurate and comprehensive

Style scale:
1 (Poor): Too formal, uses some overly complex words
2 (Good): Good balance of technical content and accessibility, but still uses formal words and expressions
3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.

Instruction: {instruction}
Answer: {answer}

Provide your evaluation in JSON format with the following structure:
{{
    "accuracy": {{ "analysis": "...", "score": 0 }},
    "style": {{ "analysis": "...", "score": 0 }}
}}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who evaluates answers based on accuracy and style..."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        temperature=0.9,
    )
    return json.loads(completion.choices[0].message.content)
```

**Key Concepts**:
- **Two orthogonal criteria**: accuracy (factual correctness) and style (accessible blog/social tone). The project's twin is judged on style as much as correctness.
- **Anchored 1–3 scales with explicit descriptions** reduce judge variance. Each score level has a concrete definition.
- **Worked style examples (bad vs excellent)** are the strongest lever for consistency; the judge copies the distinction.
- **`response_format={"type": "json_object"}`** forces valid JSON, so no fence-stripping is needed.
- **`temperature=0.9`** adds judge variability. This is intentional for diversity but means scores are noisy; report means over the dataset, not single judgments.

---

### 3. Parallel Batch Evaluation

```python
def evaluate_batch(batch, start_index):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return [(i, evaluate_answer(instr, ans, client)) for i, (instr, ans) in enumerate(batch, start=start_index)]


def evaluate_answers(model_id: str, num_threads: int = 10, batch_size: int = 5) -> Dataset:
    dataset = load_dataset(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results", split="all")

    batches = [
        (i, list(zip(dataset["instruction"][i : i + batch_size], dataset["answers"][i : i + batch_size], strict=False)))
        for i in range(0, len(dataset), batch_size)
    ]

    evaluations = [None] * len(dataset)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(evaluate_batch, batch, start_index) for start_index, batch in batches]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            for index, evaluation in future.result():
                evaluations[index] = evaluation
    ...
```

**Key Concepts**:
- **Indexed batches**: each batch carries its start index, and results are placed back at the original index. This preserves alignment even though futures complete out of order.
- **`ThreadPoolExecutor(max_workers=10)`** parallelizes the judge API calls. Ten workers balances throughput against OpenAI rate limits.
- **`tqdm`** shows progress, which matters for long evaluations.
- **Error tolerance in post-processing**: any malformed evaluation appends `None` for accuracy and style, keeping the columns aligned with the dataset.

**Score aggregation**:

```python
def evaluate_answers(...):
    ...
    for evaluation in dataset["evaluation"]:
        try:
            eval_dict = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
            accuracy_scores.append(eval_dict["accuracy"]["score"])
            style_scores.append(eval_dict["style"]["score"])
        except (json.JSONDecodeError, KeyError, TypeError):
            accuracy_scores.append(None)
            style_scores.append(None)

    dataset = dataset.add_column("accuracy", accuracy_scores)
    dataset = dataset.add_column("style", style_scores)
    dataset.push_to_hub(...)
```

`None` entries keep alignment; when averaging, skip them or treat missing as excluded.

---

### 4. Running on SageMaker

```python
# llm_engineering/model/evaluation/sagemaker.py
def run_evaluation_on_sagemaker(is_dummy: bool = True) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.OPENAI_API_KEY, "OpenAI API key is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    env = {
        "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "DATASET_HUGGINGFACE_WORKSPACE": huggingface_user,
        "MODEL_HUGGINGFACE_WORKSPACE": huggingface_user,
    }
    if is_dummy:
        env["IS_DUMMY"] = "True"

    hfp = HuggingFaceProcessor(
        role=settings.AWS_ARN_ROLE,
        instance_count=1,
        instance_type="ml.g5.2xlarge",
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        base_job_name="evaluate-llm-twin",
        env=env,
    )
    hfp.run(code="evaluate.py", source_dir=str(evaluation_dir))
```

**Key Concepts**:
- **`HuggingFaceProcessor`** runs a SageMaker *processing* job (not training) - the right primitive for batch generation + judging.
- **Env vars carry the three credentials**: HF token (model/dataset access), OpenAI key (judge), and the workspaces.
- **`is_dummy`** sets `IS_DUMMY=True` in the container, which `evaluate.py` reads to trim the dataset to 10 samples.

---

### 5. The ZenML Pipeline

```python
# pipelines/evaluating.py
@pipeline
def evaluating(is_dummy: bool = False) -> None:
    evaluating_steps.evaluate(is_dummy=is_dummy)
```

```python
# steps/evaluating/evaluate.py
@step
def evaluate(is_dummy: bool = False) -> None:
    run_evaluation_on_sagemaker(is_dummy=is_dummy)
```

Run with:

```bash
python -m tools.run --run-evaluation
```

---

## 🔬 Deep Dive: LLM-as-a-Judge

### Pointwise vs pairwise

| Mode | Input | Output | Project use |
|------|-------|--------|-------------|
| Pointwise | one answer | absolute score | ✅ (accuracy/style 1–3) |
| Pairwise | two answers | which is better | not used here |

Pointwise scales to many models and one-dimensional criteria, and it is what `evaluate_answer` implements. Pairwise is more reliable for close comparisons but requires pairing every model with every other.

### Known biases and mitigations

- **Position bias** (pairwise only): the first answer wins more often. Not applicable to pointwise.
- **Verbosity bias**: longer answers tend to score higher. The project controls for this by using concise style examples and a style criterion.
- **Self-preference**: a judge may favor answers resembling its own outputs. Using `gpt-4o-mini` as judge for Llama models partially avoids this.
- **Noise from `temperature=0.9`**: report the **mean over the dataset**, and treat single-point differences between models as non-significant.

### Why accuracy *and* style

An LLM Twin must both be correct and *sound like the author*. A model that is accurate but formal fails the style criterion; one that is stylish but wrong fails accuracy. The two scores together capture the twin objective.

---

## 🛠️ Hands-On: Run a Small Evaluation

### Step 1: Dummy end-to-end

```bash
python -m tools.run --run-evaluation
# or directly:
python llm_engineering/model/evaluation/evaluate.py   # with IS_DUMMY=True
```

### Step 2: In a local environment (GPU)

```bash
export OPENAI_API_KEY=...
export DATASET_HUGGINGFACE_WORKSPACE=<you>
export MODEL_HUGGINGFACE_WORKSPACE=<you>
export IS_DUMMY=True
python llm_engineering/model/evaluation/evaluate.py
```

### Step 3: Inspect results on the Hub

Open `{workspace}/TwinLlama-3.1-8B-DPO-results` and browse the `answers`, `evaluation`, `accuracy`, and `style` columns.

### Step 4: Compare means

```python
from datasets import load_dataset
for m in ["TwinLlama-3.1-8B", "TwinLlama-3.1-8B-DPO"]:
    ds = load_dataset("<workspace>/" + m + "-results", split="all")
    acc = [a for a in ds["accuracy"] if a is not None]
    sty = [s for s in ds["style"] if s is not None]
    print(m, "accuracy", sum(acc)/len(acc), "style", sum(sty)/len(sty))
```

---

## 📝 Exercise: Add a Pairwise Comparison

### Task

Implement a pairwise judge between two models' answers to the same instruction.

1. Load both `-results` datasets.
2. For each instruction, ask `gpt-4o-mini` which answer is better, **randomizing which is A/B** and tracking the mapping.
3. Report a win rate for DPO vs SFT.
4. Compare the pairwise result with the pointwise means.

**Goal**: Experience why pairwise judging is preferred for close comparisons and how to defend against position bias by randomizing order.

---

## 🐛 Common Pitfalls

- **vLLM VRAM**: loading an 8B model for generation needs substantial VRAM; the batch is token-bounded, but `max_model_len=2048` still needs headroom. On a 16 GB card, evaluate one model at a time and `gc.collect()` between them.
- **Judge cost**: one judge call per sample. Estimate `samples × models` before a full run.
- **Missing model/data**: the `check_if_huggingface_*_exists` fallbacks keep the script running with public defaults, which can silently evaluate the wrong model. Read the printed warnings.
- **Unkilled nulls**: if all evaluations failed for a sample, `accuracy`/`style` are `None`; ensure averages skip them.

---

## 🎓 Knowledge Check

1. **What two criteria does the judge score?**
   - Answer: accuracy (factual correctness) and style (accessible blog/social tone), each 1–3.

2. **Why use vLLM for generation?**
   - Answer: High-throughput batched inference via PagedAttention.

3. **How are out-of-order batch results re-aligned?**
   - Answer: Each batch carries its start index; results are written back at the original index.

4. **Why does the post-processing append `None` on parse errors?**
   - Answer: To keep accuracy/style columns aligned with the dataset.

5. **Which model is the absolute-quality baseline?**
   - Answer: `meta-llama/Llama-3.1-8B-Instruct`.

6. **What is the main bias of pointwise judging, and how is it mitigated here?**
   - Answer: Verbosity/length bias; mitigated with a style criterion and concise style examples - and by averaging over the dataset.

---

## 🔗 Next Session

**Session 8.1**: Docker & Local Infrastructure

We containerize the application and study the local service topology.

---

## 📚 Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [LLM-as-a-Judge (Zheng et al.)](https://arxiv.org/abs/2306.05685)
- [SageMaker Hugging Face Processor](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/)
- [Ragas metrics](https://docs.ragas.io/)

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Sessions 3.1, 5.1, 5.2, 5.3

**Outcome**: You can generate, judge, and compare model outputs with an LLM-as-a-judge pipeline, and reason about its biases.
