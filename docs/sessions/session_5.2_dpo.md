# Session 5.2: Direct Preference Optimization (DPO)

## 🎯 Learning Objectives

By the end of this session, you will:
- Explain the DPO objective and the role of `beta`
- Format preference triples for `DPOTrainer`
- Configure DPO to run on top of the SFT checkpoint
- Compare SFT-only output against SFT+DPO output
- Understand why DPO uses a much smaller learning rate than SFT

---

## 🏗️ Architecture Overview

```
SFT checkpoint (Session 5.1)
   {workspace}/TwinLlama-3.1-8B
        │  (this is π_ref, the frozen reference)
        ▼
┌────────────────────────────────────────────────────────────────┐
│                     finetune(finetuning_type="dpo")            │
│                                                                 │
│  load_model(sft_checkpoint, load_in_4bit=...)                  │
│  PatchDPOTrainer()                                             │
│        │                                                        │
│  format_samples_dpo()                                          │
│        │  prompt  = alpaca_template.format(prompt, "")         │
│        │  chosen  = chosen + EOS                               │
│        │  rejected= rejected + EOS                             │
│        ▼                                                        │
│  DPOTrainer(beta=0.5, ref_model=None,                          │
│             max_length=1024, max_prompt_length=1024)           │
│        │  DPOConfig(lr=2e-6, epochs=1, optim=adamw_8bit)       │
│        ▼                                                        │
│  save_pretrained_merged  →  TwinLlama-3.1-8B-DPO               │
└────────────────────────────────────────────────────────────────┘
```

---

## 🧠 The DPO Objective, Precisely

```
L_DPO(πθ; πref) = - E_(x, y_w, y_l) [ log σ( β · Δ ) ]

Δ = log(πθ(y_w|x) / πref(y_w|x)) - log(πθ(y_l|x) / πref(y_l|x))
```

- **`πref`** is the frozen SFT model; **`πθ`** is the policy being trained (starting from the same weights).
- **`β` (beta)** controls how strongly the policy is pulled back toward the reference. Low beta allows large deviations; high beta keeps the model close to SFT.
- The loss increases the **relative** log-probability of chosen versus rejected, normalized by the reference. This is why the reference matters: it subtracts out what the SFT model already finds likely, so DPO only learns the *preference* signal.
- No reward model and no reinforcement loop are needed, unlike PPO-based RLHF - it is a single supervised-style loss over pairs.

**Why the learning rate is tiny (2e-6 vs 3e-4 for SFT)**: DPO starts from a good model and makes a small, targeted adjustment. A large LR would blow past the reference and collapse the preference signal (and often destroy the SFT quality).

---

## 📁 Key Files Explained

### 1. Formatting the Preference Triples

```python
# llm_engineering/model/finetuning/finetune.py (dpo branch)
    elif finetuning_type == "dpo":
        PatchDPOTrainer()

        def format_samples_dpo(example):
            example["prompt"] = alpaca_template.format(example["prompt"], "")
            example["chosen"] = example["chosen"] + EOS_TOKEN
            example["rejected"] = example["rejected"] + EOS_TOKEN

            return {"prompt": example["prompt"], "chosen": example["chosen"], "rejected": example["rejected"]}

        dataset = load_dataset(f"{dataset_huggingface_workspace}/llmtwin-dpo", split="train")
```

**Key Concepts**:
- **The prompt uses the Alpaca template with an empty response**: `alpaca_template.format(prompt, "")` ends at `### Response:\n`, so the model is asked to produce the answer.
- **Both chosen and rejected get an EOS token**, matching the SFT formatting. The loss compares the full continuation probabilities of each.
- **Column names `prompt`/`chosen`/`rejected`** are exactly what the `PreferenceDataset.to_huggingface()` export produced in Session 3.2 - the schema is intentionally aligned end to end.

---

### 2. Dataset Loading and Dummy Mode

```python
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")
        print(f"Loaded dataset with {len(dataset)} samples.")

        dataset = dataset.map(format_samples_dpo)
        dataset = dataset.train_test_split(test_size=0.05)
```

Same dummy/split pattern as SFT: 400 samples for smoke tests, 5% evaluation split.

---

### 3. `DPOTrainer` Configuration

```python
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            beta=beta,                                  # default 0.5
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=max_seq_length // 2,             # 2048 // 2 = 1024
            max_prompt_length=max_seq_length // 2,
            args=DPOConfig(
                learning_rate=learning_rate,            # 2e-6 from __main__
                num_train_epochs=num_train_epochs,      # 1
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                eval_steps=0.2,
                logging_steps=1,
                report_to="comet_ml",
                seed=0,
            ),
        )
```

**Key Concepts**:
- **`ref_model=None`** is the Unsloth optimization: the reference is the *disabled adapter* of the same model. With PEFT, disabling the LoRA adapter recovers the frozen base, so no second model copy is loaded. This roughly halves VRAM versus loading an explicit reference.
- **`beta=0.5`** is a moderately strong KL anchor. Lower values (0.1) allow bigger stylistic shifts; higher values (0.9) stay close to SFT.
- **`max_length = max_seq_length // 2 = 1024`**: DPO stores both chosen and rejected completions, so each sequence budget is halved to fit both in memory.
- **`max_prompt_length`** caps the shared prompt portion.
- **`eval_steps=0.2`** evaluates 5 times per epoch (fraction-based), which is useful given DPO runs only 1 epoch.

---

### 4. The `__main__` DPO Entry Point

```python
    elif args.finetuning_type == "dpo":
        sft_base_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
        sft_base_model_repo_id = check_if_huggingface_model_exists(sft_base_model_repo_id)
        print(f"Training from base model '{sft_base_model_repo_id}'")

        output_dir_dpo = Path(args.model_dir) / "output_dpo"
        model, tokenizer = finetune(
            finetuning_type="dpo",
            model_name=sft_base_model_repo_id,
            output_dir=str(output_dir_dpo),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            num_train_epochs=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=2e-6,
            is_dummy=args.is_dummy,
        )
        inference(model, tokenizer)

        dpo_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B-DPO"
        save_model(model, tokenizer, "model_dpo", push_to_hub=True, repo_id=dpo_output_model_repo_id)
```

**Key Concepts**:
- **`check_if_huggingface_model_exists`** falls back to `mlabonne/TwinLlama-3.1-8B` if your SFT model is not on the Hub. This is a convenience so readers can run DPO without having trained SFT themselves.
- **DPO starts from the SFT checkpoint**, not the raw Llama base. This is what makes the reference meaningful.
- **`learning_rate=2e-6`, `num_train_epochs=1`** are the production values, overriding the general defaults.

---

## 🔬 Deep Dive: SFT vs DPO

| Aspect | SFT | DPO |
|--------|-----|-----|
| Data | `instruction` / `output` | `prompt` / `chosen` / `rejected` |
| Objective | Cross-entropy on the target answer | Preference contrast vs reference |
| Reference model | none | the SFT checkpoint (`πref`) |
| Learning rate | `3e-4` | `2e-6` |
| Epochs | 3 | 1 |
| Memory | base + adapter + one sequence | base + adapter + two sequences (chosen/rejected) |
| Output | `TwinLlama-3.1-8B` | `TwinLlama-3.1-8B-DPO` |

**What changes in behaviour**: SFT makes the model answer in your style. DPO makes the model prefer the *style and faithfulness* of the verbatim corpus excerpts over its own generated attempts. The result is a model that is both on-style and more faithful to the source.

---

## 🛠️ Hands-On: Run and Compare DPO

### Step 1: Confirm the SFT checkpoint

```python
from huggingface_hub import HfApi
api = HfApi()
print(api.model_info("<workspace>/TwinLlama-3.1-8B").id)
```

### Step 2: Inspect a formatted DPO sample

```python
from datasets import load_dataset

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

ds = load_dataset("<workspace>/llmtwin-dpo", split="train")
ex = ds[0]
print("PROMPT:", alpaca_template.format(ex["prompt"], ""))
print("CHOSEN:", ex["chosen"])
print("REJECTED:", ex["rejected"])
```

### Step 3: Dummy DPO run

```bash
python llm_engineering/model/finetuning/finetune.py \
    --finetuning_type dpo \
    --is_dummy True \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-6
```

### Step 4: Compare generations

Load `TwinLlama-3.1-8B` and `TwinLlama-3.1-8B-DPO`, run the same prompt through both, and compare tone and faithfulness. The DPO model should more often reproduce corpus phrasing.

---

## 📝 Exercise: Beta Sweep

### Task

Understand how `beta` balances style and safety.

1. Run DPO three times with `beta=0.1`, `0.5`, and `0.9` (dummy mode).
2. For each, generate the same three prompts.
3. Judge: which is most faithful? Which drifts most from SFT?
4. Explain how beta is acting as a KL anchor.

**Goal**: See the tradeoff between moving toward preferred behaviour and staying close to the trusted SFT model.

---

## 🐛 Common Pitfalls

- **Starting DPO from the base model**: without an SFT reference the preferences have no anchor; quality degrades. Always start from `TwinLlama-3.1-8B`.
- **Learning rate too high**: `3e-4` (the SFT value) will wreck the model. DPO needs `~2e-6`.
- **Sequence budget**: `max_length = max_seq_length // 2`; raising it without lowering batch size will OOM because two completions are held per example.
- **Empty `chosen` after filtering**: if Session 3.2 filters removed almost everything, DPO trains on too little data. Check the sample count first.

---

## 🎓 Knowledge Check

1. **What plays the role of `πref` in this project?**
   - Answer: The SFT checkpoint `TwinLlama-3.1-8B`, reused via Unsloth's disabled-adapter mechanism (`ref_model=None`).

2. **Why is `beta` called a KL anchor?**
   - Answer: It controls how far the policy may deviate from the reference; higher beta keeps it closer.

3. **Why does DPO use two completions per example in memory?**
   - Answer: The loss compares chosen and rejected, so both sequences must be present.

4. **Why is the DPO learning rate ~150× smaller than SFT's?**
   - Answer: DPO makes a small alignment adjustment from an already-good model; a large LR would overshoot.

5. **What does `check_if_huggingface_model_exists` guard against?**
   - Answer: A missing custom SFT model, by falling back to the public `mlabonne/TwinLlama-3.1-8B`.

6. **What schema does the DPO dataset use?**
   - Answer: `prompt`, `chosen`, `rejected` - matching the Session 3.2 export exactly.

---

## 🔗 Next Session

**Session 5.3**: AWS SageMaker Deployment

We deploy the DPO model to a real-time SageMaker endpoint with autoscaling.

---

## 📚 Additional Resources

- [Direct Preference Optimization (Rafailov et al.)](https://arxiv.org/abs/2305.18290)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [Unsloth DPO Guide](https://docs.unsloth.ai/basics/dpo)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 3.2, 5.1

**Outcome**: You can align an SFT model with preference data and reason about the beta/learning-rate tradeoffs.
