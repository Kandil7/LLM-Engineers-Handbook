# Session 5.1: Supervised Fine-Tuning (SFT)

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand the Unsloth + LoRA + 4-bit QLoRA training stack
- Read the `finetune()` function end to end
- Format instruction-answer samples into the Alpaca template
- Configure `SFTTrainer` for memory-efficient training
- Merge and publish adapters back to a full model

---

## 🏗️ Architecture Overview

### The SFT Stage in Context

```
Instruction dataset (Session 3.1)
        │  Hugging Face: {workspace}/llmtwin
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     finetune(finetuning_type="sft")            │
│                                                                │
│  FastLanguageModel.from_pretrained(base, load_in_4bit=True)   │
│        │  4-bit NF4 weights (frozen)                          │
│        ▼                                                       │
│  FastLanguageModel.get_peft_model(r=32, ...)                  │
│        │  trainable LoRA adapters on q/k/v/o/gate/up/down      │
│        ▼                                                       │
│  format_samples_sft()  →  alpaca_template + EOS               │
│        ▼                                                       │
│  SFTTrainer(packing=True, bf16, adamw_8bit)                   │
│        ▼                                                       │
│  save_pretrained_merged(save_method="merged_16bit")           │
│        ▼                                                       │
│  push_to_hub  →  {workspace}/TwinLlama-3.1-8B                 │
└───────────────────────────────────────────────────────────────┘
```

The output of this stage becomes the **base for DPO** (Session 5.2) and the **RAG generator** model.

---

## 📁 Key Files Explained

### 1. `finetune.py` - The Patch-First Import Order

```python
# llm_engineering/model/finetuning/finetune.py
from unsloth import PatchDPOTrainer

PatchDPOTrainer()

from typing import Any, List, Literal, Optional  # noqa: E402

import torch  # noqa
from datasets import concatenate_datasets, load_dataset  # noqa: E402
from transformers import TextStreamer, TrainingArguments  # noqa: E402
from trl import DPOConfig, DPOTrainer, SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402
```

**Key Concepts**:
- **`PatchDPOTrainer()` must run before importing TRL trainers.** It monkey-patches TRL for Unsloth compatibility, so the import order is load-bearing. The `# noqa: E402` comments silence the "imports not at top" lint that this ordering creates.
- **Everything is imported through Unsloth** (`FastLanguageModel`, `is_bfloat16_supported`), which provides fused kernels and optimized Triton implementations.

---

### 2. `load_model` - 4-bit Base + LoRA Adapters

```python
def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    chat_template: str,
) -> tuple:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    return model, tokenizer
```

**Key Concepts**:
- **`load_in_4bit=True`** quantizes the frozen base weights to 4-bit NF4, cutting base memory roughly 4×. Forward/backward run in higher precision; only the adapters are stored in full precision.
- **`r=32, alpha=32`**: the LoRA rank and scaling. `alpha == r` keeps the effective update scale near 1. Higher rank fits more but uses more memory.
- **`target_modules`** span attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`). Adapting the MLP projections is what lets a small rank capture style changes.
- **`lora_dropout=0.0`**: Unsloth recommends 0 for its fused kernels.
- **`get_chat_template(tokenizer, "chatml")`** installs the ChatML template so the tokenizer knows the special tokens for this model family.

**VRAM reality check (RTX 5000, 16 GB)**:

| Component | Approx size (7–8B) |
|-----------|--------------------|
| 4-bit base weights | ~4–5 GB |
| LoRA adapters (r=32) | <1 GB |
| Optimizer (adamw_8bit over adapters) | small |
| Activations + gradients (seq 2048, batch 2) | several GB |

A local 8B QLoRA run at `max_seq_length=2048, batch=2, grad_accum=8` is tight on 16 GB and may OOM. The project's default path is **SageMaker `ml.g5.2xlarge` (24 GB A10G)**. Reduce `max_seq_length` or `per_device_train_batch_size` for local experimentation, or use a smaller base model.

---

### 3. The Alpaca Template

```python
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
```

**Key Concepts**:
- Every SFT sample is `alpaca_template.format(instruction, output) + EOS_TOKEN`.
- The EOS token is essential: without it the model never learns where an answer stops, causing runaway generation.
- The same template string is reused at DPO time (with an empty response) and at inference, keeping train/inference formatting identical.

---

### 4. Data Preparation

```python
    if finetuning_type == "sft":
        def format_samples_sft(examples):
            text = []
            for instruction, output in zip(examples["instruction"], examples["output"], strict=False):
                message = alpaca_template.format(instruction, output) + EOS_TOKEN
                text.append(message)
            return {"text": text}

        dataset1 = load_dataset(f"{dataset_huggingface_workspace}/llmtwin", split="train")
        dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
        dataset = concatenate_datasets([dataset1, dataset2])
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")

        dataset = dataset.map(format_samples_sft, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.train_test_split(test_size=0.05)
```

**Key Concepts**:
- **Two datasets are concatenated**: your domain-specific `llmtwin` data plus 10,000 general Alpaca samples. Mixing prevents catastrophic forgetting of general instruction-following.
- **`batched=True`** formats samples in batches for speed.
- **`remove_columns=dataset.column_names`** drops the original `instruction`/`output` columns, leaving only `text`, which `SFTTrainer` expects via `dataset_text_field="text"`.
- **`is_dummy`** trims to 400 samples and 1 epoch so the full pipeline can be smoke-tested cheaply.
- **5% held out** for evaluation.

---

### 5. `SFTTrainer` Configuration

```python
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=True,
            args=TrainingArguments(
                learning_rate=learning_rate,          # default 3e-4
                num_train_epochs=num_train_epochs,    # default 3
                per_device_train_batch_size=per_device_train_batch_size,  # 2
                gradient_accumulation_steps=gradient_accumulation_steps,  # 8
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                report_to="comet_ml",
                seed=0,
            ),
        )
```

**Key Concepts**:
- **`packing=True`** concatenates short samples into full `max_seq_length` sequences, eliminating padding waste. Huge throughput win on a fixed token budget.
- **`bf16=is_bfloat16_supported()`** picks bfloat16 when the GPU supports it (Ampere+; the RTX 5000 is Turing, so this falls back to **fp16**). This matters on this workstation: Turing has no bf16, so `fp16=True`.
- **`optim="adamw_8bit"`** stores optimizer states in 8-bit, another memory saver.
- **Effective batch** = `per_device_train_batch_size × gradient_accumulation_steps` = 2 × 8 = 16.
- **`report_to="comet_ml"`** sends metrics to Comet ML (Session 7.1).
- **`seed=0`** makes runs reproducible.

---

### 6. Save, Merge, and Publish

```python
def save_model(model, tokenizer, output_dir: str, push_to_hub: bool = False, repo_id: Optional[str] = None):
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    if push_to_hub and repo_id:
        model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")
```

**Key Concepts**:
- **`merged_16bit`** merges the LoRA adapter back into the base weights and stores full-precision (fp16) weights. This produces a standalone model that vLLM/SageMaker can serve without PEFT.
- **The trade-off**: merged 16-bit is larger on disk (~16 GB for 8B) but has zero inference-time adapter overhead. Serving LoRA adapters directly is smaller but requires PEFT-aware infrastructure.

---

### 7. The `__main__` SFT Entry Point

```python
    if args.finetuning_type == "sft":
        base_model_name = "meta-llama/Llama-3.1-8B"
        output_dir_sft = Path(args.model_dir) / "output_sft"
        model, tokenizer = finetune(
            finetuning_type="sft",
            model_name=base_model_name,
            output_dir=str(output_dir_sft),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
        )
        inference(model, tokenizer)

        sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
        save_model(model, tokenizer, "model_sft", push_to_hub=True, repo_id=sft_output_model_repo_id)
```

**Key Concepts**:
- **`SM_OUTPUT_DATA_DIR`, `SM_MODEL_DIR`, `SM_NUM_GPUS`** are SageMaker-provided environment variables. The script is written to run inside a SageMaker training container.
- **`inference()` runs right after training** as a smoke test, streaming a sample generation.
- Base model is `meta-llama/Llama-3.1-8B`, output `{workspace}/TwinLlama-3.1-8B`.

---

## 🛠️ Hands-On: Prepare and Smoke-Test SFT

### Step 1: Check the dataset exists on the Hub

```python
from datasets import load_dataset
ds = load_dataset("<workspace>/llmtwin", split="train")
print(len(ds), ds.column_names)   # ['instruction', 'output']
print(ds[0])
```

### Step 2: Preview the formatted text

```python
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EX = ds[0]
print(alpaca_template.format(EX["instruction"], EX["output"]))
```

### Step 3: Launch dummy training locally

```bash
python llm_engineering/model/finetuning/finetune.py \
    --finetuning_type sft \
    --is_dummy True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 3e-4
```

Watch the loss go down and confirm a `model_sft/` directory appears.

### Step 4: Verify bf16 vs fp16 on this GPU

```python
import torch
print("bf16 supported:", torch.cuda.is_bf16_supported())  # False on Turing RTX 5000
print("fp16 supported:", torch.cuda.is_available())
```

---

## 📝 Exercise: Ablation on LoRA Rank

### Task

Measure how LoRA rank affects quality and memory.

1. Run dummy SFT with `r=8`, `r=32`, and `r=64`.
2. Record peak VRAM (`torch.cuda.max_memory_allocated()`) and final training loss.
3. Generate the same prompt from each checkpoint.
4. Decide which rank gives the best quality-per-GB tradeoff.

**Goal**: Build intuition that rank is the primary capacity knob, and that beyond a point it only increases memory.

---

## 🐛 Common Pitfalls

- **Import order**: moving `PatchDPOTrainer()` below the TRL imports silently breaks DPO patching. Keep the ordering.
- **bf16 on Turing**: `is_bfloat16_supported()` returns `False` on the RTX 5000; the code correctly falls back to fp16. Do not force bf16.
- **Missing EOS**: forgetting `+ EOS_TOKEN` produces a model that never stops.
- **Local OOM**: an 8B QLoRA run at seq 2048/batch 2 is at the edge of 16 GB. Reduce sequence length first, then batch size.
- **Packing + short samples**: packing is beneficial but can produce long sequences; keep `max_seq_length` bounded.

---

## 🎓 Knowledge Check

1. **Why is `PatchDPOTrainer()` called before the TRL imports?**
   - Answer: It monkey-patches TRL classes; the patch must be in place before TRL is imported.

2. **What does `load_in_4bit=True` do to memory and what stays trainable?**
   - Answer: It quantizes frozen base weights to 4-bit; only the LoRA adapters are trained.

3. **Why concatenate a general Alpaca dataset with the domain dataset?**
   - Answer: To preserve general instruction-following and reduce catastrophic forgetting.

4. **What is `packing=True` and why does it help?**
   - Answer: It concatenates short samples into full-length sequences, removing padding waste.

5. **Which precision is used on the RTX 5000, and why?**
   - Answer: fp16, because Turing does not support bf16.

6. **What does `save_method="merged_16bit"` produce?**
   - Answer: A standalone fp16 model with the LoRA adapter merged into the base weights.

---

## 🔗 Next Session

**Session 5.2**: Direct Preference Optimization (DPO)

We align the SFT model with the preference dataset using `DPOTrainer`.

---

## 📚 Additional Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [LoRA (Hu et al.)](https://arxiv.org/abs/2106.09685)
- [QLoRA (Dettmers et al.)](https://arxiv.org/abs/2305.14314)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Session 3.1

**Outcome**: You can configure and run a memory-efficient SFT/QLoRA job, and explain every hyperparameter in the trainer.
