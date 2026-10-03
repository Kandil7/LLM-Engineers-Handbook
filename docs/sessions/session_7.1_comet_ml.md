# Session 7.1: Experiment Tracking with Comet ML

## 🎯 Learning Objectives

By the end of this session, you will:
- Understand how the trainings report metrics to Comet ML
- Configure Comet ML locally and inside SageMaker
- Compare SFT and DPO runs side by side
- Know which metrics to watch during LLM fine-tuning
- Recognize that Opik traces and Comet experiments share one account

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      Comet ML account                             │
│                                                                    │
│  Comet Experiments (training metrics)                              │
│      ▲                    ▲                                        │
│      │ report_to          │ report_to                              │
│      │ "comet_ml"         │ "comet_ml"                              │
│  SFT TrainingArguments    DPOConfig                               │
│      ▲                    ▲                                        │
│      └────────┬───────────┘                                        │
│          COMET_API_KEY  +  COMET_PROJECT / COMET_PROJECT_NAME      │
│                                                                    │
│  Opik (prompt/LLM tracing, Session 7.2)                            │
│      ▲                                                             │
│      └── same COMET_API_KEY, project routed via OPIK_PROJECT_NAME  │
└──────────────────────────────────────────────────────────────────┘
```

One vendor, two products: **Comet ML** for experiment metrics, **Opik** for LLM/prompt tracing. Both authenticate with `COMET_API_KEY`.

---

## 📁 Key Files Explained

### 1. `settings.py` - Comet Configuration

```python
# llm_engineering/settings.py
    # Comet ML (during training)
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "twin"
```

**Key Concepts**:
- **`COMET_API_KEY`** is required to log anywhere.
- **`COMET_PROJECT`** defaults to `twin`. In training, the trainer uses this as the Comet project; in the SageMaker job it is passed as `COMET_PROJECT_NAME` (the env var name Comet's HF integration reads).

---

### 2. SFT Reporting

```python
# llm_engineering/model/finetuning/finetune.py (SFT TrainingArguments)
            args=TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                warmup_steps=10,
                output_dir=output_dir,
                report_to="comet_ml",
                seed=0,
            ),
```

**Key Concepts**:
- **`report_to="comet_ml"`** is the single switch. Hugging Face `Trainer` detects the Comet integration and starts an experiment automatically.
- **`logging_steps=1`** logs every step; useful for small/dummy runs and for spotting divergence early.
- **`output_dir`** is the local checkpoint directory; it is also logged as an artifact location.
- All `TrainingArguments` fields are auto-logged as hyperparameters, so the experiment dashboard shows the full config.

---

### 3. DPO Reporting

```python
# finetune.py (DPOConfig)
                args=DPOConfig(
                    learning_rate=learning_rate,
                    num_train_epochs=num_train_epochs,
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
```

**Key Concepts**:
- Same `report_to="comet_ml"`, so SFT and DPO runs land in the **same project** and can be overlaid.
- **`eval_steps=0.2`** produces 5 evaluation points per epoch, which gives enough resolution to watch the preference loss fall.

**DPO-specific metrics to watch**:
| Metric | Meaning | Healthy signal |
|--------|---------|----------------|
| `loss` | DPO objective | decreasing |
| `rewards/chosen` | log-prob advantage of chosen | rising |
| `rewards/rejected` | log-prob advantage of rejected | falling |
| `rewards/accuracies` | fraction where chosen outranks rejected | rising toward 1 |
| `rewards/margins` | chosen − rejected gap | rising |
| `logps/chosen` / `logps/rejected` | raw log-probs | diverge from reference |

---

### 4. SageMaker Environment Pass-Through

```python
# llm_engineering/model/finetuning/sagemaker.py
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
```

**Key Concepts**:
- **`COMET_PROJECT_NAME`** (not `COMET_PROJECT`) is the variable Comet's Hugging Face integration reads inside the training container.
- **`COMET_API_KEY`** is injected as an environment variable so logs flow from the remote job back to your account.
- Locally, Comet reads `COMET_API_KEY` and `COMET_PROJECT` from `.env` directly.

---

## 🛠️ Hands-On: Run a Tracked Experiment

### Step 1: Configure the environment

```env
COMET_API_KEY=your-comet-key
COMET_PROJECT=twin
```

### Step 2: Run a dummy SFT experiment

```bash
python llm_engineering/model/finetuning/finetune.py --finetuning_type sft --is_dummy True
```

### Step 3: Open the Comet dashboard

Go to `https://www.comet.com/` and open the `twin` project. You should see:
- A run named after the script/session.
- **Charts**: `loss`, `learning_rate`, and for DPO also `rewards/*`.
- **Hyperparameters**: every `TrainingArguments` field.
- **System metrics**: GPU utilization and memory (if enabled).

### Step 4: Compare runs

Select the SFT and DPO runs and overlay the `loss` chart. Note that the DPO loss is on a different scale and only runs one epoch.

---

## 📝 Exercise: Add a Custom Metric Callback

### Task

Log the average training loss manually at the end, plus a tag.

```python
from transformers import TrainerCallback
import comet_ml  # noqa: F401

class CometSummaryCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        experiment = comet_ml.get_global_experiment()
        if experiment:
            experiment.log_metric("final_train_loss", state.log_history[-1].get("loss"))
            experiment.add_tag("twin-finetune")
```

1. Pass `callbacks=[CometSummaryCallback()]` to `SFTTrainer`.
2. Re-run a dummy job and confirm `final_train_loss` and the tag appear.

**Goal**: Learn the Comet API for custom metrics and tags beyond what the HF integration logs automatically.

---

## 🐛 Common Pitfalls

- **Wrong env var in SageMaker**: using `COMET_PROJECT` instead of `COMET_PROJECT_NAME` in the container means runs land in the default project. The code correctly uses `COMET_PROJECT_NAME`.
- **Missing key**: without `COMET_API_KEY`, `report_to="comet_ml"` logs a warning and training continues untracked.
- **Dummy vs real scale**: dummy runs (1 epoch, 400 samples) are for plumbing checks, not for comparing model quality.
- **Comparing incompatible runs**: SFT and DPO have different metric spaces; compare `rewards/*` within DPO and use the Session 7.3 evaluation for cross-model quality.

---

## 🎓 Knowledge Check

1. **What single argument enables Comet logging in the trainers?**
   - Answer: `report_to="comet_ml"`.

2. **Which env var name does the SageMaker container need for the project?**
   - Answer: `COMET_PROJECT_NAME`.

3. **Which DPO metric tells you the chosen answer is being preferred more often?**
   - Answer: `rewards/accuracies` (and `rewards/margins`).

4. **What does `logging_steps=1` do?**
   - Answer: Logs a point every training step.

5. **Do Opik and Comet experiments use different credentials?**
   - Answer: No, both use `COMET_API_KEY`.

6. **Why can SFT and DPO runs be compared directly in Comet?**
   - Answer: Both report to the same project, so their charts can be overlaid.

---

## 🔗 Next Session

**Session 7.2**: Prompt Monitoring with Opik

We inspect the traces generated by the RAG inference path.

---

## 📚 Additional Resources

- [Comet ML Documentation](https://www.comet.com/docs/v2/)
- [Hugging Face Trainer integrations](https://huggingface.co/docs/transformers/main_classes/trainer#integrations)
- [TRL DPO metrics](https://huggingface.co/docs/trl/dpo_trainer)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Sessions 5.1, 5.2

**Outcome**: You can configure Comet ML, interpret SFT/DPO training charts, and extend logging with custom metrics.
