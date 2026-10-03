# Session 9.3: Security Best Practices

## 🎯 Learning Objectives

By the end of this session, you will:
- Inventory every secret and where it must (and must not) live
- Recognize the security gaps in the current API and local stack
- Apply secret scanning, least-privilege, and input validation
- Understand prompt-injection risk in RAG context
- Produce a prioritized hardening plan

> This session is defensive: it audits this codebase and shows how to reduce risk. It is not a penetration-testing guide.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Trust boundaries                                │
│                                                                        │
│  Secrets:  .env (local)  ──►  ZenML secret store (remote)             │
│            SageMaker job: environment vars (NOT hyperparameters)      │
│            CI: GitHub repository secrets                              │
│                                                                        │
│  API:      internet ──► /rag (no auth, no rate limit) ──► AWS         │
│                                                                        │
│  Data:     MongoDB (default creds)   Qdrant (no auth locally)         │
│                                                                        │
│  Supply chain: poetry.lock pinned  +  gitleaks in CI                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Secret Management

### 1. Never commit secrets

```bash
# .env.example - template only, values are placeholders
OPENAI_API_KEY=str
HUGGINGFACE_ACCESS_TOKEN=str
COMET_API_KEY=str
DATABASE_HOST="mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
AWS_ACCESS_KEY=str
AWS_SECRET_KEY=str
```

- **`.env.example` holds placeholders**, not real values.
- **`.env` is gitignored** (verify in `.gitignore`) and is the local secret store.
- **`gitleaks-check` runs in CI** and scans `llm_engineering/` for committed secrets. This is the mechanical backstop.

**Rule:** if a secret ever reaches a commit, **rotate it immediately**. Removing the file does not remove it from history.

### 2. Remote runs use the ZenML secret store

```python
# settings.py
    @classmethod
    def load_settings(cls) -> "Settings":
        try:
            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError):
            settings = Settings()   # falls back to .env / defaults
        return settings

    def export(self) -> None:
        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)
        client = Client()
        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning("Secret 'scope' already exists. ...")
```

- **`export()` pushes all settings to ZenML** so remote steps read secrets from the platform, not a shipped `.env`.
- **`load_settings()` degrades gracefully** to `.env`/defaults, which is convenient locally but means a misconfigured remote run could silently use weak defaults. Verify the "Loading settings from the ZenML secret store" log line in production.

### 3. Training/evaluation containers

```python
# sagemaker.py
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
        hyperparameters=hyperparameters,
```

- **Secrets go in `environment`, never `hyperparameters`.** Hyperparameters are logged and visible in the AWS console; environment variables are not surfaced the same way.
- This is a deliberate, correct choice in the code.

---

## 📁 API Security Gaps

### 1. No authentication or rate limiting

```python
@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Gaps**:
- **No auth**: anyone who can reach the service can invoke the LLM, incurring cost.
- **No rate limiting**: an unauthenticated client can exhaust your OpenAI/SageMaker budget.
- **`detail=str(e)`** leaks exception messages (and potentially stack-derived internals) to clients.

**Hardening**:

```python
from fastapi import Depends, Header, HTTPException

API_KEY = settings.COMET_API_KEY  # or a dedicated service key

async def require_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest, _=Depends(require_key)):
    try:
        answer = rag(query=request.query)
        return {"answer": answer}
    except Exception:
        logger.exception("RAG failed")          # log full detail server-side
        raise HTTPException(status_code=500, detail="Internal error") from None
```

Add `slowapi` or an API gateway for rate limiting.

### 2. Binding and transport

```python
uvicorn.run("tools.ml_service:app", host="0.0.0.0", port=8000, reload=True)
```

- **`0.0.0.0`** exposes the service on all interfaces. On an untrusted network, this is reachable by anyone on the LAN.
- **No TLS**: the API is plain HTTP. Terminate TLS at a reverse proxy or gateway.
- **`reload=True`** must be off in production.

### 3. Input validation

- **`QueryRequest` validates shape** (`query: str`), but not size. An enormous `query` inflates retrieval and generation cost. Add a max length:

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
```

- **No CORS policy** is configured; add `CORSMiddleware` with an explicit allow-list if a browser client is added.

---

## 📁 Data-Layer Security

### 1. Local databases have no auth protection beyond defaults

```yaml
# docker-compose.yml
    environment:
      MONGO_INITDB_ROOT_USERNAME: "llm_engineering"
      MONGO_INITDB_ROOT_PASSWORD: "llm_engineering"
```

- **Default credentials are public knowledge.** Acceptable only on an isolated dev host. Never expose ports 27017/6333 to a public network.
- **Qdrant has no authentication locally.** Do not bind it publicly.
- **Production**: put both behind a private network, use strong unique credentials, and enable TLS. Qdrant Cloud (via `USE_QDRANT_CLOUD`) provides managed auth.

### 2. AWS least privilege

`create_execution_role.py` attaches **four `*FullAccess` managed policies**:

```python
        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
            "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        ]
```

**Risk**: the role can touch far more than it needs. **Recommendation**: replace with scoped policies limited to the specific S3 buckets, ECR repositories, and SageMaker resources the pipeline uses. This is a textbook least-privilege improvement.

`create_sagemaker_role.py` is even broader (adds `IAMFullAccess` and `AWSCloudFormationFullAccess`) and creates a long-lived **IAM user with an access key**. Prefer the execution-role pattern; avoid static user keys where possible (use roles/OIDC).

---

## 📁 RAG-Specific Security: Prompt Injection

Retrieved documents are inserted into the generation prompt:

```python
        self.prompt = """
You are a content creator. ... 
User query: {query}
Context: {context}
            """
        ...
        inputs=self.prompt.format(query=self.query, context=self.context),
```

**Risks**:
- **Prompt injection**: if crawled content contains instructions like "ignore previous instructions", the model may follow them. The corpus is untrusted (arbitrary websites).
- **`str.format` fragility**: content containing `{` or `}` raises `KeyError`/`ValueError`, a denial-of-service vector from crafted content.
- **Data exfiltration**: a malicious document could attempt to steer the model into revealing other context.

**Mitigations**:
- Treat retrieved content as data, not instructions. Use a template engine that does not interpret braces (or escape `{`/`}` in context).
- Delimit context clearly and instruct the model to use it only as reference material.
- Consider a guardrail/filter on inputs and outputs.

---

## 📁 Observability and PII

Opik traces store full queries, retrieved context, and answers:

```python
    opik_context.update_current_trace(tags=["rag"], metadata={..., "query_tokens": ..., ...})
```

**Risk**: personal data and proprietary content end up in a third-party trace store.

**Mitigations**:
- Use local Opik (`use_local=True`) for sensitive corpora.
- Redact or hash PII before it enters the trace.
- Set a retention policy in Opik.

---

## 📁 Supply Chain and Container Hygiene

- **`poetry.lock` pins dependencies**, and CI installs from it - good reproducibility.
- **`gitleaks` in CI** blocks committed secrets.
- **Docker runs as root**: the `Dockerfile` never creates a non-root `USER`. Add one for defense in depth:

```dockerfile
RUN useradd -m appuser
USER appuser
```

- **Base image**: `python:3.11-slim-bullseye` is fine; pin a digest for reproducibility and scan with Trivy/Dependabot.

---

## 🛠️ Hands-On: Audit and Harden

### Step 1: Confirm `.env` is ignored

```bash
git check-ignore -v .env
# should print a .gitignore rule
```

### Step 2: Run the secret scan locally

```bash
poetry poe gitleaks-check
```

### Step 3: Grep for risky patterns

```bash
git grep -n "str(e)" -- llm_engineering
git grep -n "0.0.0.0" -- tools
git grep -n "FullAccess" -- llm_engineering
```

### Step 4: Add API-key auth and a query length limit

Apply the snippets above, then verify:

```bash
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d "{\"query\":\"hi\"}"
# 401 without the key
curl -X POST http://localhost:8000/rag -H "x-api-key: <key>" -H "Content-Type: application/json" -d "{\"query\":\"hi\"}"
# 200 with the key
```

---

## 📝 Exercise: Write a Hardening Plan

### Task

Produce a prioritized remediation table for this repository.

| Priority | Finding | Impact | Mitigation |
|----------|---------|--------|------------|
| P0 | Unauthenticated `/rag` | Cost abuse | API key / gateway auth |
| P0 | Default DB credentials exposed | Data compromise | Strong creds + private network |
| P1 | `detail=str(e)` leaks internals | Info disclosure | Log server-side, generic message |
| P1 | `*FullAccess` IAM policies | Over-privilege | Scoped policies |
| P1 | Prompt injection via context | Output manipulation | Treat context as data, escape braces |
| P2 | `0.0.0.0` + no TLS | Network exposure | Reverse proxy + TLS |
| P2 | Root container | Lateral movement | Non-root `USER` |
| P2 | PII in Opik traces | Privacy | Local Opik / redaction |

Add any findings you discover yourself, then order by exploitability × impact.

**Goal**: Practice turning an audit into an actionable, prioritized plan.

---

## 🐛 Common Pitfalls

- **Deleting a leaked secret without rotating it**: the value remains in Git history. Rotate first.
- **Trusting the corpus**: crawled content is untrusted input. Never let it define instructions unescaped.
- **Assuming local defaults are safe**: `llm_engineering/llm_engineering` in Compose is a dev convenience, not a production credential.
- **Over-broad IAM for speed**: the `*FullAccess` policies are convenient but undermine least privilege.
- **Ignoring trace PII**: observability stores real user data; treat it as sensitive.

---

## 🎓 Knowledge Check

1. **Why do secrets go in `environment` rather than `hyperparameters` for SageMaker?**
   - Answer: Hyperparameters are logged and visible; environment variables are not surfaced the same way.

2. **What is the API's biggest security gap?**
   - Answer: No authentication or rate limiting on `/rag`.

3. **Why is `detail=str(e)` risky?**
   - Answer: It can leak internal exception details to clients.

4. **What is prompt injection in this project?**
   - Answer: Untrusted crawled content in the context steering the model's behavior.

5. **Why are the `*FullAccess` IAM policies a problem?**
   - Answer: They grant far more permission than the pipeline needs, violating least privilege.

6. **What is the first action after a committed secret is found?**
   - Answer: Rotate the secret; removing the file is not enough.

---

## 🔚 Course Complete

You have worked through all nine modules:

1. Foundations and DDD
2. Data engineering
3. Dataset generation
4. Advanced RAG
5. Training and fine-tuning
6. Inference and APIs
7. Monitoring and evaluation
8. Production deployment
9. Advanced topics

The capstone is to run `end_to_end_data`, train SFT then DPO, deploy the endpoint, and evaluate — then harden the system using this session's plan.

---

## 📚 Additional Resources

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [AWS IAM Least Privilege](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Gitleaks](https://github.com/gitleaks/gitleaks)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

---

**Estimated Time**: 4-5 hours

**Prerequisites**: Sessions 6.1, 6.2, 8.2

**Outcome**: You can audit the project's trust boundaries and produce a prioritized hardening plan.
