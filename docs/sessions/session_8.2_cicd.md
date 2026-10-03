# Session 8.2: CI/CD with GitHub Actions

## 🎯 Learning Objectives

By the end of this session, you will:
- Read the CI and CD GitHub Actions workflows
- Understand the quality gates: secret scan, lint, format, tests
- Map each workflow step to its Poe task
- Configure the required repository secrets
- Know what CD builds and where it pushes

---

## 🏗️ Architecture Overview

```
Pull request                          Push to main
     │                                     │
     ▼                                     ▼
┌──────────────────────┐         ┌──────────────────────────┐
│ CI  (ci.yaml)         │         │ CD  (cd.yaml)            │
│                       │         │                          │
│ Job: QA               │         │ Job: Build & Push        │
│  ├─ gitleaks-check    │         │  ├─ checkout             │
│  ├─ ruff check        │         │  ├─ setup buildx         │
│  └─ ruff format       │         │  ├─ configure AWS creds  │
│                       │         │  ├─ login to ECR         │
│ Job: Test             │         │  └─ build amd64 image    │
│  └─ pytest            │         │       push :sha + :latest│
└──────────────────────┘         └──────────────────────────┘
```

The pattern: **CI gates merges; CD publishes immutable images** tagged by commit SHA plus a moving `latest`.

---

## 📁 Key Files Explained

### 1. `.github/workflows/ci.yaml` - Continuous Integration

```yaml
name: CI

on:
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  qa:
    name: QA
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.11"

      - name: Install poetry
        uses: abatilo/actions-poetry@v2
        with:
          poetry-version: 1.8.3

      - name: Install packages
        run: |
          poetry install --only dev
          poetry self add 'poethepoet[poetry_plugin]'

      - name: gitleaks check
        run: poetry poe gitleaks-check

      - name: Lint check [Python]
        run: poetry poe lint-check

      - name: Format check [Python]
        run: poetry poe format-check

  test:
    name: Test
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.11"

      - name: Install poetry
        uses: abatilo/actions-poetry@v2
        with:
          poetry-version: 1.8.3

      - name: Install packages
        run: |
          poetry install
          poetry self add 'poethepoet[poetry_plugin]'

      - name: Run tests
        run: |
          echo "Running tests..."
          poetry poe test
```

**Key Concepts**:
- **Two independent jobs**: `qa` and `test` run in parallel. Both must pass for the PR to be mergeable (when branch protection requires them).
- **`qa` installs only dev dependencies** (`poetry install --only dev`) - lint/format/secret-scan need no runtime packages, so it is fast.
- **`test` installs everything** because the tests import the application.
- **`concurrency` with `cancel-in-progress`** supersedes in-flight runs when new commits are pushed, saving CI minutes.
- **`poetry poe <task>`** means CI and local development run the *same* commands - the Poe task is the single source of truth.

### The Poe task definitions behind each step

```toml
# pyproject.toml
lint-check = "poetry run ruff check ."
format-check = "poetry run ruff format --check ."
gitleaks-check = "docker run -v .:/src zricethezav/gitleaks:latest dir /src/llm_engineering"

[tool.poe.tasks.test]
cmd = "poetry run pytest tests/"
env = { ENV_FILE = ".env.testing" }
```

**Key Concepts**:
- **`lint-check`** runs `ruff check` (flake8/import-sort/bugbear rules).
- **`format-check`** runs `ruff format --check`, which fails if any file is not formatted. Never let CI format code; it only verifies.
- **`gitleaks-check`** mounts the repo into a Gitleaks container and scans `llm_engineering/` for committed secrets. This is the security gate.
- **`test`** runs `pytest tests/` with `ENV_FILE=.env.testing`, so tests use a dedicated env file rather than production `.env`.

### The linter configuration

```toml
# ruff.toml
line-length = 120
target-version = "py311"
extend-exclude = [".github", "graphql_client", "graphql_schemas"]

[lint]
extend-select = ["I", "B", "G", "T20", "PTH", "RUF"]

[lint.isort]
case-sensitive = true

[lint.pydocstyle]
convention = "google"
```

- **`I`** import sorting, **`B`** bugbear, **`G`** logging format, **`T20`** bans `print`, **`PTH`** prefers `pathlib`, **`RUF`** Ruff-native rules.
- **`T20` is why the scripts use `print(...)  # noqa`** in `finetune.py` and `evaluate.py`: those prints are intentional and explicitly exempted.

---

### 2. `.github/workflows/cd.yaml` - Continuous Delivery

```yaml
name: CD

on:
  push:
    branches:
      - main

concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: true

jobs:
  build:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build images & push to ECR
        id: build-image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          tags: |
            ${{ steps.login-ecr.outputs.registry }}/${{ secrets.AWS_ECR_NAME }}:${{ github.sha }}
            ${{ steps.login-ecr.outputs.registry }}/${{ secrets.AWS_ECR_NAME }}:latest
          push: true
```

**Key Concepts**:
- **Triggered on push to `main`** only, so images are published after merge, never from a PR.
- **`configure-aws-credentials`** assumes the repository secrets to get ECR access.
- **`amazon-ecr-login`** authenticates Docker to the account's registry and exposes the registry URL as an output.
- **Two tags**: `:${{ github.sha }}` (immutable, traceable to a commit) and `:latest` (convenience). Pin deployments to the SHA tag for reproducibility.
- **`push: true`** builds and pushes in one step via Buildx.

### Required repository secrets

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | ECR auth |
| `AWS_SECRET_ACCESS_KEY` | ECR auth |
| `AWS_REGION` | ECR region |
| `AWS_ECR_NAME` | Target repository name |

Set these under **Settings → Secrets and variables → Actions**. Never echo them in logs.

---

## 🔬 Deep Dive: The Gate Philosophy

```
       local                      CI                         CD
    ┌─────────┐            ┌──────────────┐           ┌──────────────┐
    │ ruff    │            │ gitleaks     │           │ build amd64  │
    │ pytest  │  ──PR──►   │ ruff (check) │  ──merge──►│ push to ECR  │
    │ poe *   │            │ pytest       │           │ tag sha+latest│
    └─────────┘            └──────────────┘           └──────────────┘
```

- **Fast feedback first**: `qa` (seconds) before `test` (slower). Developers fix lint before waiting on the suite.
- **Secrets before code**: `gitleaks-check` runs first in `qa`, catching a leaked key before anything else.
- **Format is enforced, not applied**: `--check` fails; the developer runs `poetry poe format-fix` locally.
- **Immutable artifacts**: the CD image is addressed by commit SHA, so any deploy can be traced back to exact code.

---

## 🛠️ Hands-On: Reproduce CI Locally

### Step 1: Install dev tooling

```bash
poetry install --only dev
poetry self add 'poethepoet[poetry_plugin]'
```

### Step 2: Run each gate

```bash
poetry poe gitleaks-check
poetry poe lint-check
poetry poe format-check
poetry poe test
```

### Step 3: Fix what fails

```bash
poetry poe lint-fix
poetry poe format-fix
```

### Step 4: Simulate CD with docker

```bash
docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile .
```

---

## 📝 Exercise: Add a Type-Check Job

### Task

Add a `pyright` (or `mypy`) type check to CI.

1. Add `pyright` to the dev dependencies.
2. Add a Poe task `type-check = "poetry run pyright llm_engineering"`.
3. Add a step to the `qa` job: `- name: Type check [Python]` → `run: poetry poe type-check`.
4. Run it locally on the current code and note any existing type errors.

**Goal**: Experience adding a new gate to both local and CI in sync. (Expect some errors - the codebase is not fully typed.)

---

## 🐛 Common Pitfalls

- **`gitleaks-check` mounts the whole repo**: it scans `llm_engineering/`, so a secret anywhere in that tree fails the build. Rotate the secret and purge it from history if leaked; deleting the file is not enough.
- **Format drift**: developers who do not run `format-fix` fail CI. Pre-commit hooks (`.pre-commit-config.yaml`) reduce this.
- **Deprecated action versions**: `actions/checkout@v3` and `setup-python@v3` are older; new repos should use `@v4`/`@v5`. Upgrade when convenient.
- **`latest` tag races**: two rapid merges can move `latest` twice. Deploy by SHA tag.
- **Secrets in forks**: PRs from forks do not receive secrets, so `configure-aws-credentials` steps fail there. Keep AWS steps only in CD.

---

## 🎓 Knowledge Check

1. **What triggers CI vs CD?**
   - Answer: CI on pull requests; CD on pushes to `main`.

2. **Which four gates run in the pipeline?**
   - Answer: gitleaks (secret scan), ruff lint, ruff format check, pytest.

3. **Why does the `qa` job install only dev dependencies?**
   - Answer: Lint/format/secret-scan do not need runtime packages, so it is faster.

4. **What does `format-check` do versus `format-fix`?**
   - Answer: `--check` fails CI if unformatted; `format-fix` rewrites files locally.

5. **What two tags does CD publish?**
   - Answer: the commit SHA and `latest`.

6. **Why is the image built for `linux/amd64`?**
   - Answer: SageMaker and most cloud runners are amd64; ARM builds would not run there.

---

## 🔗 Next Session

**Session 8.3**: ZenML Orchestration

We connect all the pipelines into a reproducible DAG with artifacts and caching.

---

## 📚 Additional Resources

- [GitHub Actions](https://docs.github.com/en/actions)
- [Ruff](https://docs.astral.sh/ruff/)
- [Gitleaks](https://github.com/gitleaks/gitleaks)
- [AWS ECR Login Action](https://github.com/aws-actions/amazon-ecr-login)

---

**Estimated Time**: 3-4 hours

**Prerequisites**: Session 8.1

**Outcome**: You can read, reproduce, and extend the CI/CD workflows and their quality gates.
