# Session 5.3: AWS SageMaker Deployment

## 🎯 Learning Objectives

By the end of this session, you will:
- Create the IAM roles SageMaker needs
- Launch a training job on a managed GPU instance
- Deploy a Hugging Face LLM to a real-time endpoint
- Configure inference via environment variables (quantization, token limits)
- Add target-tracking autoscaling and clean up the endpoint
- Run the deployment script for a dry tooling check and clean up safely

> ⚠️ **Cost warning**: `ml.g5.2xlarge` and larger instances bill by the second. Do not leave an endpoint running. Always run the delete script when finished.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AWS account                                   │
│                                                                        │
│  IAM roles (create_execution_role.py / create_sagemaker_role.py)      │
│        │                                                               │
│        ▼                                                               │
│  SageMaker Training Job (model/finetuning/sagemaker.py)               │
│        │  HuggingFace estimator, entry_point=finetune.py             │
│        │  instance ml.g5.2xlarge, role=AWS_ARN_ROLE                  │
│        ▼                                                               │
│  Model on Hugging Face Hub  {workspace}/TwinLlama-3.1-8B-DPO         │
│        │                                                               │
│        ▼                                                               │
│  SageMaker Endpoint (deploy/huggingface/run.py)                       │
│        │  HuggingFaceModel(image=get_huggingface_llm_image_uri)      │
│        │  env: quantize=bitsandbytes, max tokens, SM_NUM_GPUS        │
│        ▼                                                               │
│  Autoscaling (autoscaling_sagemaker_endpoint.py)                      │
│        └─ target tracking on InvocationsPerCopy                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Strategy Pattern for Deployment

```
DeploymentStrategy (domain/inference.py, ABC)
        ▲
        │ implements
SagemakerHuggingfaceStrategy  ──delegates──►  DeploymentService
   (public API)                                 (boto3 + HuggingFaceModel)
```

The strategy is swappable: a different cloud could implement `DeploymentStrategy` without touching the deploy script.

---

## 📁 Key Files Explained

### 1. `infrastructure/aws/roles/create_execution_role.py` - Execution Role

**Purpose**: The role SageMaker assumes to run jobs and read S3/ECR.

```python
# llm_engineering/infrastructure/aws/roles/create_execution_role.py
def create_sagemaker_execution_role(role_name: str):
    assert settings.AWS_REGION, "AWS_REGION is not set."
    assert settings.AWS_ACCESS_KEY, "AWS_ACCESS_KEY is not set."
    assert settings.AWS_SECRET_KEY, "AWS_SECRET_KEY is not set."

    iam = boto3.client(
        "iam",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )

    trust_relationship = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}
        ],
    }

    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_relationship),
            Description="Execution role for SageMaker",
        )

        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
            "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        ]
        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)

        logger.info(f"Role '{role_name}' created successfully.")
        logger.info(f"Role ARN: {role['Role']['Arn']}")
        return role["Role"]["Arn"]

    except iam.exceptions.EntityAlreadyExistsException:
        logger.warning(f"Role '{role_name}' already exists. Fetching its ARN...")
        role = iam.get_role(RoleName=role_name)
        return role["Role"]["Arn"]
```

**Key Concepts**:
- **Trust relationship** lets `sagemaker.amazonaws.com` assume the role (`sts:AssumeRole`).
- **Four managed policies**: SageMaker, S3 (training data/models), CloudWatch Logs, and ECR (pull the Docker image).
- **Idempotent**: on `EntityAlreadyExistsException`, it fetches the existing ARN instead of failing.
- **The ARN is written to `sagemaker_execution_role.json`** and copied into `.env` as `AWS_ARN_ROLE`.

`create_sagemaker_role.py` is the alternative approach: it creates a full IAM **user** (`sagemaker-deployer`) with an access key. Prefer the execution role for deployments; the user is for a personal deployer identity.

---

### 2. `model/finetuning/sagemaker.py` - Training Job

**Purpose**: Submit `finetune.py` as a SageMaker training job.

```python
# llm_engineering/model/finetuning/sagemaker.py
def run_finetuning_on_sagemaker(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."

    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]

    hyperparameters = {
        "finetuning_type": finetuning_type,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "dataset_huggingface_workspace": dataset_huggingface_workspace,
        "model_output_huggingface_workspace": huggingface_user,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True

    huggingface_estimator = HuggingFace(
        entry_point="finetune.py",
        source_dir=str(finetuning_dir),
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        role=settings.AWS_ARN_ROLE,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        requirements_file=finetuning_requirements_path,
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
    )

    huggingface_estimator.fit()
```

**Key Concepts**:
- **`entry_point="finetune.py"` + `source_dir`**: SageMaker packages the finetuning folder and runs the script in a managed container.
- **`hyperparameters`** are passed as CLI flags to `finetune.py`; this is why the script uses `argparse`.
- **`environment`** injects secrets (HF token, Comet key) into the container as env vars - never as hyperparameters, which are logged.
- **`requirements_file`** installs `unsloth`, `trl`, and the rest at container startup.
- **`transformers_version`/`pytorch_version`/`py_version`** pin the base image; Unsloth may require newer versions than the SageMaker-managed image provides. If the job fails on Unsloth install, use a custom container image.

---

### 3. `deploy/huggingface/run.py` - Create the Endpoint

```python
# llm_engineering/infrastructure/aws/deploy/huggingface/run.py
def create_endpoint(endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED) -> None:
    assert settings.AWS_ARN_ROLE is not None, "AWS_ARN_ROLE is not set in the .env file."

    logger.info(f"Creating endpoint with endpoint_type = {endpoint_type} and model_id = {settings.HF_MODEL_ID}")

    llm_image = get_huggingface_llm_image_uri("huggingface", version="2.2.0")

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=settings.AWS_ARN_ROLE,
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        endpoint_config_name=settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
        gpu_instance_type=settings.GPU_INSTANCE_TYPE,
        resources=model_resource_config,
        endpoint_type=endpoint_type,
    )
```

**Key Concepts**:
- **`get_huggingface_llm_image_uri("huggingface", version="2.2.0")`** returns the AWS-owned Deep Learning Container (DLC) purpose-built for Hugging Face LLM inference. It includes the serving stack, so you do not build an image.
- **Two endpoint types**:
  - `MODEL_BASED`: one model per endpoint.
  - `INFERENCE_COMPONENT_BASED`: supports multiple models and per-component autoscaling. Required for the autoscaling script below.

---

### 4. `deploy/huggingface/config.py` - Runtime Environment

```python
# llm_engineering/infrastructure/aws/deploy/huggingface/config.py
hugging_face_deploy_config = {
    "HF_MODEL_ID": settings.HF_MODEL_ID,
    "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
    "SM_NUM_GPUS": json.dumps(settings.SM_NUM_GPUS),
    "MAX_INPUT_LENGTH": json.dumps(settings.MAX_INPUT_LENGTH),
    "MAX_TOTAL_TOKENS": json.dumps(settings.MAX_TOTAL_TOKENS),
    "MAX_BATCH_TOTAL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "MAX_BATCH_PREFILL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "HF_MODEL_QUANTIZE": "bitsandbytes",
}

model_resource_config = ResourceRequirements(
    requests={
        "copies": settings.COPIES,
        "num_accelerators": settings.GPUS,
        "num_cpus": settings.CPUS,
        "memory": 5 * 1024,
    },
)
```

**Key Concepts**:
- **`HF_MODEL_ID`** points at the merged model on the Hub (for example `{workspace}/TwinLlama-3.1-8B-DPO`).
- **`HF_MODEL_QUANTIZE="bitsandbytes"`** loads the model in 8-bit on the endpoint, cutting VRAM enough to fit within `ml.g5.2xlarge` (24 GB).
- **`MAX_INPUT_LENGTH` / `MAX_TOTAL_TOKENS` / `MAX_BATCH_TOTAL_TOKENS`** map directly onto TGI (Text Generation Inference) scheduling limits. `MAX_BATCH_PREFILL_TOKENS` must equal `MAX_BATCH_TOTAL_TOKENS` in this config.
- **`model_resource_config`** declares per-replica resources used with inference components.

---

### 5. `sagemaker_huggingface.py` - Strategy and Service

```python
# llm_engineering/infrastructure/aws/deploy/huggingface/sagemaker_huggingface.py
class SagemakerHuggingfaceStrategy(DeploymentStrategy):
    def __init__(self, deployment_service) -> None:
        self.deployment_service = deployment_service

    def deploy(self, role_arn, llm_image, config, endpoint_name, endpoint_config_name,
               gpu_instance_type, resources=None, endpoint_type=EndpointType.MODEL_BASED) -> None:
        self.deployment_service.deploy(
            role_arn=role_arn, llm_image=llm_image, config=config,
            endpoint_name=endpoint_name, endpoint_config_name=endpoint_config_name,
            gpu_instance_type=gpu_instance_type, resources=resources, endpoint_type=endpoint_type,
        )


class DeploymentService:
    def __init__(self, resource_manager):
        self.sagemaker_client = boto3.client("sagemaker", region_name=settings.AWS_REGION, ...)
        self.resource_manager = resource_manager

    @staticmethod
    def prepare_and_deploy_model(role_arn, llm_image, config, endpoint_name, update_endpoint,
                                 gpu_instance_type, resources=None, endpoint_type=EndpointType.MODEL_BASED) -> None:
        huggingface_model = HuggingFaceModel(role=role_arn, image_uri=llm_image, env=config)

        huggingface_model.deploy(
            instance_type=gpu_instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            update_endpoint=update_endpoint,
            resources=resources,
            tags=[{"Key": "task", "Value": "model_task"}],
            endpoint_type=endpoint_type,
            container_startup_health_check_timeout=900,
        )
```

**Key Concepts**:
- **`container_startup_health_check_timeout=900`** allows 15 minutes for the container to download the model and start TGI. LLM endpoints are slow to boot; the default timeout is too short.
- **`ResourceManager.endpoint_config_exists` / `endpoint_exists`** check state with boto3 before deploying, so the flow is idempotent.
- **`SagemakerHuggingfaceStrategy` implements the `DeploymentStrategy` ABC** from `domain/inference.py`, keeping the domain contract independent of boto3.

---

### 6. Autoscaling and Teardown

```python
# llm_engineering/infrastructure/aws/deploy/autoscaling_sagemaker_endpoint.py
class AutoscalingSagemakerEndpoint:
    def __init__(self, auto_scaling_client, inference_component_name, endpoint_name,
                 initial_copy_count: int = 1, max_copy_count: int = 6, target_value: float = 4.0):
        ...
        self.scalable_dimension = "sagemaker:inference-component:DesiredCopyCount"
        self.resource_id = f"inference-component/{self.inference_component_name}"

    def setup_autoscaling(self):
        scalable_target = ScalableTarget(...)
        scalable_target.register()
        policy = TargetTrackingScalingPolicy(..., target_value=self.target_value + 1,
                                             scale_in_cooldown=200, scale_out_cooldown=200)
        policy.apply_policy()
```

- **Target tracking** on `SageMakerInferenceComponentInvocationsPerCopy` scales the number of copies to hold a target load per copy.
- **Cooldowns (200s)** prevent flapping.
- **Only works with `INFERENCE_COMPONENT_BASED` endpoints**; model-based endpoints have no inference component to scale.

```python
# delete_sagemaker_endpoint.py
def delete_endpoint_and_config(endpoint_name) -> None:
    # describe_endpoint → config_name
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
    sagemaker_client.delete_model(ModelName=model_name)
```

Deletes the endpoint, then its config, then the model. Run this to stop billing.

---

## 🛠️ Hands-On: Deploy and Call the Endpoint

### Step 1: One-time IAM setup

```bash
python -m llm_engineering.infrastructure.aws.roles.create_execution_role
# copy the printed ARN into .env as AWS_ARN_ROLE
```

### Step 2: Create the endpoint

```bash
python -m llm_engineering.infrastructure.aws.deploy.huggingface.run
```

This takes 10–15 minutes (model download + container boot).

### Step 3: Call the endpoint

```python
import json
import boto3
from llm_engineering.settings import settings

client = boto3.client("sagemaker-runtime", region_name=settings.AWS_REGION)
payload = {"inputs": "Below is an instruction...\n### Instruction:\nWhat is an LLM Twin?\n### Response:\n",
           "parameters": {"max_new_tokens": 150, "temperature": 0.01, "top_p": 0.9}}

response = client.invoke_endpoint(
    EndpointName=settings.SAGEMAKER_ENDPOINT_INFERENCE,
    ContentType="application/json",
    Body=json.dumps(payload),
)
print(json.loads(response["Body"].read()))
```

### Step 4: Delete the endpoint (do not skip)

```bash
python -m llm_engineering.infrastructure.aws.deploy.delete_sagemaker_endpoint
```

---

## 📝 Exercise: Scale Policy Tuning

### Task

Reason about autoscaling without burning money.

1. Read `TargetTrackingScalingPolicy.apply_policy`.
2. Set a target of **2 invocations per copy** vs **10 per copy**.
3. Predict which configuration scales out sooner and which uses more idle GPU time.
4. Explain the cooldown values in terms of cold-start time (model boot is slow).

**Goal**: Connect autoscaling parameters to cost and latency for LLM endpoints.

---

## 🐛 Common Pitfalls

- **Forgetting to delete the endpoint**: the top cost risk. Always run the delete script.
- **Role not set**: `create_endpoint` asserts `AWS_ARN_ROLE`. Create the role first.
- **Unsloth in the SageMaker image**: the managed HuggingFace image may not satisfy Unsloth's version requirements. If training fails, build a custom container.
- **Endpoint type mismatch**: autoscaling requires `INFERENCE_COMPONENT_BASED`. The `run.py` default is component-based; the `__main__` block uses `MODEL_BASED`, so pass the type explicitly in code.
- **Secrets in hyperparameters**: hyperparameters are logged in CloudWatch. Put tokens in `environment`, as the code does.

---

## 🎓 Knowledge Check

1. **What does the trust relationship in the execution role allow?**
   - Answer: `sagemaker.amazonaws.com` to assume the role.

2. **Why is `container_startup_health_check_timeout` 900 seconds?**
   - Answer: LLM containers must download multi-GB weights before serving traffic.

3. **What does `HF_MODEL_QUANTIZE="bitsandbytes"` accomplish?**
   - Answer: Loads the model quantized (8-bit) so it fits the endpoint GPU's VRAM.

4. **Which endpoint type can be autoscaled?**
   - Answer: `INFERENCE_COMPONENT_BASED`.

5. **Why are HF and Comet tokens passed via `environment`, not `hyperparameters`?**
   - Answer: Hyperparameters are logged; environment variables are not surfaced the same way.

6. **What three resources does the delete script remove?**
   - Answer: The endpoint, its endpoint configuration, and the model.

---

## 🔗 Next Session

**Session 6.1**: FastAPI REST API

We build the local inference API that wraps the model (SageMaker endpoint or local vLLM).

---

## 📚 Additional Resources

- [SageMaker Hugging Face Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/)
- [Deploy LLMs with Hugging Face on SageMaker](https://huggingface.co/docs/sagemaker/inference)
- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference)
- [Application Auto Scaling for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)

---

**Estimated Time**: 5-6 hours

**Prerequisites**: Sessions 5.1, 5.2, AWS account

**Outcome**: You can provision IAM roles, run a SageMaker training job, deploy a quantized LLM endpoint, autoscale it, and tear it down.
