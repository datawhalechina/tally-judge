# OCI Gen AI 集成 { #oci-gen-ai-integration }

本指南介绍如何使用 Oracle Cloud Infrastructure（OCI）生成式 AI 模型开展 Ragas 评估。

## 安装 { #installation }

先安装 OCI 依赖。

```bash
pip install ragas[oci]
```

## 配置 { #setup }

### 1. 配置 OCI 身份验证 { #1-configure-oci-authentication }

选择以下任一方式配置。

#### 方式 A，OCI CLI 配置 { #option-a-oci-cli-configuration }
```bash
oci setup config
```

#### 方式 B，环境变量 { #option-b-environment-variables }
```bash
export OCI_CONFIG_FILE=~/.oci/config
export OCI_PROFILE=DEFAULT
```

#### 方式 C，手动配置 { #option-c-manual-configuration }
```python
config = {
    "user": "ocid1.user.oc1..example",
    "key_file": "~/.oci/private_key.pem",
    "fingerprint": "your_fingerprint",
    "tenancy": "ocid1.tenancy.oc1..example",
    "region": "us-ashburn-1"
}
```

### 2. 获取所需 ID { #2-get-required-ids }

需要以下信息。
- **模型 ID**，例如 `cohere.command`、`meta.llama-3-8b`
- **区间 ID**，OCI compartment 的 OCID
- **端点 ID**，使用自定义端点时提供，可选

## 用法 { #usage }

### 基本用法 { #basic-usage }

```python
from ragas.llms import oci_genai_factory
from ragas import evaluate
from datasets import Dataset

# Initialize OCI Gen AI LLM
llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)

# Your dataset
dataset = Dataset.from_dict({
    "question": ["What is the capital of France?"],
    "answer": ["Paris"],
    "contexts": [["France is a country in Europe. Its capital is Paris."]],
    "ground_truth": ["Paris"]
})

# Evaluate with OCI Gen AI
result = evaluate(
    dataset,
    llm=llm,
    embeddings=None  # You can use any embedding model
)
```

### 进阶配置 { #advanced-configuration }

```python
from ragas.llms import oci_genai_factory
from ragas.run_config import RunConfig

# Custom OCI configuration
config = {
    "user": "ocid1.user.oc1..example",
    "key_file": "~/.oci/private_key.pem",
    "fingerprint": "your_fingerprint",
    "tenancy": "ocid1.tenancy.oc1..example",
    "region": "us-ashburn-1"
}

# Custom run configuration
run_config = RunConfig(
    timeout=60,
    max_retries=3
)

# Initialize with custom config and endpoint
llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example",
    config=config,
    endpoint_id="ocid1.endpoint.oc1..example",  # Optional
    run_config=run_config
)
```

### 使用不同模型 { #using-with-different-models }

```python
# Cohere Command model
llm_cohere = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)

# Meta Llama model
llm_llama = oci_genai_factory(
    model_id="meta.llama-3-8b",
    compartment_id="ocid1.compartment.oc1..example"
)

# Using with different endpoints
llm_endpoint = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example",
    endpoint_id="ocid1.endpoint.oc1..example"
)
```

## 可用模型 { #available-models }

OCI Gen AI 支持多类模型，示例如下。

- **Cohere**: `cohere.command`, `cohere.command-light`
- **Meta**: `meta.llama-3-8b`, `meta.llama-3-70b`
- **Mistral**: `mistral.mistral-7b-instruct`
- **更多型号**，请查阅 OCI 文档中的最新列表

## 错误处理 { #error-handling }

OCI Gen AI 封装提供错误处理机制。

```python
try:
    result = evaluate(dataset, llm=llm)
except Exception as e:
    print(f"Evaluation failed: {e}")
```

## 性能考虑 { #performance-considerations }

1. **速率限制**，配置合适的重试策略。
2. **超时**，按场景设置超时时间。
3. **批量处理**，封装支持多次补全的批处理。

## 故障排查 { #troubleshooting }

### 常见问题 { #common-issues }

1. **身份验证错误**
   ```
   Error: OCI SDK authentication failed
   ```
   检查 OCI 配置与凭据。

2. **找不到模型**
   ```
   Error: Model not found in compartment
   ```
   确认模型 ID 在当前区间中存在。

3. **权限错误**
   ```
   Error: Insufficient permissions
   ```
   确保用户拥有生成式 AI 服务所需的 IAM 策略。

### 调试模式 { #debug-mode }

启用调试日志，排查问题。

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your OCI Gen AI code here
```

## 示例 { #examples }

### 完整评估示例 { #complete-evaluation-example }

```python
from ragas import evaluate
from ragas.llms import oci_genai_factory
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# Initialize OCI Gen AI
llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)

# Create dataset
dataset = Dataset.from_dict({
    "question": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?"
    ],
    "answer": [
        "Paris is the capital of France.",
        "William Shakespeare wrote Romeo and Juliet."
    ],
    "contexts": [
        ["France is a country in Europe. Its capital is Paris."],
        ["Romeo and Juliet is a play by William Shakespeare."]
    ],
    "ground_truth": [
        "Paris",
        "William Shakespeare"
    ]
})

# Evaluate
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm
)

print(result)
```

### 使用 OCI Gen AI 创建自定义指标 { #custom-metrics-with-oci-gen-ai }

```python
from ragas.metrics import MetricWithLLM

# Create custom metric using OCI Gen AI
class CustomMetric(MetricWithLLM):
    def __init__(self):
        super().__init__()
        self.llm = oci_genai_factory(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example"
        )

# Use in evaluation
result = evaluate(
    dataset,
    metrics=[CustomMetric()],
    llm=llm
)
```

## 最佳实践 { #best-practices }

1. **选择合适模型**，根据评估需求选型。
2. **监控费用**，OCI Gen AI 按用量计费。
3. **处理错误**，生产环境中实现完善的异常处理。
4. **使用缓存**，重复评估时启用缓存。
5. **批量操作**，在合适时使用批处理提高效率。

## 获取支持 { #support }

OCI Gen AI 集成问题可参考以下资源。
- [OCI 文档](https://docs.oracle.com/en-us/iaas/Content/generative-ai/)
- [OCI Python SDK](https://docs.oracle.com/en-us/iaas/tools/python/2.160.1/api/generative_ai.html)
- [Ragas GitHub issues](https://github.com/vibrantlabsai/ragas/issues)
