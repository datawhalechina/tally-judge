# Google Gemini 集成指南 { #google-gemini-integration-guide }

本指南介绍如何配置 Google Gemini 模型，并用于 Ragas 评估。

## 概览 { #overview }

Ragas 支持自动为 Gemini 选择适配器，同时兼容新版 `google-genai` SDK（推荐）与旧版 `google-generativeai` SDK。

## 配置 { #setup }

### 前置条件 { #prerequisites }

- 具有 Gemini API 访问权限的 Google API 密钥
- Python 3.8+
- 已安装 Ragas

### 安装 { #installation }

安装所需依赖。

```bash
# Recommended: New Google GenAI SDK
pip install ragas google-genai

# Legacy (deprecated, support ends Aug 2025)
pip install ragas google-generativeai
```

## 配置 { #configuration }

### 方式一，使用新版 Google GenAI SDK（推荐） { #option-1-using-new-google-genai-sdk-recommended }

建议使用新版 `google-genai` SDK。

```python
import os
from google import genai
from ragas.llms import llm_factory

# Create client with API key
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Create LLM - adapter is auto-detected for google provider
llm = llm_factory(
    "gemini-2.0-flash",
    provider="google",
    client=client
)
```

### 方式二，使用旧版 SDK（已弃用） { #option-2-using-legacy-sdk-deprecated }

旧版 `google-generativeai` SDK 仍可用于文中示例，但已弃用，其支持期截至 2025 年 8 月。

```python
import os
import google.generativeai as genai
from ragas.llms import llm_factory

# Configure with your API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Create client
client = genai.GenerativeModel("gemini-2.0-flash")

# Create LLM
llm = llm_factory(
    "gemini-2.0-flash",
    provider="google",
    client=client
)
```

### 方式三，使用 LiteLLM 代理（进阶） { #option-3-using-litellm-proxy-advanced }

需要 LiteLLM 代理功能时，先配置代理服务器，再按以下方式使用。

```python
import os
from openai import OpenAI
from ragas.llms import llm_factory

# Requires running: litellm --model gemini-2.0-flash
client = OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"  # LiteLLM proxy endpoint
)

# Create LLM with explicit adapter selection
llm = llm_factory("gemini-2.0-flash", client=client, adapter="litellm")
```

## 支持的模型 { #supported-models }

Ragas 可以与 Gemini 系列模型一起使用，文中列举以下型号。

- **较新系列示例**：`gemini-2.0-flash`
- **1.5 系列**：`gemini-1.5-pro`、`gemini-1.5-flash`
- **1.0 系列**：`gemini-1.0-pro`

最新可用模型与价格请查看 [Google AI Studio](https://aistudio.google.com/apikey)。

## 嵌入模型配置 { #embeddings-configuration }

Ragas 指标可按是否需要嵌入模型分组。

1. **仅使用 LLM 的指标**，无须嵌入模型。
   - ContextPrecision
   - ContextRecall
   - Faithfulness
   - AspectCritic

2. **依赖嵌入模型的指标**，需要提供嵌入模型。
   - AnswerCorrectness
   - AnswerRelevancy
   - AnswerSimilarity
   - SemanticSimilarity
   - ContextEntityRecall

### 自动匹配服务商 { #automatic-provider-matching }

使用旧版评估流程时，嵌入服务商会**自动匹配** LLM 服务商。传入 Gemini LLM，默认使用 Google 嵌入，**无须 OpenAI API 密钥**。

### 方式一，默认嵌入模型（推荐） { #option-1-default-embeddings-recommended }

让 Ragas 根据 LLM 自动选择嵌入模型。

```python
import os
from datasets import Dataset
from google import genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness
)

# Initialize Gemini client (new SDK)
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Create sample evaluation data
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["France is a country in Western Europe. Paris is its capital."]],
    "ground_truth": ["Paris"]
}

dataset = Dataset.from_dict(data)

# Define metrics - embeddings are auto-configured for Google
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm)  # Uses Google embeddings automatically
]

# Run evaluation
results = evaluate(dataset, metrics=metrics)
print(results)
```

### 方式二，显式配置嵌入模型 { #option-2-explicit-embeddings }

需要完全控制时，可以单独创建 Google 嵌入模型，并选择不同配置方式。

```python
import os
from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.embeddings.base import embedding_factory
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerCorrectness, ContextPrecision, ContextRecall, Faithfulness

# Initialize Gemini client (new SDK)
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Initialize Google embeddings (multiple options):

# Option A: Using the same client (recommended for new SDK)
embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

# Option B: Using embedding factory
embeddings = embedding_factory("google", model="gemini-embedding-001")

# Option C: Auto-import (creates client automatically)
embeddings = GoogleEmbeddings(model="gemini-embedding-001")

# Create sample evaluation data
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["France is a country in Western Europe. Paris is its capital."]],
    "ground_truth": ["Paris"]
}

dataset = Dataset.from_dict(data)

# Define metrics with explicit embeddings
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm, embeddings=embeddings)
]

# Run evaluation
results = evaluate(dataset, metrics=metrics)
print(results)
```

## 示例，完整评估 { #example-complete-evaluation }

以下使用 Gemini 评估 RAG 应用，并自动匹配嵌入服务商。

```python
import os
from datasets import Dataset
from google import genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness
)

# Initialize Gemini client (new SDK)
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Create sample evaluation data
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["France is a country in Western Europe. Paris is its capital."]],
    "ground_truth": ["Paris"]
}

dataset = Dataset.from_dict(data)

# Define metrics - embeddings automatically use Google provider
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm)
]

# Run evaluation
results = evaluate(dataset, metrics=metrics)
print(results)
```

## 性能考虑 { #performance-considerations }

### 模型选择 { #model-selection }

- **gemini-2.0-flash**，侧重速度与效率
- **gemini-1.5-pro**，适合较复杂评估的推理
- **gemini-1.5-flash**，兼顾速度与成本

### 成本优化 { #cost-optimization }

大规模评估时，可以考虑以下做法。

1. 多数指标使用 `gemini-2.0-flash`
2. 批量处理多次评估
3. 尽可能缓存提示词，Gemini 支持提示词缓存

### 异步支持 { #async-support }

高吞吐评估可以使用异步操作。

```python
import os
from google import genai
from ragas.llms import llm_factory

# Create client (new SDK)
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Use in async evaluation
# response = await llm.agenerate(prompt, ResponseModel)
```

## 适配器选择 { #adapter-selection }

Ragas 根据配置自动选择合适的适配器。

```python
# Auto-detection happens automatically
# For Gemini: uses LiteLLM adapter
# For other providers: uses Instructor adapter

# Explicit selection (if needed)
llm = llm_factory(
    "gemini-2.0-flash",
    client=client,
    adapter="litellm"  # Explicit adapter selection
)

# Check auto-detected adapter
from ragas.llms.adapters import auto_detect_adapter
adapter_name = auto_detect_adapter(client, "google")
print(f"Using adapter: {adapter_name}")  # Output: Using adapter: litellm
```

## 故障排查 { #troubleshooting }

### API 密钥问题 { #api-key-issues }

```python
# Make sure your API key is set
import os
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set")
```

### 已知问题，新 SDK 的 Instructor 安全设置 { #known-issue-instructor-safety-settings-new-sdk }

原文记录了 Instructor 上游的一项问题：使用新版 `google-genai` SDK 时，可能向 Gemini API 发送无效安全设置，并出现以下错误。

```
Invalid value at 'safety_settings[5].category'... "HARM_CATEGORY_JAILBREAK"
```

**应对方法**

1. 使用 OpenAI 兼容端点。
```python
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
llm = llm_factory("gemini-2.0-flash", provider="openai", client=client)
```

2. 跟踪上游 issue：[instructor#1658](https://github.com/567-labs/instructor/issues/1658)。

嵌入功能可以正常使用新版 SDK，该问题只影响 LLM 生成。

### 速率限制 { #rate-limits }

Gemini 存在调用限额。LLM 适配器会处理重试与超时，需要更细控制时，可在 HTTP 客户端层配置合适的超时。

### 模型可用性 { #model-availability }

模型不可用时，检查以下事项。

1. 在 [Google Cloud Console](https://console.cloud.google.com) 检查地区与配额
2. 尝试支持列表中的其他模型
3. 确认 API 密钥具有 Generative AI API 访问权限

## 从其他服务商迁移 { #migration-from-other-providers }

### 从 OpenAI 迁移 { #from-openai }

```python
# Before: OpenAI-only
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = llm_factory("gpt-4o", client=client)

# After: Gemini with new SDK
from google import genai
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
```

### 从 Anthropic 迁移 { #from-anthropic }

```python
# Before: Anthropic
from anthropic import Anthropic
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
llm = llm_factory("claude-3-sonnet", provider="anthropic", client=client)

# After: Gemini with new SDK
from google import genai
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
```

### 从旧版 google-generativeai SDK 迁移 { #from-legacy-google-generativeai-sdk }

```python
# Before: Legacy SDK (deprecated)
import google.generativeai as genai
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# After: New SDK (recommended)
from google import genai
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
```

## 结合新版 Metrics Collections 使用 { #using-with-metrics-collections-modern-approach }

使用新版 collections API 时，需要显式创建 LLM 与嵌入模型。

```python
import os
from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import AnswerCorrectness, ContextPrecision

# Create client (new SDK)
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Create LLM
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Create embeddings using the same client
embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

# Create metrics with explicit LLM and embeddings
metrics = [
    ContextPrecision(llm=llm),  # LLM-only metric
    AnswerCorrectness(llm=llm, embeddings=embeddings),  # Needs both
]

# Use metrics with your evaluation workflow
result = await metrics[1].ascore(
    user_input="What is the capital of France?",
    response="Paris",
    reference="Paris is the capital of France."
)
```

**与旧版用法的区别**
- 旧版 `evaluate()` 根据 LLM 服务商自动创建嵌入模型
- 新版 collections 需要为每个指标显式传入嵌入模型

这样可以更精确地控制配置，并结合 Gemini 使用。

## 支持的指标 { #supported-metrics }

Ragas 各类指标可结合 Gemini 使用。

- 答案正确性
- 回答相关性
- 答案相似度
- 维度评判
- 上下文精确率
- 上下文召回率
- 上下文实体召回率
- 忠实度
- 自然语言推断评估（NLI Eval）
- 回答相关性（Response Relevancy）

详情参阅[指标参考](../../concepts/metrics/index.md)。

## 进阶，自定义模型参数 { #advanced-custom-model-parameters }

向 Gemini 传入自定义参数。

```python
llm = llm_factory(
    "gemini-2.0-flash",
    client=client,
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
    top_k=40,
)
```

## 参考资料 { #resources }

- [Google GenAI SDK 文档](https://googleapis.github.io/python-genai/)
- [Google Gemini API 文档](https://ai.google.dev/gemini-api/docs)
- [Ragas 指标文档](../../concepts/metrics/index.md)
- [Ragas LLM 工厂指南](../customizations/customize_models.md)
