# LLM 适配器，使用不同的结构化输出后端 { #llm-adapters-using-multiple-structured-output-backends }

Ragas 通过适配器模式支持多种结构化输出后端。本指南介绍如何为不同 LLM 服务商选择适配器。

## 概览 { #overview }

Ragas 使用适配器处理不同服务商的结构化输出。

- **Instructor 适配器**，支持 OpenAI、Anthropic、Azure、Groq、Mistral、Cohere 等服务商
- **LiteLLM 适配器**，支持 LiteLLM 覆盖的 100 多个服务商，包括 Gemini、Ollama、vLLM、Bedrock 等

框架会自动选择适合服务商的适配器，也可以显式指定。

## 快速开始 { #quick-start }

### 自动选择适配器（推荐） { #automatic-adapter-selection-recommended }

让 Ragas 自动检测适配器。

```python
from ragas.llms import llm_factory
from openai import OpenAI

# For OpenAI - automatically uses Instructor adapter
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o-mini", client=client)
```

```python
from ragas.llms import llm_factory
import google.generativeai as genai

# For Gemini - automatically uses LiteLLM adapter
genai.configure(api_key="...")
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
```

### 显式选择适配器 { #explicit-adapter-selection }

需要更多控制时，可以指定适配器。

```python
from ragas.llms import llm_factory

# Force using Instructor adapter
llm = llm_factory("gpt-4o", client=client, adapter="instructor")

# Force using LiteLLM adapter
llm = llm_factory("gemini-2.0-flash", client=client, adapter="litellm")
```

## 自动检测逻辑 { #auto-detection-logic }

`adapter="auto"`（默认）时，Ragas 按以下规则选择。

1. **检查客户端类型**，来自 `litellm` 模块时使用 LiteLLM 适配器
2. **检查服务商**，为 `google` 或 `gemini` 时使用 LiteLLM 适配器
3. **默认情况**，其余使用 Instructor 适配器

```python
from ragas.llms.adapters import auto_detect_adapter

# See which adapter will be used
adapter_name = auto_detect_adapter(client, "google")
print(adapter_name)  # Output: "litellm"

adapter_name = auto_detect_adapter(client, "openai")
print(adapter_name)  # Output: "instructor"
```

## 各服务商示例 { #provider-specific-examples }

### OpenAI

```python
from openai import OpenAI
from ragas.llms import llm_factory

client = OpenAI(api_key="your-key")
llm = llm_factory("gpt-4o", client=client)
# Uses Instructor adapter automatically
```

### Anthropic Claude

```python
from anthropic import Anthropic
from ragas.llms import llm_factory

client = Anthropic(api_key="your-key")
llm = llm_factory("claude-3-sonnet", provider="anthropic", client=client)
# Uses Instructor adapter automatically
```

### Google Gemini（使用 google-generativeai，推荐） { #google-gemini-with-google-generativeai-recommended }

```python
import google.generativeai as genai
from ragas.llms import llm_factory

genai.configure(api_key="your-key")
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
# Uses LiteLLM adapter automatically for google provider
```

### Google Gemini（使用 LiteLLM 代理，进阶） { #google-gemini-with-litellm-proxy-advanced }

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Requires running: litellm --model gemini-2.0-flash
client = OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"  # LiteLLM proxy endpoint
)
llm = llm_factory("gemini-2.0-flash", client=client, adapter="litellm")
# Uses LiteLLM adapter explicitly
```

### 本地模型（Ollama） { #local-models-ollama }

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Ollama exposes OpenAI-compatible API
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)
llm = llm_factory("mistral", provider="openai", client=client)
# Uses Instructor adapter
```

### AWS Bedrock

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Use LiteLLM proxy for Bedrock
# Note: Set up LiteLLM with Bedrock credentials first
client = OpenAI(
    api_key="",  # Bedrock uses IAM auth
    base_url="http://0.0.0.0:4000"  # LiteLLM proxy endpoint
)
llm = llm_factory("claude-3-sonnet", client=client, adapter="litellm")
```

### Groq

```python
from groq import Groq
from ragas.llms import llm_factory

client = Groq(api_key="your-key")
llm = llm_factory("mixtral-8x7b", provider="groq", client=client)
# Uses Instructor adapter automatically
```

### Mistral

```python
from mistralai import Mistral
from ragas.llms import llm_factory

client = Mistral(api_key="your-key")
llm = llm_factory("mistral-large", provider="mistral", client=client)
# Uses Instructor adapter automatically
```

### Cohere

```python
from cohere import Cohere
from ragas.llms import llm_factory

client = Cohere(api_key="your-key")
llm = llm_factory("command-r-plus", provider="cohere", client=client)
# Uses Instructor adapter automatically
```

## 适配器选择指南 { #adapter-selection-guide }

根据需要选择适配器。

### 以下情况适合 Instructor 适配器 { #use-instructor-adapter-if }
- 使用 OpenAI、Anthropic、Azure、Groq、Mistral 或 Cohere
- 服务商被 Instructor 原生支持
- 需要稳定且经过充分测试的方案
- 服务商无须特殊处理

### 以下情况适合 LiteLLM 适配器 { #use-litellm-adapter-if }
- 使用 Google Gemini
- 使用 Ollama、vLLM 等本地模型
- 使用 Bedrock 等服务商，或其他 LiteLLM 支持的后端
- 需要尽可能广泛的服务商兼容性
- 自动检测为该服务商选择了 LiteLLM

## 直接使用适配器 { #working-with-adapters-directly }

### 获取可用适配器 { #get-available-adapters }

```python
from ragas.llms.adapters import ADAPTERS

print(ADAPTERS)
# Output: {
#     "instructor": InstructorAdapter(),
#     "litellm": LiteLLMAdapter()
# }
```

### 获取指定适配器 { #get-specific-adapter }

```python
from ragas.llms.adapters import get_adapter

instructor = get_adapter("instructor")
litellm = get_adapter("litellm")

# Create LLM using adapter directly
llm = instructor.create_llm(client, "gpt-4o", "openai")
```

## 进阶用法 { #advanced-usage }

### 模型参数 { #model-arguments }

所有适配器支持相同的模型参数。

```python
llm = llm_factory(
    "gpt-4o",
    client=client,
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
)
```

### 系统提示词 { #system-prompts }

两类适配器都支持系统提示词，满足模型对特定指令的要求。

```python
llm = llm_factory(
    "gpt-4o",
    client=client,
    system_prompt="You are a helpful assistant that evaluates RAG systems."
)
```

系统提示词适用于以下情况。
- LLM 需要特定行为指令
- 微调模型使用了自定义系统提示词
- 希望统一所有指标的评估风格

系统提示词会作为 system 消息，加入每次 LLM 调用的开头。

### 自定义 Instructor 模式 { #custom-instructor-modes }

Instructor 适配器支持多种结构化输出生成模式，默认为 `Mode.JSON`。后端不支持某些功能时，可以指定其他模式。

```python
import instructor
from ragas.llms import llm_factory
from openai import OpenAI

# Use MD_JSON mode for backends without response_format support
client = OpenAI(api_key="...", base_url="https://custom-backend")
llm = llm_factory(
    "custom-model",
    provider="openai",
    client=client,
    mode=instructor.Mode.MD_JSON
)
```

可用模式包括以下几种。
- `Mode.JSON`（默认），使用 OpenAI 的 response_format 参数
- `Mode.MD_JSON`，在提示词中要求 Markdown JSON，适用于兼容性回退
- `Mode.TOOLS`，使用函数调用
- `Mode.JSON_SCHEMA`，使用 JSON Schema 验证

遇到以下错误时，可以尝试 `Mode.MD_JSON`。
```
Error code: 400 - {'message': 'only pytorch backend can use response_format now'}
```

### 异步支持 { #async-support }

两类适配器都支持异步操作。

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

async_client = AsyncOpenAI(api_key="...")
llm = llm_factory("gpt-4o", client=async_client)

# Async generation
response = await llm.agenerate(prompt, ResponseModel)
```

### 通过 LiteLLM 使用自定义服务商 { #custom-providers-with-litellm }

LiteLLM 支持许多 Instructor 未覆盖的服务商，可以采用 LiteLLM 代理方式。

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Set up LiteLLM proxy first:
# litellm --model grok-1  (for xAI)
# litellm --model deepseek-chat  (for DeepSeek)
# etc.

client = OpenAI(
    api_key="your-provider-api-key",
    base_url="http://0.0.0.0:4000"  # LiteLLM proxy endpoint
)

# xAI Grok
llm = llm_factory("grok-1", client=client, adapter="litellm")

# DeepSeek
llm = llm_factory("deepseek-chat", client=client, adapter="litellm")

# Together AI
llm = llm_factory("mistral-7b", client=client, adapter="litellm")
```

## 完整评估示例 { #complete-evaluation-example }

```python
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
)

# Initialize LLM with your provider
import google.generativeai as genai
genai.configure(api_key="...")
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Create evaluation dataset
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris"],
    "contexts": [["France is in Europe. Paris is its capital."]],
    "ground_truth": ["Paris"]
}
dataset = Dataset.from_dict(data)

# Define metrics
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm),
]

# Evaluate
results = evaluate(dataset, metrics=metrics)
print(results)
```

## 故障排查 { #troubleshooting }

### 出现“Unknown adapter: xyz” { #unknown-adapter-xyz }

检查适配器名称是否有效。

```python
# Valid: "instructor" or "litellm"
llm = llm_factory("model", client=client, adapter="instructor")

# Invalid: "dspy" (not yet implemented)
# llm = llm_factory("model", client=client, adapter="dspy")  # Error!
```

### 出现“Failed to initialize provider client” { #failed-to-initialize-provider-client }

确保满足以下条件。
1. 客户端已正确初始化
2. API 密钥有效
3. 适配器支持该服务商

```python
# Check if adapter can handle your provider
from ragas.llms.adapters import auto_detect_adapter
adapter = auto_detect_adapter(client, "my-provider")
print(f"Will use: {adapter}")
```

### 适配器不匹配 { #adapter-mismatch }

自动检测能够处理多数情况，也可以显式指定。

```python
# If auto-detection picks the wrong adapter:
llm = llm_factory(
    "model",
    provider="provider-name",
    client=client,
    adapter="litellm"  # Explicit override
)
```

## 迁移指南 { #migration-guide }

### 从纯文本输出迁移到结构化输出 { #from-text-only-to-structured-output }

从纯文本 LLM 用法升级时，可以按以下方式修改。

```python
# Before (deprecated)
# from ragas.llms import LangchainLLMWrapper
# llm = LangchainLLMWrapper(langchain_llm)

# After (new way)
from ragas.llms import llm_factory
llm = llm_factory("gpt-4o", client=client)
```

### 切换服务商 { #switching-providers }

从 OpenAI 切换到 Gemini 的示例如下。

```python
# Before: OpenAI
from openai import OpenAI
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o", client=client)

# After: Gemini (similar code pattern!)
import google.generativeai as genai
genai.configure(api_key="...")
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
# Adapter automatically switches to LiteLLM for google provider
```

## 另请参阅 { #see-also }

- [Gemini 集成指南](integrations/gemini.md)，详细配置
- [LLM 工厂参考](customizations/customize_models.md)，完整 API 参考
- [指标文档](../concepts/metrics/index.md)，结合 LLM 使用指标
