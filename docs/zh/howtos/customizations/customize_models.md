## 自定义模型 { #customize-models }

Ragas 在评估和合成数据生成中可能使用 LLM、嵌入模型，或两者同时使用。你可以根据可用资源自行配置。

Ragas 提供工厂函数 `llm_factory` 与 `embedding_factory`，支持多个服务商。

- **直接支持** OpenAI、Anthropic 和 Google
- **通过 LiteLLM 支持其他服务商**，包括 Azure OpenAI、AWS Bedrock、Google Vertex AI 等 100 多个服务商

工厂函数使用 [Instructor](https://python.useinstructor.com/) 实现结构化输出，使用 [LiteLLM](https://docs.litellm.ai/) 统一访问多个 LLM 服务商。

## 系统提示词 { #system-prompts }

可以提供系统提示词，统一调整所有评估中的 LLM 行为。

```python
from ragas.llms import llm_factory
from openai import OpenAI

client = OpenAI(api_key="your-key")
llm = llm_factory(
    "gpt-4o",
    client=client,
    system_prompt="You are a helpful assistant that evaluates RAG systems."
)
```

系统提示词尤其适用于以下场景。
- 微调模型需要特定的系统指令
- 需要统一引导评估行为
- 模型需要自定义提示词才能正常工作

## 示例 { #examples }

- [自定义模型](#customize-models)
- [系统提示词](#system-prompts)
- [示例](#examples)
  - [Azure OpenAI](#azure-openai)
  - [Google Vertex](#google-vertex)
  - [AWS Bedrock](#aws-bedrock)


### Azure OpenAI

```bash
pip install litellm
```

```python
import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

azure_configs = {
    "api_base": "https://<your-endpoint>.openai.azure.com/",
    "api_key": "your-api-key",
    "api_version": "2024-02-15-preview",
    "model_deployment": "your-deployment-name",
    "embedding_deployment": "your-embedding-deployment-name",
}

# Configure LiteLLM for Azure OpenAI (used by LLM calls)
litellm.api_base = azure_configs["api_base"]
litellm.api_key = azure_configs["api_key"]
litellm.api_version = azure_configs["api_version"]

# Create LLM using llm_factory with litellm provider
# Note: Use deployment name, not model name for Azure
# Important: Pass litellm.completion (the function), not the module
azure_llm = llm_factory(
    f"azure/{azure_configs['model_deployment']}",
    provider="litellm",
    client=litellm.completion,
    # Optional: Add system prompt
    # system_prompt="You are a helpful assistant that evaluates RAG systems."
)

# Create embeddings using embedding_factory
# Note: Pass Azure config directly to embedding_factory
azure_embeddings = embedding_factory(
    "litellm",
    model=f"azure/{azure_configs['embedding_deployment']}",
    api_base=azure_configs["api_base"],
    api_key=azure_configs["api_key"],
    api_version=azure_configs["api_version"],
)
```
现在可以将 Ragas 与 Azure OpenAI 端点一起使用了。

### Google Vertex

```bash
pip install litellm google-cloud-aiplatform
```

```python
import litellm
import os
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

config = {
    "project_id": "<your-project-id>",
    "location": "us-central1",  # e.g., "us-central1", "us-east1"
    "chat_model_id": "gemini-1.5-pro-002",
    "embedding_model_id": "text-embedding-005",
}

# Set environment variables for Vertex AI (used by litellm)
os.environ["VERTEXAI_PROJECT"] = config["project_id"]
os.environ["VERTEXAI_LOCATION"] = config["location"]

# Create LLM using llm_factory with litellm provider
# Important: Pass litellm.completion (the function), not the module
vertex_llm = llm_factory(
    f"vertex_ai/{config['chat_model_id']}",
    provider="litellm",
    client=litellm.completion,
    # Optional: Add system prompt
    # system_prompt="You are a helpful assistant that evaluates RAG systems."
)

# Create embeddings using embedding_factory
# Note: Embeddings use the environment variables set above
vertex_embeddings = embedding_factory(
    "litellm",
    model=f"vertex_ai/{config['embedding_model_id']}",
)
```
现在可以将 Ragas 与 Google Vertex AI 端点一起使用了。

### AWS Bedrock

```bash
pip install litellm
```

```python
import litellm
import os
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

config = {
    "region_name": "us-east-1",  # E.g. "us-east-1"
    "llm": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Your LLM model ID
    "embeddings": "amazon.titan-embed-text-v2:0",  # Your embedding model ID
    "temperature": 0.4,
}

# Set AWS credentials as environment variables
# Option 1: Use AWS credentials file (~/.aws/credentials)
# Option 2: Set environment variables directly
os.environ["AWS_REGION_NAME"] = config["region_name"]
# os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"

# Create LLM using llm_factory with litellm provider
# Important: Pass litellm.completion (the function), not the module
bedrock_llm = llm_factory(
    f"bedrock/{config['llm']}",
    provider="litellm",
    client=litellm.completion,
    temperature=config["temperature"],
    # Optional: Add system prompt
    # system_prompt="You are a helpful assistant that evaluates RAG systems."
)

# Create embeddings using embedding_factory
# Note: Embeddings use the environment variables set above
bedrock_embeddings = embedding_factory(
    "litellm",
    model=f"bedrock/{config['embeddings']}",
)
```
现在可以将 Ragas 与 AWS Bedrock 端点一起使用了。
