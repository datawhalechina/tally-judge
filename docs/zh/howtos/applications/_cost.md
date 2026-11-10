# 估算评估与测试集生成的成本和用量 { #how-to-estimate-cost-and-usage-of-evaluations-and-testset-generation }

使用 LLM 进行评估和测试集生成时，需要考虑成本。Ragas 提供了一些工具来统计用量与估算费用。

## 实现 `TokenUsageParser` { #implement-tokenusageparser }

Ragas 默认不统计 `evaluate()` 的 token 用量，因为 LangChain 的不同 LLM 并不总以统一格式返回用量信息。要获取这些数据，需要实现 `TokenUsageParser`。

`TokenUsageParser` 是一个函数，将 LangChain 模型 `generate_prompt()` 返回的 `LLMResult` 或 `ChatResult` 解析成 Ragas 所需的 `TokenUsage`。

下面使用已定义的解析器，演示如何解析 OpenAI 用量。


```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompt_values import StringPromptValue

gpt4o = ChatOpenAI(model="gpt-4o")
p = StringPromptValue(text="hai there")
llm_result = gpt4o.generate_prompt([p])

# lets import a parser for OpenAI
from ragas.cost import get_token_usage_for_openai

get_token_usage_for_openai(llm_result)
```
输出如下。
```
TokenUsage(input_tokens=9, output_tokens=9, model='')
```


可以自行定义解析器，也可以导入现有解析器。如果希望建议支持某个 LLM 服务商，或贡献解析器，请查看[此 issue](https://github.com/vibrantlabsai/ragas/issues/1151)。

## 评估的 Token 用量 { #token-usage-for-evaluations }

使用 `get_token_usage_for_openai` 解析器统计一次评估的 token 用量。


```python
from ragas import EvaluationDataset
from datasets import load_dataset

dataset = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")

eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])
```
输出如下。
```
Repo card metadata block was not found. Setting CardData to empty.
```

将解析器传给 `evaluate()`，统计结果会包含在返回的 `Result` 对象中，便于计算费用。


```python
from ragas import evaluate
from ragas.metrics import LLMContextRecall

from ragas.cost import get_token_usage_for_openai

result = evaluate(
    eval_dataset,
    metrics=[LLMContextRecall()],
    llm=gpt4o,
    token_usage_parser=get_token_usage_for_openai,
)
```
输出如下。
```
Evaluating:   0%|          | 0/20 [00:00<?, ?it/s]
```


```python
result.total_tokens()
```
输出如下。
```
TokenUsage(input_tokens=25097, output_tokens=3757, model='')
```


将每个 token 的价格传给 `Result.total_cost()`，即可计算每次运行的费用。

此示例按 GPT-4o 每 100 万输入 token 5 美元、每 100 万输出 token 15 美元估算。


```python
result.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```

输出如下。
```
1.1692900000000002
```


## 测试集生成的 Token 用量 { #token-usage-for-testset-generation }

测试集生成也可使用相同解析器，将其通过 `token_usage_parser` 传给 `generate()`。当前只统计生成过程的费用，变换过程的费用另行计算。

下面加载已有知识图谱并生成测试集。更多背景请参阅[测试集生成](../../getstarted/rag_testset_generation.md#a-deeper-look)。


```python
from ragas.testset.graph import KnowledgeGraph

# loading an existing KnowledgeGraph
# make sure to change the path to the location of the KnowledgeGraph file
kg = KnowledgeGraph.load("../../../experiments/scratchpad_kg.json")
kg
```

输出如下。
```
KnowledgeGraph(nodes: 47, relationships: 109)
```



### 选择 LLM { #choose-your-llm }

=== "OpenAI"
    安装 langchain-openai 包。

    ```bash
    pip install langchain-openai
    ```

    在环境中设置 OpenAI API 密钥。

    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "your-openai-key"
    ```

    通过 `LangchainLLMWrapper` 包装 LLM，以便用于 Ragas。

    ```python
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    import openai

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    openai_client = openai.OpenAI()
    generator_embeddings = OpenAIEmbeddings(client=openai_client)
    ```


=== "AWS"
    安装 langchain-aws 包。

    ```bash
    pip install langchain-aws
    ```

    配置 AWS 凭据与参数。

    ```python
    config = {
        "credentials_profile_name": "your-profile-name",  # E.g "default"
        "region_name": "your-region-name",  # E.g. "us-east-1"
        "llm": "your-llm-model-id",  # E.g "anthropic.claude-3-5-sonnet-20241022-v2:0"
        "embeddings": "your-embedding-model-id",  # E.g "amazon.titan-embed-text-v2:0"
        "temperature": 0.4,
    }
    ```

    定义 LLM，并通过 `LangchainLLMWrapper` 包装，以便用于 Ragas。

    ```python
    from langchain_aws import ChatBedrockConverse
    from langchain_aws import BedrockEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    generator_llm = LangchainLLMWrapper(ChatBedrockConverse(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        base_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
        model=config["llm"],
        temperature=config["temperature"],
    ))
    generator_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        model_id=config["embeddings"],
    ))
    ```

    其他 AWS 服务的用法参阅 [langchain-aws 文档](https://python.langchain.com/docs/integrations/providers/aws/)。

=== "Google Cloud"
    Google 提供 Google AI 和 Google Cloud Vertex AI 两种模型访问方式。Google AI 只需 Google 账户与 API 密钥，Vertex AI 则需要 Google Cloud 账户，适合企业功能需求。

    首先安装所需软件包。

    ```bash
    pip install langchain-google-genai langchain-google-vertexai
    ```

    根据所选 API 配置凭据。

    使用 Google AI 时，

    ```python
    import os
    os.environ["GOOGLE_API_KEY"] = "your-google-ai-key"  # From https://ai.google.dev/
    ```

    使用 Vertex AI 时，

    ```python
    # Ensure you have credentials configured (gcloud, workload identity, etc.)
    # Or set service account JSON path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service-account.json"
    ```

    定义配置。

    ```python
    config = {
        "model": "gemini-1.5-pro",  # or other model IDs
        "temperature": 0.4,
        "max_tokens": None,
        "top_p": 0.8,
        # For Vertex AI only:
        "project": "your-project-id",  # Required for Vertex AI
        "location": "us-central1",     # Required for Vertex AI
    }
    ```

    初始化并包装 LLM，以便用于 Ragas。

    ```python
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Choose the appropriate import based on your API:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_vertexai import ChatVertexAI

    # Initialize with Google AI Studio
    generator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
    ))

    # Or initialize with Vertex AI
    generator_llm = LangchainLLMWrapper(ChatVertexAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        project=config["project"],
        location=config["location"],
    ))
    ```


    也可以按需配置安全设置。

    ```python
    from langchain_google_genai import HarmCategory, HarmBlockThreshold

    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        # Add other safety settings as needed
    }

    # Apply to your LLM initialization
    generator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"],
        safety_settings=safety_settings,
    ))
    ```

    初始化并包装嵌入模型，以便用于 Ragas。

    ```python
    # Google AI Studio Embeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    generator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Google's text embedding model
        task_type="retrieval_document"  # Optional: specify the task type
    ))
    ```

    ```python
    # Vertex AI Embeddings
    from langchain_google_vertexai import VertexAIEmbeddings

    generator_embeddings = LangchainEmbeddingsWrapper(VertexAIEmbeddings(
        model_name="textembedding-gecko@001",  # or other available model
        project=config["project"],  # Your GCP project ID
        location=config["location"]  # Your GCP location
    ))
    ```

    有关可用模型、功能与配置的更多信息，请参阅以下文档。
    - [Vertex AI 文档](https://cloud.google.com/vertex-ai/docs)
    - [LangChain Google AI 集成](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
    - [LangChain Vertex AI 集成](https://python.langchain.com/docs/integrations/chat/google_vertex_ai)


=== "Azure"
    安装 langchain-openai 包。

    ```bash
    pip install langchain-openai
    ```

    在环境中设置 Azure OpenAI API 密钥。

    ```python
    import os
    os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-key"

    # other configuration
    azure_config = {
        "base_url": "",  # your endpoint
        "model_deployment": "",  # your model deployment name
        "model_name": "",  # your model name
        "embedding_deployment": "",  # your embedding deployment name
        "embedding_name": "",  # your embedding name
    }

    ```

    定义 LLM，并通过 `LangchainLLMWrapper` 包装，以便用于 Ragas。

    ```python
    from langchain_openai import AzureChatOpenAI
    from langchain_openai import AzureOpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    generator_llm = LangchainLLMWrapper(AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    ))

    # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    generator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    ))
    ```

    其他 Azure 服务的用法参阅 [langchain-azure 文档](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)。

=== "其他服务商"
    使用其他服务商、且通过 LangChain 调用时，可通过 `LangchainLLMWrapper` 包装 LLM。

    ```python
    from ragas.llms import LangchainLLMWrapper
    generator_llm = LangchainLLMWrapper(your_llm_instance)
    ```

    详细方法参阅[自定义模型指南](../customizations/customize_models.md)。

    使用 LlamaIndex 时，可以通过 `LlamaIndexLLMWrapper` 包装 LLM。

    ```python
    from ragas.llms import LlamaIndexLLMWrapper
    generator_llm = LlamaIndexLLMWrapper(your_llm_instance)
    ```

    详情参阅 [LlamaIndex 集成指南](../integrations/_llamaindex.md)。

    如果仍无法使用所需服务商，请在[此 issue](https://github.com/vibrantlabsai/ragas/issues/1617)中反馈，帮助我们完善支持。🙂


```python
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory

tg = TestsetGenerator(llm=llm_factory(), knowledge_graph=kg)
# generating a testset
testset = tg.generate(testset_size=10, token_usage_parser=get_token_usage_for_openai)
```


```python
# total cost for the generation process
testset.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```

Output
```
0.20967000000000002
```

