# 为 RAG 生成测试集 { #testset-generation-for-rag }

本指南介绍如何利用自己的文档生成测试集，用于评估 RAG 流水线。

## 快速开始 { #quickstart }
先通过一个简单示例了解如何生成 RAG 测试集，再深入了解测试集生成流水线的主要组件。

### 获取示例文档 { #load-sample-documents }

本教程使用[此仓库](https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown)中的示例文档。你可以替换为自己的文档。

```bash
git clone https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown
```

### 加载文档 { #load-documents }

使用 [langchain_community](https://python.langchain.com/docs/concepts/document_loaders/) 中的 `DirectoryLoader` 加载示例数据集里的文档。也可以使用 [llama_index](https://docs.llamaindex.ai/en/stable/understanding/loading/llamahub/) 提供的加载器。

```shell
pip install langchain-community
```

```python
from langchain_community.document_loaders import DirectoryLoader

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
```

### 选择 LLM { #choose-your-llm }

你可以选择[所需的 LLM](../howtos/customizations/customize_models.md)。
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

    详细方法参阅[自定义模型指南](../howtos/customizations/customize_models.md)。

    使用 LlamaIndex 时，可以通过 `LlamaIndexLLMWrapper` 包装 LLM。

    ```python
    from ragas.llms import LlamaIndexLLMWrapper
    generator_llm = LlamaIndexLLMWrapper(your_llm_instance)
    ```

    详情参阅 [LlamaIndex 集成指南](../howtos/integrations/_llamaindex.md)。

    如果仍无法使用所需服务商，请在[此 issue](https://github.com/vibrantlabsai/ragas/issues/1617)中反馈，帮助我们完善支持。🙂

### 生成测试集 { #generate-testset }

加载文档并配置 LLM 后，即可生成测试数据。如果使用 `llama_index` 加载文档，请改用 `generate_with_llama_index_docs` 方法。

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
```

### 分析测试集 { #analyzing-the-testset }

生成测试集后，可以查看内容并挑选适合纳入最终测试集的查询。将测试集导出为 pandas DataFrame，即可开展进一步分析。

```python
dataset.to_pandas()
```

输出如下。
![测试集](../../getstarted/testset_output.png)

!!! note
生成合成测试数据可能并不容易。我们已为多种场景构建了测试数据生成流水线，如果需要帮助，可以[预约交流](https://bit.ly/3EBYq4J)，或发送邮件至 [founders@vibrantlabs.com](mailto:founders@vibrantlabs.com)。

## 深入理解 { #a-deeper-look }

了解基本用法后，接下来看看测试集生成流水线的主要组件，以及如何按需定制。

测试集生成的核心包含两项操作。

1. **构建知识图谱**。首先根据文档创建[知识图谱][ragas.testset.graph.KnowledgeGraph]，再通过各类[变换][ragas.testset.transforms.base.BaseGraphTransformation]为图谱补充信息，用于后续生成测试集。详情参阅[核心概念](../concepts/test_data_generation/rag.md#knowledge-graph-creation)。
2. **生成测试集**。利用[知识图谱][ragas.testset.graph.KnowledgeGraph]生成一组[场景][ragas.testset.synthesizers.base.BaseScenario]，再根据场景生成[测试集][ragas.testset.synthesizers.generate.Testset]。详情参阅[核心概念](../concepts/test_data_generation/rag.md#scenario-generation)。

下面通过示例看看这些组件如何协同生成测试集。

### 构建知识图谱 { #knowledgegraph-creation }

首先，使用先前加载的文档创建[知识图谱][ragas.testset.graph.KnowledgeGraph]。

```python
from ragas.testset.graph import KnowledgeGraph

kg = KnowledgeGraph()
```
输出如下。
```
KnowledgeGraph(nodes: 0, relationships: 0)
```

然后将文档添加到知识图谱中。

```python
from ragas.testset.graph import Node, NodeType

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )
```
输出如下。
```
KnowledgeGraph(nodes: 10, relationships: 0)
```

接下来，通过[变换][ragas.testset.transforms.base.BaseGraphTransformation]为知识图谱补充信息。这里使用 [default_transforms][ragas.testset.transforms.default_transforms] 创建一组默认变换，并指定 LLM 与嵌入模型。
你也可以按需组合变换，或创建自己的变换。

```python
from ragas.testset.transforms import default_transforms, apply_transforms


# define your LLM and Embedding Model
# here we are using the same LLM and Embedding Model that we used to generate the testset
transformer_llm = generator_llm
embedding_model = generator_embeddings

trans = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
apply_transforms(kg, trans)
```

现在知识图谱已补充了相关信息，可以将其保存。

```python
kg.save("knowledge_graph.json")
loaded_kg = KnowledgeGraph.load("knowledge_graph.json")
loaded_kg
```

输出如下。
```
KnowledgeGraph(nodes: 48, relationships: 605)
```

### 生成测试集 { #testset-generation }

使用 `loaded_kg` 创建 [TestsetGenerator][ragas.testset.synthesizers.generate.TestsetGenerator]。

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=loaded_kg)
```

还可以指定要生成的查询分布。这里采用默认分布。

```python
from ragas.testset.synthesizers import default_query_distribution

query_distribution = default_query_distribution(generator_llm)
```

输出如下。
```
[
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
]
```

现在可以生成测试集。

```python
testset = generator.generate(testset_size=10, query_distribution=query_distribution)
testset.to_pandas()
```
输出如下。
![测试集](../../getstarted/testset_output.png)
