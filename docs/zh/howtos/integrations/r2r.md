# R2R 集成 { #r2r-integration }

R2R 是具备生产功能的一体化 RAG 方案，支持多模态内容接入、混合检索、用户和文档管理等。

## 概览 { #overview }
本教程将完成以下工作。

- 使用 R2R 的 `/rag` 端点，在小型数据集上执行检索增强生成
- 评估生成回答
- 分析评估追踪

## 配置 R2R { #r2r-setup }

#### 安装依赖 { #installing-the-dependencies }

先安装所需软件包。


```python
%pip install r2r -q
```

#### 配置本地环境 { #setting-up-the-local-environment }

配置 `R2R_API_KEY`、`OPENAI_API_KEY` 和可选的 `RAGAS_APP_TOKEN`。


```python
from dotenv import load_dotenv

load_dotenv()
```

#### 获取数据 { #getting-the-data }


```python
dataset = [
    "OpenAI is one of the most recognized names in the large language model space, known for its GPT series of models. These models excel at generating human-like text and performing tasks like creative writing, answering questions, and summarizing content. GPT-4, their latest release, has set benchmarks in understanding context and delivering detailed responses.",
    "Anthropic is well-known for its Claude series of language models, designed with a strong focus on safety and ethical AI behavior. Claude is particularly praised for its ability to follow complex instructions and generate text that aligns closely with user intent.",
    "DeepMind, a division of Google, is recognized for its cutting-edge Gemini models, which are integrated into various Google products like Bard and Workspace tools. These models are renowned for their conversational abilities and their capacity to handle complex, multi-turn dialogues.",
    "Meta AI is best known for its LLaMA (Large Language Model Meta AI) series, which has been made open-source for researchers and developers. LLaMA models are praised for their ability to support innovation and experimentation due to their accessibility and strong performance.",
    "Meta AI with it's LLaMA models aims to democratize AI development by making high-quality models available for free, fostering collaboration across industries. Their open-source approach has been a game-changer for researchers without access to expensive resources.",
    "Microsoft’s Azure AI platform is famous for integrating OpenAI’s GPT models, enabling businesses to use these advanced models in a scalable and secure cloud environment. Azure AI powers applications like Copilot in Office 365, helping users draft emails, generate summaries, and more.",
    "Amazon’s Bedrock platform is recognized for providing access to various language models, including its own models and third-party ones like Anthropic’s Claude and AI21’s Jurassic. Bedrock is especially valued for its flexibility, allowing users to choose models based on their specific needs.",
    "Cohere is well-known for its language models tailored for business use, excelling in tasks like search, summarization, and customer support. Their models are recognized for being efficient, cost-effective, and easy to integrate into workflows.",
    "AI21 Labs is famous for its Jurassic series of language models, which are highly versatile and capable of handling tasks like content creation and code generation. The Jurassic models stand out for their natural language understanding and ability to generate detailed and coherent responses.",
    "In the rapidly advancing field of artificial intelligence, several companies have made significant contributions with their large language models. Notable players include OpenAI, known for its GPT Series (including GPT-4); Anthropic, which offers the Claude Series; Google DeepMind with its Gemini Models; Meta AI, recognized for its LLaMA Series; Microsoft Azure AI, which integrates OpenAI’s GPT Models; Amazon AWS (Bedrock), providing access to various models including Claude (Anthropic) and Jurassic (AI21 Labs); Cohere, which offers its own models tailored for business use; and AI21 Labs, known for its Jurassic Series. These companies are shaping the landscape of AI by providing powerful models with diverse capabilities.",
]
```

#### 配置 R2R 客户端 { #setting-up-the-r2r-client }


```python
from r2r import R2RClient

client = R2RClient()
```

#### 接入数据 { #ingesting-the-data }


```python
ingest_response = client.documents.create(
    chunks=dataset,
)
```

#### 使用 `/rag` 端点 { #using-the-rag-endpoint }

[`/rag`](https://r2r-docs.sciphi.ai/api-and-sdks/retrieval/rag-app) 将检索结果与语言模型输出结合。通过 `rag_generation_config` 配置生成过程，通过 `search_settings` 配置检索过程。


```python
query = "What makes Meta AI’s LLaMA models stand out?"

search_settings = {
        "limit": 2,
        "graph_settings": {"enabled": False, "limit": 2},
    }

response = client.retrieval.rag(
    query=query,
    search_settings=search_settings
)

print(response.results.generated_answer)
```
输出如下。
```
Meta AI’s LLaMA models stand out due to their open-source nature, which supports innovation and experimentation by making high-quality models accessible to researchers and developers [1]. This approach democratizes AI development, fostering collaboration across industries and enabling researchers without access to expensive resources to work with advanced AI models [2].
```

## 评估 { #evaluations }

#### **使用 Ragas 评估 R2R 客户端** { #evaluating-the-r2r-client-with-ragas }

配置客户端后，通过 Ragas 的 r2r 集成评估，需要以下组件。

- **1. R2R 客户端与配置**
`R2RClient` 及指定 RAG 行为的 `/rag` 配置。

- **2. 评估数据集**
包含所选指标全部必要输入的 Ragas `EvaluationDataset`。

- **3. Ragas 指标**
指标涵盖忠实度、回答相关性和上下文召回率等维度，完整列表见 [Ragas 文档](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)。


#### 构建 Ragas EvaluationDataset { #constructing-a-ragas-evaluationdataset }
[`EvaluationDataset`](../../concepts/components/eval_dataset.md) 表示一组评估样本，结构与用法见[核心概念](../../concepts/components/eval_dataset.md)。

通过 Ragas 的 `transform_to_ragas_dataset` 转换数据。

```python
questions = [
    "Who are the major players in the large language model space?",
    "What is Microsoft’s Azure AI platform known for?",
    "What kind of models does Cohere provide?",
]

references = [
    "The major players include OpenAI (GPT Series), Anthropic (Claude Series), Google DeepMind (Gemini Models), Meta AI (LLaMA Series), Microsoft Azure AI (integrating GPT Models), Amazon AWS (Bedrock with Claude and Jurassic), Cohere (business-focused models), and AI21 Labs (Jurassic Series).",
    "Microsoft’s Azure AI platform is known for integrating OpenAI’s GPT models, enabling businesses to use these models in a scalable and secure cloud environment.",
    "Cohere provides language models tailored for business use, excelling in tasks like search, summarization, and customer support.",
]

r2r_responses = []

search_settings = {
    "limit": 2,
    "graph_settings": {"enabled": False, "limit": 2},
}

for que in questions:
    response = client.retrieval.rag(query=que, search_settings=search_settings)
    r2r_responses.append(response)
```


```python
from ragas.integrations.r2r import transform_to_ragas_dataset

ragas_eval_dataset = transform_to_ragas_dataset(
    user_inputs=questions, r2r_responses=r2r_responses, references=references
)
```
输出如下。
```
EvaluationDataset(features=['user_input', 'retrieved_contexts', 'response', 'reference'], len=3)
```


#### 选择指标 { #selecting-the-metrics }

这里使用以下指标评估 RAG 端点。

- [回答相关性](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/#response-relevancy)，衡量回答与用户查询的相关程度。
- [上下文精确率](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)，衡量相关检索片段是否优先排列。
- [忠实度](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)，衡量回答与检索上下文在事实层面的一致性。


```python
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
from ragas import evaluate
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = LangchainLLMWrapper(llm)

ragas_metrics = [AnswerRelevancy(llm=evaluator_llm), ContextPrecision(llm=evaluator_llm), Faithfulness(llm=evaluator_llm)]

results = evaluate(dataset=ragas_eval_dataset, metrics=ragas_metrics)
```
输出如下。
```
Querying Client: 100%|██████████| 3/3 [00:00<?, ?it/s]

Evaluating: 100%|██████████| 9/9 [00:00<?, ?it/s]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>answer_relevancy</th>
      <th>context_precision</th>
      <th>faithfulness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Who are the major players in the large languag...</td>
      <td>[In the rapidly advancing field of artificial ...</td>
      <td>The major players in the large language model ...</td>
      <td>The major players include OpenAI (GPT Series),...</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is Microsoft’s Azure AI platform known for?</td>
      <td>[Microsoft’s Azure AI platform is famous for i...</td>
      <td>Microsoft’s Azure AI platform is known for int...</td>
      <td>Microsoft’s Azure AI platform is known for int...</td>
      <td>0.948908</td>
      <td>1.0</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What kind of models does Cohere provide?</td>
      <td>[Cohere is well-known for its language models ...</td>
      <td>Cohere provides language models tailored for b...</td>
      <td>Cohere provides language models tailored for b...</td>
      <td>0.903765</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### 跟踪评估 { #tracing-the-evaluations }

通过以下代码获取追踪与判定理由，进一步理解得分。


```python
results.upload()
```
![](../../../_static/r2r_integration_ragas_app.png)

祝编码顺利。
