# Griptape 集成 { #griptape-integration }

如果你熟悉 Griptape 的 RAG Engine，希望评估系统表现，本教程将介绍如何使用 Ragas 评价它生成的回答。

## 配置 Griptape { #griptape-setup }

### 配置环境 { #setting-up-our-environment }

先安装所需软件包。


```shell
%pip install "griptape[all]" ragas -q
```

### 创建数据集 { #creating-our-dataset }

使用介绍主要 LLM 服务商的少量文本片段，配置简单的 RAG 流水线。


```python
chunks = [
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

### 将数据写入向量存储 { #ingesting-data-in-vector-store }


```python
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
```


```python
from griptape.drivers.embedding.openai import OpenAiEmbeddingDriver
from griptape.drivers.vector.local import LocalVectorStoreDriver

# Set up a simple vector store with our data
vector_store = LocalVectorStoreDriver(embedding_driver=OpenAiEmbeddingDriver())
vector_store.upsert_collection({"major_llm_providers": chunks})
```

### 配置 RAG Engine { #setting-up-the-rag-engine }


```python
from griptape.engines.rag import RagContext, RagEngine
from griptape.engines.rag.modules import (
    PromptResponseRagModule,
    VectorStoreRetrievalRagModule,
)
from griptape.engines.rag.stages import (
    ResponseRagStage,
    RetrievalRagStage,
)

# Create a basic RAG pipeline
rag_engine = RagEngine(
    # Stage for retrieving relevant chunks
    retrieval_stage=RetrievalRagStage(
        retrieval_modules=[
            VectorStoreRetrievalRagModule(
                name="VectorStore_Retriever",
                vector_store_driver=vector_store,
                query_params={"namespace": "major_llm_providers"},
            ),
        ],
    ),
    # Stage for generating a response
    response_stage=ResponseRagStage(
        response_modules=[
            PromptResponseRagModule(),
        ]
    ),
)
```

### 测试 RAG 流水线 { #testing-our-rag-pipeline }

使用示例查询检查流水线是否正常。


```python
rag_context = RagContext(query="What makes Meta AI’s LLaMA models stand out?")
rag_context = rag_engine.process(rag_context)
rag_context.outputs[0].to_text()
```
输出如下。
```
"Meta AI's LLaMA models stand out for their open-source nature, which makes them accessible to researchers and developers. This accessibility supports innovation and experimentation, allowing for collaboration across industries. By making high-quality models available for free, Meta AI aims to democratize AI development, which has been a game-changer for researchers without access to expensive resources."
```

## Ragas 评估 { #ragas-evaluation }

### 创建 Ragas 评估数据集 { #creating-a-ragas-evaluation-dataset }


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

griptape_rag_contexts = []

for que in questions:
    rag_context = RagContext(query=que)
    griptape_rag_contexts.append(rag_engine.process(rag_context))
```


```python
from ragas.integrations.griptape import transform_to_ragas_dataset

ragas_eval_dataset = transform_to_ragas_dataset(
    grip_tape_rag_contexts=griptape_rag_contexts, references=references
)
```


```python
ragas_eval_dataset.to_pandas()
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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Who are the major players in the large languag...</td>
      <td>[In the rapidly advancing field of artificial ...</td>
      <td>The major players in the large language model ...</td>
      <td>The major players include OpenAI (GPT Series),...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is Microsoft’s Azure AI platform known for?</td>
      <td>[Microsoft’s Azure AI platform is famous for i...</td>
      <td>Microsoft’s Azure AI platform is known for int...</td>
      <td>Microsoft’s Azure AI platform is known for int...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What kind of models does Cohere provide?</td>
      <td>[Cohere is well-known for its language models ...</td>
      <td>Cohere provides language models tailored for b...</td>
      <td>Cohere provides language models tailored for b...</td>
    </tr>
  </tbody>
</table>
</div>



### 运行 Ragas 评估 { #running-the-ragas-evaluation }

接下来使用 Ragas 指标评估系统。

#### 评估检索 { #evaluating-retrieval }

可以使用内置指标，或按需创建自定义指标。完整列表与定制方法参阅[文档]()。

使用 `ContextPrecision`、`ContextRecall` 和 `ContextRelevance` 衡量检索表现。

- [ContextPrecision](../../concepts/metrics/available_metrics/context_precision.md)，衡量相关片段是否排在检索结果前列，根据相关位置的 precision@k 均值计算。
- [ContextRecall](../../concepts/metrics/available_metrics/context_recall.md)，衡量知识库中的相关信息被成功检索的比例。
- [ContextRelevance](../../concepts/metrics/available_metrics/nvidia_metrics.md#context-relevance)，通过两次 LLM 评判，衡量检索上下文与用户查询的相关程度。


```python
from ragas.metrics import ContextPrecision, ContextRecall, ContextRelevance
from ragas import evaluate
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = LangchainLLMWrapper(llm)

ragas_metrics = [
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
    ContextRelevance(llm=evaluator_llm),
]

retrieval_results = evaluate(dataset=ragas_eval_dataset, metrics=ragas_metrics)
retrieval_results.to_pandas()
```
```
Evaluating: 100%|██████████| 9/9 [00:15<00:00,  1.77s/it]
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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>context_precision</th>
      <th>context_recall</th>
      <th>nv_context_relevance</th>
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
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is Microsoft’s Azure AI platform known for?</td>
      <td>[Microsoft’s Azure AI platform is famous for i...</td>
      <td>Microsoft’s Azure AI platform is known for int...</td>
      <td>Microsoft’s Azure AI platform is known for int...</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What kind of models does Cohere provide?</td>
      <td>[Cohere is well-known for its language models ...</td>
      <td>Cohere provides language models tailored for b...</td>
      <td>Cohere provides language models tailored for b...</td>
      <td>0.833333</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 评估生成 { #evaluating-generation }

生成评估使用 `FactualCorrectness`、`Faithfulness` 和上下文相关指标。

- [FactualCorrectness](../../concepts/metrics/available_metrics/factual_correctness.md)，检查回答中的陈述是否得到参考答案支持。
- [Faithfulness](../../concepts/metrics/available_metrics/faithfulness.md)，衡量回答与检索上下文在事实层面的一致性。
- [ResponseGroundedness](../../concepts/metrics/available_metrics/nvidia_metrics.md#response-groundedness)，检查回答是否有上下文依据，帮助识别幻觉与编造信息。


```python
from ragas.metrics import FactualCorrectness, Faithfulness, ResponseGroundedness

ragas_metrics = [
    FactualCorrectness(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    ResponseGroundedness(llm=evaluator_llm),
]

genration_results = evaluate(dataset=ragas_eval_dataset, metrics=ragas_metrics)
genration_results.to_pandas()
```
```
Evaluating: 100%|██████████| 9/9 [00:17<00:00,  1.90s/it]
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
      <th>factual_correctness(mode=f1)</th>
      <th>faithfulness</th>
      <th>nv_response_groundedness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Who are the major players in the large languag...</td>
      <td>[In the rapidly advancing field of artificial ...</td>
      <td>The major players in the large language model ...</td>
      <td>The major players include OpenAI (GPT Series),...</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is Microsoft’s Azure AI platform known for?</td>
      <td>[Microsoft’s Azure AI platform is famous for i...</td>
      <td>Microsoft’s Azure AI platform is known for int...</td>
      <td>Microsoft’s Azure AI platform is known for int...</td>
      <td>0.57</td>
      <td>0.833333</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What kind of models does Cohere provide?</td>
      <td>[Cohere is well-known for its language models ...</td>
      <td>Cohere provides language models tailored for b...</td>
      <td>Cohere provides language models tailored for b...</td>
      <td>0.57</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## 总结 { #conclusion }

现在已为 Griptape RAG 系统配置了 Ragas 评估流水线，可以分析检索相关信息与生成准确回答的能力。

RAG 评估是持续迭代的过程。通过指标定位薄弱环节，改进后重新评估，直到满足目标要求。

祝你顺利构建 RAG！😄
