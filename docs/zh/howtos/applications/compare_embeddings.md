---
search:
  exclude: true
---

# 比较检索器的嵌入模型 { #compare-embeddings-for-retriever }

检索器表现是影响 RAG 整体效果的关键因素，嵌入模型的质量尤其影响检索内容的质量。

本教程逐步介绍如何通过 Ragas，基于自己的数据比较并选择嵌入模型。

<figure markdown="span">
![比较嵌入模型](../../../_static/imgs/compare-embeddings.jpeg){width="600"}
<figcaption>比较嵌入模型</figcaption>
</figure>

## 创建合成测试数据 { #create-synthetic-test-data }


!!! tip
    Ragas 也支持已有数据集，参阅[数据准备](../customizations/testgenerator/index.md)。

Ragas 的测试生成方法能够根据检索与生成任务构建专属评估数据集，并从文档语料中生成多种具有挑战性的测试用例。

!!! tip
    工作原理参阅[测试集生成](../../getstarted/rag_testset_generation.md)。

本教程使用 Semantic Scholar 上与大语言模型相关的论文构建 RAG。

```python
from llama_index.core import download_loader
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai

SemanticScholarReader = download_loader("SemanticScholarReader")
loader = SemanticScholarReader()
query_space = "large language models"
documents = loader.load_data(query=query_space, limit=100)

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o")
openai_client = openai.OpenAI()
embeddings = OpenAIEmbeddings(client=openai_client)

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)


distributions = {
    simple: 0.5,
    multi_context: 0.4,
    reasoning: 0.1
}

# generate testset
testset = generator.generate_with_llamaindex_docs(documents, 100,distributions)
test_df = testset.to_pandas()
```

<figure markdown="span">
![测试集输出](../../../_static/imgs/testset_output.png){width="800"}
<figcaption>测试输出</figcaption>
</figure>

```python
test_questions = test_df['question'].values.tolist()
test_answers = [[item] for item in test_df['answer'].values.tolist()]
```


## 构建 RAG { #build-your-rag }

这里使用 llama-index 和文档构建基础 RAG 流水线，收集每个测试问题的检索上下文与生成答案。Ragas 与多个 RAG 框架集成，便于开展评估。

!!! note
    使用 LangChain 评估的方法见 [LangChain 教程](../integrations/_langchain.md)。

```python

import nest_asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import OpenAIEmbeddings
import openai
import pandas as pd

nest_asyncio.apply()


def build_query_engine(embed_model):
    vector_index = VectorStoreIndex.from_documents(
        documents, service_context=ServiceContext.from_defaults(chunk_size=512),
        embed_model=embed_model,
    )

    query_engine = vector_index.as_query_engine(similarity_top_k=2)
    return query_engine
```

## 从 Ragas 导入指标 { #import-metrics-from-ragas }

导入评估检索组件所需的指标。

```python
from ragas.metrics import (
    context_precision,
    context_recall,
)

metrics = [
    context_precision,
    context_recall,
]
```

## 评估 OpenAI 嵌入模型 { #evaluate-openai-embeddings }

```python
from ragas.llama_index import evaluate

openai_model = OpenAIEmbedding()
query_engine1 = build_query_engine(openai_model)
result = evaluate(query_engine1, metrics, test_questions, test_answers)
```

```python
{'context_precision': 0.2378, 'context_recall': 0.7159}
```

## 评估 BGE 嵌入模型 { #evaluate-bge-embeddings }

```python
from ragas.llama_index import evaluate

flag_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
query_engine2 = build_query_engine(flag_model)
result = evaluate(query_engine2, metrics, test_questions, test_answers)
```

```python
{'context_precision': 0.2655, 'context_recall': 0.7227}

```

## 比较得分 { #compare-scores }

在本例的数据集与 RAG 流水线上，BGE 模型的 `context_precision` 和 `context_recall` 得分略高于 OpenAI Ada 模型。

可以将结果导出为 pandas 对象，进一步分析得分。

```python
result_df = result.to_pandas()
result_df.head()
```

<figure markdown="span">
![嵌入模型比较结果](../../../_static/imgs/compare-emb-results.png){width="800"}
<figcaption>嵌入模型比较结果</figcaption>
</figure>
