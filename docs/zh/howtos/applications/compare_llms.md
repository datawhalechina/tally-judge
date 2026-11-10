---
search:
  exclude: true
---

# 使用 Ragas 评估比较 LLM { #compare-llms-using-ragas-evaluations }

RAG 系统所用的 LLM 对输出质量影响显著。评估不同 LLM 生成的结果，可以帮助选择适合具体场景的模型。

本教程逐步介绍如何使用 Ragas，基于自己的数据比较并选择 LLM。

<figure markdown="span">
![比较 LLM](../../../_static/imgs/compare-llms-front.jpeg){width="800"}
<figcaption>比较 LLM</figcaption>
</figure>



## 创建合成测试数据 { #create-synthetic-test-data }


!!! tip
    Ragas 也支持已有数据集。参阅[数据准备](../../concepts/components/eval_dataset.md)，了解如何使用自己的数据。

Ragas 的测试生成方法能够根据检索与生成任务构建专属评估数据集，并从文档语料中生成多种具有挑战性的测试用例。

!!! tip
    工作原理参阅[测试集生成](../../concepts/test_data_generation/index.md)。

本教程使用 arXiv 上与大语言模型相关的论文构建 RAG。

!!! note
    建议使用测试集生成器生成 50 条以上样本，以获得更可靠的结果。

```python
import os
from llama_index import download_loader, SimpleDirectoryReader
from ragas.testset import TestsetGenerator
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = 'Your OPEN AI key'

# load documents
reader = SimpleDirectoryReader("./arxiv-papers/",num_files_limit=30)
documents = reader.load_data()

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

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
testset = generator.generate_with_llama_index_docs(documents, 100,distributions)
testset.to_pandas()
```

![测试集输出](../../../_static/imgs/compare-llms-testset.png){width="800" height="600"}

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
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceInferenceAPI
from llama_index.embeddings import HuggingFaceInferenceAPIEmbedding
import pandas as pd

nest_asyncio.apply()


def build_query_engine(llm):
    vector_index = VectorStoreIndex.from_documents(
        documents, service_context=ServiceContext.from_defaults(chunk_size=512, llm=llm),
        embed_model=HuggingFaceInferenceAPIEmbedding,
    )

    query_engine = vector_index.as_query_engine(similarity_top_k=2)
    return query_engine

# Function to evaluate as Llama index does not support async evaluation for HFInference API
def generate_responses(query_engine, test_questions, test_answers):
  responses = [query_engine.query(q) for q in test_questions]

  answers = []
  contexts = []
  for r in responses:
    answers.append(r.response)
    contexts.append([c.node.get_content() for c in r.source_nodes])
  dataset_dict = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
  }
  if test_answers is not None:
    dataset_dict["ground_truth"] = test_answers
  ds = Dataset.from_dict(dataset_dict)
  return ds
```

## 从 Ragas 导入指标 { #import-metrics-from-ragas }

导入评估检索组件所需的指标。

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
)

metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
]
```

## 评估 Zephyr 7B Alpha { #evaluate-zephyr-7b-alpha-llm }

第一个模型使用 HuggingFace 的 [zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)，通过 HuggingFaceInferenceAPI 生成答案。示例使用其免费接口，并通过 [HuggingFace Token](https://huggingface.co/docs/hub/security-tokens) 配置访问凭据。

```python
# Use zephyr model using HFInference API
zephyr_llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    token="Your Hugging Face token"
)
query_engine1 = build_query_engine(zephyr_llm)
result_ds = generate_responses(query_engine1, test_questions, test_answers)
result_zephyr = evaluate(
    result_ds,
    metrics=metrics,
)

result_zephyr
```

```python
{'faithfulness': 0.8365, 'answer_relevancy': 0.8831, 'answer_correctness': 0.6605}
```

## 评估 Falcon-7B-Instruct { #evaluate-falcon-7b-instruct-llm }
第二个模型使用 [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)，同样通过 HuggingFaceInferenceAPI 调用。

```python
falcon_llm = HuggingFaceInferenceAPI(
    model_name="tiiuae/falcon-7b-instruct",
    token="Your Huggingface token"
)
query_engine2 = build_query_engine(falcon_llm)
result_ds_falcon = generate_responses(query_engine2, test_questions, test_answers)
result = evaluate(
    result_ds_falcon,
    metrics=metrics,
)

result
```

```python
{'faithfulness': 0.6909, 'answer_relevancy': 0.8651, 'answer_correctness': 0.5850}
```

## 比较得分 { #compare-scores }

在本例的数据集与 RAG 流水线上，HuggingFace zephyr-7b-alpha 的 `faithfulness`、`answer_correctness` 和 `answer_relevancy` 得分略高于 falcon-7b-instruct。

完整 Colab notebook 见[此链接](https://colab.research.google.com/drive/10dNeU56XLOGUJ9gRuBFryyRwoy70rIeS?usp=sharing)。

```python
import numpy as np
import matplotlib.pyplot as plt

def analysis(zephyr_df, falcon_df):
  sns.set_style("whitegrid")
  fig, axs = plt.subplots(1,3, figsize=(12, 5))
  for i,col in enumerate(zephyr_df.columns):
    sns.kdeplot(data=[zephyr_df[col].values,falcon_df[col].values],legend=False,ax=axs[i],fill=True)
    axs[i].set_title(f'{col} scores distribution')
    axs[i].legend(labels=["zephyr", "falcon"])
  plt.tight_layout()
  plt.show()

result_zephyr_df = result_zephyr.to_pandas()
result_falcon_df = result.to_pandas()
analysis(
    result_zephyr_df[['faithfulness', 'answer_relevancy', 'answer_correctness']],
    result_falcon_df[['faithfulness', 'answer_relevancy', 'answer_correctness']]
)
```

### 得分分布分析 { #score-distribution-analysis }

<figure markdown="span">
![比较 LLM](../../../_static/imgs/compare-llm-result.png){width="800"}
<figcaption>比较 LLM</figcaption>
</figure>
