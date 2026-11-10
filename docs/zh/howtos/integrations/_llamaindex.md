# LlamaIndex

[LlamaIndex](https://github.com/run-llama/llama_index) 是 LLM 应用的数据框架，用于接入、组织和访问私有或领域数据。要为自己的数据选择最佳配置，需要客观衡量表现。Ragas 可以评估 `QueryEngine`，为调整配置和改进得分提供依据。

本指南假设你已熟悉 LlamaIndex。

## 构建测试集 { #building-the-testset }

评估 `QueryEngine` 需要测试集，可以自行创建，也可以使用 Ragas 的[测试集生成模块](../../getstarted/rag_testset_generation.md)，先生成一个小型合成数据集。

下面介绍如何结合 LlamaIndex 使用。

## 加载文档 { #load-the-documents }


```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./nyc_wikipedia").load_data()
```

配置相应的生成与评判 LLM，初始化 `TestsetGenerator`。


```python
from ragas.testset import TestsetGenerator

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# generator with openai models
generator_llm = OpenAI(model="gpt-4o")
embeddings = OpenAIEmbedding(model="text-embedding-3-large")

generator = TestsetGenerator.from_llama_index(
    llm=generator_llm,
    embedding_model=embeddings,
)
```

现在可以生成数据集。


```python
# generate testset
testset = generator.generate_with_llamaindex_docs(
    documents,
    testset_size=5,
)
```


```python
df = testset.to_pandas()
df.head()
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
      <th>reference_contexts</th>
      <th>reference</th>
      <th>synthesizer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cud yu pleese explane the role of New York Cit...</td>
      <td>[New York, often called New York City or NYC, ...</td>
      <td>New York City serves as the geographical and d...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>So like, what was New York City called before ...</td>
      <td>[History == === Early history === In the pre-C...</td>
      <td>Before it was called New York, the area was kn...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>what happen in new york with slavery and how i...</td>
      <td>[and rechristened it "New Orange" after Willia...</td>
      <td>In the early 18th century, New York became a c...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What historical significance does Long Island ...</td>
      <td>[&lt;1-hop&gt;\n\nHistory == === Early history === I...</td>
      <td>Long Island holds historical significance in t...</td>
      <td>multi_hop_specific_query_synthesizer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What role does the Staten Island Ferry play in...</td>
      <td>[&lt;1-hop&gt;\n\nto start service in 2017; this wou...</td>
      <td>The Staten Island Ferry plays a significant ro...</td>
      <td>multi_hop_specific_query_synthesizer</td>
    </tr>
  </tbody>
</table>
</div>



有了测试集，接下来构建并评估 `QueryEngine`。

## 构建 `QueryEngine` { #building-the-queryengine }

以[纽约市维基百科页面](https://en.wikipedia.org/wiki/New_York_City)为例，构建 `VectorStoreIndex` 并使用 Ragas 评估。

直接使用先前加载到 `documents` 的文档。


```python
# build query engine
from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents)

query_engine = vector_index.as_query_engine()
```

从生成的测试集中选一个问题，检查查询引擎是否正常。


```python
# convert it to pandas dataset
df = testset.to_pandas()
df["user_input"][0]
```




    'Cud yu pleese explane the role of New York City within the Northeast megalopolis, and how it contributes to the cultural and economic vibrancy of the region?'




```python
response_vector = query_engine.query(df["user_input"][0])

print(response_vector)
```

    New York City serves as a key hub within the Northeast megalopolis, playing a significant role in enhancing the cultural and economic vibrancy of the region. Its status as a global center of creativity, entrepreneurship, and cultural diversity contributes to the overall dynamism of the area. The city's renowned arts scene, including Broadway theatre and numerous cultural institutions, attracts artists and audiences from around the world, enriching the cultural landscape of the Northeast megalopolis. Economically, New York City's position as a leading financial and fintech center, home to major stock exchanges and a bustling real estate market, bolsters the region's economic strength and influence. Additionally, the city's diverse culinary scene, influenced by its immigrant history, adds to the cultural richness of the region, making New York City a vital component of the Northeast megalopolis's cultural and economic tapestry.


## 评估 `QueryEngine` { #evaluating-the-queryengine }

为 `VectorStoreIndex` 创建查询引擎后，使用 Ragas 的 llama_index 集成评估。

需要准备以下三项。

1. LlamaIndex `QueryEngine`，被评估的对象
2. 指标，衡量查询引擎的不同维度，完整列表与含义见[指标文档](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
3. 问题，用于测试查询引擎的问题列表

理想情况下，应使用生产环境中真实出现的问题，使评估分布贴近实际使用，得分更能反映生产表现。这里先使用几个示例问题。

导入评估所需指标。


```python
# import metrics
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# init metrics with evaluator LLM
from ragas.llms import LlamaIndexLLMWrapper

evaluator_llm = LlamaIndexLLMWrapper(OpenAI(model="gpt-4o"))
metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]
```

`evaluate()` 需要包含 question 与 ground_truth 的数据，可以方便地将 `testset` 转为所需格式。


```python
# convert to Ragas Evaluation Dataset
ragas_dataset = testset.to_evaluation_dataset()
ragas_dataset
```




    EvaluationDataset(features=['user_input', 'reference_contexts', 'reference'], len=6)



最后运行评估。


```python
from ragas.integrations.llama_index import evaluate

result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=ragas_dataset,
)
```


```python
# final scores
print(result)
```

    {'faithfulness': 0.7454, 'answer_relevancy': 0.9348, 'context_precision': 0.6667, 'context_recall': 0.4667}


可以将结果转换为 pandas DataFrame，继续分析。


```python
result.to_pandas()
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
      <th>reference_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_precision</th>
      <th>context_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cud yu pleese explane the role of New York Cit...</td>
      <td>[and its ideals of liberty and peace. In the 2...</td>
      <td>[New York, often called New York City or NYC, ...</td>
      <td>New York City plays a significant role within ...</td>
      <td>New York City serves as the geographical and d...</td>
      <td>0.615385</td>
      <td>0.918217</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>So like, what was New York City called before ...</td>
      <td>[New York City is the headquarters of the glob...</td>
      <td>[History == === Early history === In the pre-C...</td>
      <td>New York City was named New Amsterdam before i...</td>
      <td>Before it was called New York, the area was kn...</td>
      <td>1.000000</td>
      <td>0.967821</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>what happen in new york with slavery and how i...</td>
      <td>[=== Province of New York and slavery ===\n\nI...</td>
      <td>[and rechristened it "New Orange" after Willia...</td>
      <td>Slavery became a significant part of New York'...</td>
      <td>In the early 18th century, New York became a c...</td>
      <td>1.000000</td>
      <td>0.919264</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What historical significance does Long Island ...</td>
      <td>[==== River crossings ====\n\nNew York City is...</td>
      <td>[&lt;1-hop&gt;\n\nHistory == === Early history === I...</td>
      <td>Long Island played a significant role in the e...</td>
      <td>Long Island holds historical significance in t...</td>
      <td>0.500000</td>
      <td>0.931895</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What role does the Staten Island Ferry play in...</td>
      <td>[==== Buses ====\n\nNew York City's public bus...</td>
      <td>[&lt;1-hop&gt;\n\nto start service in 2017; this wou...</td>
      <td>The Staten Island Ferry serves as a vital mode...</td>
      <td>The Staten Island Ferry plays a significant ro...</td>
      <td>0.500000</td>
      <td>0.936920</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>How does Central Park's role as a cultural and...</td>
      <td>[==== State parks ====\n\nThere are seven stat...</td>
      <td>[&lt;1-hop&gt;\n\nCity has over 28,000 acres (110 km...</td>
      <td>Central Park's role as a cultural and historic...</td>
      <td>Central Park, located in middle-upper Manhatta...</td>
      <td>0.857143</td>
      <td>0.934841</td>
      <td>1.0</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>


