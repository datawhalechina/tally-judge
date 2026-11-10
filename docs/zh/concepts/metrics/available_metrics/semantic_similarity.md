## 语义相似度 { #semantic-similarity }

**语义相似度（Semantic Similarity）**衡量生成回答与参考答案在语义上的接近程度，得分为 0～1，越高表示越接近。

此指标基于嵌入向量与余弦相似度，衡量两个答案的语义相似程度，为分析生成回答的质量提供依据。


### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import SemanticSimilarity

# Setup embeddings
client = AsyncOpenAI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)

# Create metric
scorer = SemanticSimilarity(embeddings=embeddings)

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris. It has a height of 1000ft.",
    response="The Eiffel Tower is located in Paris."
)
print(f"Semantic Similarity Score: {result.value}")
```

输出如下。

```
Semantic Similarity Score: 0.8151
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris. It has a height of 1000ft.",
        response="The Eiffel Tower is located in Paris."
    )
    ```


### 计算过程 { #how-its-calculated }

!!! example

    **参考答案**：阿尔伯特·爱因斯坦的相对论彻底改变了我们对宇宙的理解。

    **高相似度回答**：爱因斯坦开创性的相对论改变了我们对宇宙的认识。

    **低相似度回答**：艾萨克·牛顿的运动定律深刻影响了经典物理学。

以高相似度回答为例，计算过程如下。

- **第一步**，使用指定的嵌入模型将参考答案向量化。
- **第二步**，使用同一个嵌入模型将生成回答向量化。
- **第三步**，计算两个向量的余弦相似度。
- **第四步**，以余弦相似度（0～1）作为最终得分。


## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper

sample = SingleTurnSample(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. It has a height of 1000ft."
)

scorer = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(evaluator_embedding))
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.8151371879226978
```
