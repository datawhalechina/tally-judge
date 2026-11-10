## 回答相关性 { #answer-relevancy }

**回答相关性（Answer Relevancy）**衡量回答与用户输入的相关程度，通常取值为 0～1，越高表示越切合用户输入。

相关的回答应当直接、恰当地回应原始问题。该指标关注回答与问题意图的匹配程度，事实准确性由其他指标衡量。回答不完整或包含多余细节都会降低得分。

此指标使用 `user_input` 与 `response`，按以下步骤计算。

1. 根据回答生成一组问题，默认 3 个，用于反映回答的内容。
2. 计算用户输入的嵌入向量（$E_o$）与各个生成问题的嵌入向量（$E_{g_i}$）之间的余弦相似度。
3. 取这些余弦相似度的均值，作为**回答相关性**得分。

$$
\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine similarity}(E_{g_i}, E_o)
$$

$$
\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \frac{E_{g_i} \cdot E_o}{\|E_{g_i}\| \|E_o\|}
$$

其中，
- $E_{g_i}$ 为第 $i$ 个生成问题的嵌入向量。
- $E_o$ 为用户输入的嵌入向量。
- $N$ 为生成问题的数量，默认 3 个，可通过 `strictness` 参数调整。

**注意**。由于余弦相似度的数学取值范围为 -1～1，实际得分虽然通常介于 0～1 之间，但并不保证始终如此。

### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy

# Setup LLM and embeddings
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

# Create metric
scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967"
)
print(f"Answer Relevancy Score: {result.value}")
```

输出如下。

```
Answer Relevancy Score: 0.9165088378587264
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967"
    )
    ```

### 计算过程 { #how-its-calculated }

!!! example
    问题：法国位于哪里，首都是哪座城市？

    低相关性回答：法国位于西欧。

    高相关性回答：法国位于西欧，首都是巴黎。

评估回答与问题的相关性，可以分为两步。

- **第一步**，使用大语言模型（LLM）从生成答案反推 n 个可能的问题。
  例如，针对第一个回答，LLM 可能生成以下问题。
    - *问题 1*：法国位于欧洲的哪个地区？
    - *问题 2*：法国在欧洲的地理位置是什么？
    - *问题 3*：你能说出法国位于欧洲的哪个地区吗？

- **第二步**，计算生成问题与原始问题之间的平均余弦相似度。

这一方法的基本思路是，如果回答正确回应了问题，就很可能仅根据回答重建出原始问题。


## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy

sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )

scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.9165088378587264
```
