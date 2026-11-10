## 忠实度 { #faithfulness }

**忠实度（Faithfulness）**衡量 `response` 中的事实与检索上下文（`retrieved context`）是否一致，取值为 0～1，越高表示一致性越好。

如果回答中的全部陈述都能得到检索上下文的支持，就认为回答是**忠实的**。

计算步骤如下。
1. 提取回答中的全部陈述。
2. 逐一检查这些陈述能否从检索上下文中推断出来。
3. 按以下公式计算忠实度得分。

$$
\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}
$$


### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = Faithfulness(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)
print(f"Faithfulness Score: {result.value}")
```

输出如下。

```
Faithfulness Score: 1.0
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[...]
    )
    ```


### 计算过程 { #how-its-calculated }

!!! example
    **问题**：爱因斯坦出生于何时何地？

    **上下文**：阿尔伯特·爱因斯坦生于 1879 年 3 月 14 日，是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大、最有影响力的科学家之一。

    **高忠实度回答**：爱因斯坦于 1879 年 3 月 14 日出生在德国。

    **低忠实度回答**：爱因斯坦于 1879 年 3 月 20 日出生在德国。

以下以低忠实度回答为例，说明计算过程。

- **第一步**，将生成的答案拆分为独立陈述。
    - 陈述如下。
        - 陈述 1，爱因斯坦出生在德国。
        - 陈述 2，爱因斯坦出生于 1879 年 3 月 20 日。

- **第二步**，检查每条陈述能否从给定上下文中推断出来。
    - 陈述 1，可以
    - 陈述 2，不可以

- **第三步**，使用上述公式计算忠实度。

    $$
    \text{Faithfulness} = { \text{1} \over \text{2} } = 0.5
    $$


## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
scorer = Faithfulness(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

输出如下。

```
1.0
```

### 使用 HHEM-2.1-Open 计算忠实度 { #faithfulness-with-hhem-21-open }

[Vectara 的 HHEM-2.1-Open](https://vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model/) 是一个基于 T5 的分类模型，专门检测 LLM 生成文本中的幻觉。它可用于忠实度计算的第二步，判断陈述是否能够从给定上下文中推断出来。该模型免费、体积小且开源，适合在生产环境中高效使用。

使用该模型计算忠实度的方法如下。

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM


sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

可以通过 `device` 参数指定模型加载设备，通过 `batch_size` 调整推理批量大小。默认使用 CPU，批量大小为 10。

```python
my_device = "cuda:0"
my_batch_size = 10

scorer = FaithfulnesswithHHEM(device=my_device, batch_size=my_batch_size)
await scorer.single_turn_ascore(sample)
```
