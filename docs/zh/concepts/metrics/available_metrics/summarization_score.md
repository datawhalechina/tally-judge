# 任务评估指标 { #tasks-metrics }

## 摘要得分 { #summarization-score }

**摘要得分（Summarization Score）**衡量摘要（`response`）是否提取了 `reference_contexts` 中的重要信息。其基本思路是，一份好摘要应包含上下文中的全部关键信息。

首先从上下文提取关键短语，再据此生成一组问题。根据原始上下文，这些问题的答案都应为 `yes(1)`。然后仅依据摘要回答同一组问题，以答对的问题数占问题总数的比例衡量摘要质量。

问答结果表示为由 `1` 和 `0` 组成的列表。问答得分等于正确回答（答案为 `1`）的数量除以问题总数。

$$
\text{QA score} = \frac{|\text{correctly answered questions}|}{|\text{total questions}|}
$$

还可以启用简洁性得分，对过长摘要进行惩罚。启用后，最终分数为问答得分与简洁性得分的加权平均。这样可以避免直接照抄原文的摘要仅因答对全部问题而获得高分。

$$
\text{conciseness score} = 1 - \frac{\min(\text{length of summary}, \text{length of context})}{\text{length of context} + \text{1e-10}}
$$

权重系数 `coeff` 用于控制两项分数的权重，默认值为 0.5。

最终摘要得分计算如下。

$$
\text{Summarization Score} = \text{QA score}*\text{(1-coeff)} + \\
\text{conciseness score}*\text{coeff}
$$

### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import SummaryScore

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = SummaryScore(llm=llm)

# Evaluate
result = await scorer.ascore(
    reference_contexts=[
        "A company is launching a new product, a smartphone app designed to help users track their fitness goals. The app allows users to set daily exercise targets, log their meals, and track their water intake. It also provides personalized workout recommendations and sends motivational reminders throughout the day."
    ],
    response="A company is launching a fitness tracking app that helps users set exercise goals, log meals, and track water intake, with personalized workout suggestions and motivational reminders."
)
print(f"Summary Score: {result.value}")
```

输出如下。

```
Summary Score: 0.6423387096775146
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference_contexts=[...],
        response="..."
    )
    ```


## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SummarizationScore


sample = SingleTurnSample(
    response="A company is launching a fitness tracking app that helps users set exercise goals, log meals, and track water intake, with personalized workout suggestions and motivational reminders.",
    reference_contexts=[
        "A company is launching a new product, a smartphone app designed to help users track their fitness goals. The app allows users to set daily exercise targets, log their meals, and track their water intake. It also provides personalized workout recommendations and sends motivational reminders throughout the day."
    ]
)

scorer = SummarizationScore(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.6423387096775146
```
