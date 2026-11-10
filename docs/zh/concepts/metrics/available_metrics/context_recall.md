# 上下文召回率 { #context-recall }

上下文召回率衡量成功检索到的相关文档或信息的比例，关注重要信息是否遗漏。召回率越高，被遗漏的相关文档越少。

计算上下文召回率需要参考信息作为比较依据。基于 LLM 的上下文召回率使用 `reference` 代替 `reference_contexts`，减少标注参考上下文的时间成本。计算时先将参考答案拆分为多个陈述，再逐一判断它们是否有检索上下文作为依据。理想情况下，参考答案中的每条陈述都应得到检索上下文的支持。

上下文召回率的计算公式如下。

$$
\text{Context Recall} = \frac{\text{Number of claims in the reference supported by the retrieved context}}{\text{Total number of claims in the reference}}
$$

## 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextRecall(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    retrieved_contexts=["Paris is the capital of France."],
    reference="The Eiffel Tower is located in Paris."
)
print(f"Context Recall Score: {result.value}")
```

输出如下。

```
Context Recall Score: 1.0
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        user_input="Where is the Eiffel Tower located?",
        retrieved_contexts=["Paris is the capital of France."],
        reference="The Eiffel Tower is located in Paris."
    )
    ```

## 基于 LLM 的上下文召回率（旧版 API） { #llm-based-context-recall-legacy-api }

!!! warning "旧版 API"
    以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。旧版 API 将在 0.4 版本中弃用，在 1.0 版本中移除。

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["Paris is the capital of France."],
)

context_recall = LLMContextRecall(llm=evaluator_llm)
await context_recall.single_turn_ascore(sample)
```

输出如下。
```
1.0
```

## 不依赖 LLM 的上下文召回率 { #non-llm-based-context-recall }

`NonLLMContextRecall` 使用 `retrieved_contexts` 与 `reference_contexts` 计算得分，取值为 0～1，越高越好。它通过非 LLM 字符串比较指标判断检索上下文是否相关，你可以选择任意此类指标作为距离度量。

上下文召回率的计算公式如下。

$$
\text{context recall} = {|\text{Number of relevant contexts retrieved}| \over |\text{Total number of reference contexts}|}
$$

### 示例 { #example_1 }

```python


from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

sample = SingleTurnSample(
    retrieved_contexts=["Paris is the capital of France."],
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)

context_recall = NonLLMContextRecall()
await context_recall.single_turn_ascore(sample)


```
输出如下。
```
0.5
```

## 基于 ID 的上下文召回率 { #id-basedcontext-recall }

基于 ID 的上下文召回率
`IDBasedContextRecall` 直接比较检索上下文与参考上下文的 ID，高效计算召回率。如果文档有唯一 ID，且希望直接评估检索结果，可以使用此指标。

此指标使用 `retrieved_context_ids` 与 `reference_context_ids` 计算召回率，取值为 0～1，越高越好，支持字符串和整数 ID。

基于 ID 的上下文召回率计算公式如下。

$$ \text{ID-Based Context Recall} = \frac{\text{Number of reference context IDs found in retrieved context IDs}}{\text{Total number of reference context IDs}} $$

### 示例 { #example_2 }

```python

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextRecall

sample = SingleTurnSample(
    retrieved_context_ids=["doc_1", "doc_2", "doc_3"],
    reference_context_ids=["doc_1", "doc_4", "doc_5", "doc_6"]
)

id_recall = IDBasedContextRecall()
await id_recall.single_turn_ascore(sample)
```

输出如下。
```
0.25
```
