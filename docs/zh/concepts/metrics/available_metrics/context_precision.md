# 上下文精确率 { #context-precision }

上下文精确率衡量检索器能否针对给定查询，将相关片段排在无关片段之前。它关注检索上下文中的相关片段是否位于排序靠前的位置。

计算时，取每个相关片段所在位置的 precision@k，再求平均。precision@k 表示前 k 个片段中相关片段的数量占这 k 个片段的比例。

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}}
$$

$$
\text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})}
$$

其中，$K$ 是 `retrieved_contexts` 中的片段总数，$v_k \in \{0, 1\}$ 表示第 $k$ 个片段是否相关。

## 示例 { #examples }

### 上下文精确率 { #context-precision_1 }

`ContextPrecision` 将每个检索上下文与参考答案比较，判断它是否有助于回答问题，适用于已有参考答案的场景。

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextPrecision(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris.",
        "The Brandenburg Gate is located in Berlin."
    ]
)
print(f"Context Precision Score: {result.value}")
```

输出如下。
```
Context Precision Score: 0.9999999999
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[...]
    )
    ```

### 上下文利用度 { #context-utilization }

`ContextUtilization` 将每个检索上下文与生成的回答比较，判断上下文是否有用，适用于没有参考答案、但已有系统回答的场景。

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextUtilization

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextUtilization(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris.",
        "The Brandenburg Gate is located in Berlin."
    ]
)
print(f"Context Utilization Score: {result.value}")
```

输出如下。
```
Context Utilization Score: 0.9999999999
```

即使第二个位置存在无关片段，上下文精确率也不会变化。但是，如果将无关片段放在第一个位置，得分就会降低。

```python
result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "The Brandenburg Gate is located in Berlin.",
        "The Eiffel Tower is located in Paris."
    ]
)
print(f"Context Utilization Score: {result.value}")
```

输出如下。
```
Context Utilization Score: 0.49999999995
```

## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)

await context_precision.single_turn_ascore(sample)
```

输出如下。
```
0.9999999999
```

### 无参考答案的上下文精确率 { #context-precision-without-reference }

`LLMContextPrecisionWithoutReference` 无须参考答案。它使用 LLM 将 `retrieved_contexts` 中的每个片段与 `response` 比较，判断检索上下文是否相关。

#### 示例 { #example }

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)

await context_precision.single_turn_ascore(sample)
```

输出如下。
```
0.9999999999
```

### 有参考答案的上下文精确率 { #context-precision-with-reference }

`LLMContextPrecisionWithReference` 适用于 `user_input` 同时具备检索上下文与参考回答的场景。它使用 LLM 将 `retrieved_contexts` 中的每个片段与 `reference` 比较，判断是否相关。

#### 示例 { #example_1 }

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference

context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)

await context_precision.single_turn_ascore(sample)
```

输出如下。
```
0.9999999999
```

## 不依赖 LLM 的上下文精确率 { #non-llm-based-context-precision }

此指标使用非 LLM 方法，例如[莱文斯坦距离](https://en.wikipedia.org/wiki/Levenshtein_distance)，判断检索上下文是否相关。

### 使用参考上下文的上下文精确率 { #context-precision-with-reference-contexts }

`NonLLMContextPrecisionWithReference` 适用于 `user_input` 同时具备检索上下文和参考上下文的场景。它使用不依赖 LLM 的相似度度量，将 `retrieved_contexts` 中的每个片段与 `reference_contexts` 中的各个上下文比较，判断相关性。

该指标需要安装 rapidfuzz 包，执行 `pip install rapidfuzz` 即可。

#### 示例 { #example_2 }

```python
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference

context_precision = NonLLMContextPrecisionWithReference()

sample = SingleTurnSample(
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)

await context_precision.single_turn_ascore(sample)
```

输出如下。
```
0.9999999999
```

## 基于 ID 的上下文精确率 { #id-based-context-precision }

`IDBasedContextPrecision` 直接比较检索上下文与参考上下文的 ID，高效计算精确率。如果文档有唯一 ID，且希望直接评估检索结果，可以使用此指标。

此指标使用 `retrieved_context_ids` 与 `reference_context_ids` 计算精确率，取值为 0～1，越高越好，支持字符串和整数 ID。

基于 ID 的上下文精确率计算公式如下。

$$ \text{ID-Based Context Precision} = \frac{\text{Number of retrieved context IDs found in reference context IDs}}{\text{Total number of retrieved context IDs}} $$

### 示例 { #example_3 }

```python
from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision

sample = SingleTurnSample(
    retrieved_context_ids=["doc_1", "doc_2", "doc_3", "doc_4"],
    reference_context_ids=["doc_1", "doc_4", "doc_5", "doc_6"]
)

id_precision = IDBasedContextPrecision()
await id_precision.single_turn_ascore(sample)
```

输出如下。
```
0.5
```

本例中，4 个检索上下文 ID 中只有 2 个（`"doc_1"` 与 `"doc_4"`）出现在参考 ID 集合中，因此精确率为 0.5，即 50%。
