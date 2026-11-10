# 基于评分细则的评估 { #rubric-based-evaluation }

基于评分细则的指标允许你用自定义标准评估 LLM 回答。Ragas 提供两类此类指标。

1. **DomainSpecificRubrics**，初始化时设置细则，数据集中的所有样本共用
2. **InstanceSpecificRubrics**，每个样本可有独立细则，在评估时分别传入

评分细则为每个分数提供具体描述，通常为 1～5 分。LLM 根据这些描述评估回答并给出分数。

## 领域专属评分细则 { #domain-specific-rubrics }

需要对所有样本采用相同标准时，使用 `DomainSpecificRubrics`，适用于评分标准固定的领域评估。

### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import DomainSpecificRubrics

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Reference-free evaluation (default)
metric = DomainSpecificRubrics(llm=llm)
result = await metric.ascore(
    user_input="What's the longest river in the world?",
    response="The longest river in the world is the Nile, stretching approximately 6,650 kilometers through northeastern Africa.",
)
print(f"Score: {result.value}, Feedback: {result.reason}")

# Reference-based evaluation
metric_with_ref = DomainSpecificRubrics(llm=llm, with_reference=True)
result = await metric_with_ref.ascore(
    user_input="What's the longest river in the world?",
    response="The longest river in the world is the Nile.",
    reference="The Nile is a major north-flowing river in northeastern Africa.",
)
```

### 自定义评分细则 { #custom-rubrics }

可以自行定义细则，定制评分标准。

```python
from ragas.metrics.collections import DomainSpecificRubrics

my_custom_rubrics = {
    "score1_description": "Answer and ground truth are completely different",
    "score2_description": "Answer and ground truth are somewhat different",
    "score3_description": "Answer and ground truth are somewhat similar",
    "score4_description": "Answer and ground truth are similar",
    "score5_description": "Answer and ground truth are exactly the same",
}

metric = DomainSpecificRubrics(llm=llm, rubrics=my_custom_rubrics, with_reference=True)
```

### 结合检索上下文 { #with-retrieved-contexts }

此指标也支持在评估时传入检索上下文。

```python
result = await metric.ascore(
    user_input="What's the longest river in the world?",
    response="Based on the context, the Nile is the longest river.",
    retrieved_contexts=[
        "Scientists debate whether the Amazon or the Nile is the longest river.",
        "The Nile River was central to Ancient Egyptians' wealth and power.",
    ],
)
```

### 便捷类 { #convenience-classes }

可以使用以下便捷类，更清楚地表达用途。

```python
from ragas.metrics.collections import (
    RubricsScoreWithoutReference,
    RubricsScoreWithReference,
)

# Reference-free
metric_no_ref = RubricsScoreWithoutReference(llm=llm)

# Reference-based
metric_with_ref = RubricsScoreWithReference(llm=llm)
```

## 默认评分细则 { #default-rubrics }

### 无参考答案的评分细则（默认） { #reference-free-rubrics-default }

| 分数 | 描述 |
|-------|-------------|
| 1 | 回答完全错误，未回应用户输入的任何部分。 |
| 2 | 回答部分准确，但存在重大错误或明显遗漏。 |
| 3 | 回答基本准确，但在清晰度、完整性或少量细节上有所欠缺。 |
| 4 | 回答准确清晰，仅有少量遗漏或轻微偏差。 |
| 5 | 回答完全准确、清晰，全面回应了用户输入。 |

### 有参考答案的评分细则 { #reference-based-rubrics }

| 分数 | 描述 |
|-------|-------------|
| 1 | 回答完全错误、不相关，或与参考答案不符。 |
| 2 | 回答与参考答案部分一致，但存在重大错误或遗漏。 |
| 3 | 回答整体上与参考答案一致，但细节或清晰度不足。 |
| 4 | 回答基本准确，与参考答案高度一致，仅有少量问题。 |
| 5 | 回答完全准确，与参考答案完全一致，表述清楚且细节充分。 |

---

## 样本专属评分细则 { #instance-specific-rubrics }

不同样本需要不同标准时，使用 `InstanceSpecificRubrics`，适用于以下情况。
- 不同问题需要不同的评判标准
- 希望按具体任务要求定制评分
- 数据集中各样本的评估标准不同

### 示例 { #example_1 }

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import InstanceSpecificRubrics

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

metric = InstanceSpecificRubrics(llm=llm)

# Each sample can have its own rubrics
email_rubrics = {
    "score1_description": "The email is unprofessional or inappropriate",
    "score2_description": "The email lacks proper formatting or tone",
    "score3_description": "The email is acceptable but could be improved",
    "score4_description": "The email is professional with minor issues",
    "score5_description": "The email is highly professional and well-written",
}

result = await metric.ascore(
    user_input="Write a professional email declining a meeting invitation",
    response="Dear John, Thank you for the invitation...",
    rubrics=email_rubrics,
)
print(f"Score: {result.value}, Feedback: {result.reason}")

# Different rubrics for a different type of task
code_rubrics = {
    "score1_description": "The code doesn't work or has critical bugs",
    "score2_description": "The code has significant issues or is poorly structured",
    "score3_description": "The code works but lacks optimization or best practices",
    "score4_description": "The code is good with minor improvements possible",
    "score5_description": "The code is excellent, efficient, and follows best practices",
}

result = await metric.ascore(
    user_input="Write a function to sort a list",
    response="def sort_list(arr): return sorted(arr)",
    rubrics=code_rubrics,
)
```

### 结合参考答案与上下文 { #with-reference-and-contexts }

```python
result = await metric.ascore(
    user_input="Explain the water cycle",
    response="The water cycle involves evaporation, condensation, and precipitation.",
    reference="The water cycle describes how water evaporates from surfaces, rises into the atmosphere, condenses into clouds, and falls as precipitation.",
    retrieved_contexts=["Water cycle information from encyclopedia..."],
    rubrics={
        "score1_description": "Explanation is completely wrong",
        "score2_description": "Explanation has major inaccuracies",
        "score3_description": "Explanation is partially correct",
        "score4_description": "Explanation is mostly correct",
        "score5_description": "Explanation is comprehensive and accurate",
    },
)
```

---

## 旧版 API { #legacy-api }

!!! warning "已弃用"
    以下旧版 API 已弃用，请改用 `ragas.metrics.collections.DomainSpecificRubrics` 或 `ragas.metrics.collections.InstanceSpecificRubrics`。

```python
from ragas import evaluate
from datasets import Dataset

from ragas.metrics import rubrics_score_without_reference, rubrics_score_with_reference

rows = {
    "question": [
        "What's the longest river in the world?",
    ],
    "ground_truth": [
        "The Nile is a major north-flowing river in northeastern Africa.",
    ],
    "answer": [
        "The longest river in the world is the Nile, stretching approximately 6,650 kilometers (4,130 miles) through northeastern Africa.",
    ],
    "contexts": [
        [
            "Scientists debate whether the Amazon or the Nile is the longest river in the world.",
            "The Nile River was central to the Ancient Egyptians' rise to wealth and power.",
        ],
    ]
}

dataset = Dataset.from_dict(rows)

result = evaluate(
    dataset,
    metrics=[
        rubrics_score_without_reference,
        rubrics_score_with_reference
    ],
)
```

使用旧版 API 自定义评分细则的示例如下。

```python
from ragas.metrics._domain_specific_rubrics import RubricsScore

my_custom_rubrics = {
    "score1_description": "answer and ground truth are completely different",
    "score2_description": "answer and ground truth are somewhat different",
    "score3_description": "answer and ground truth are somewhat similar",
    "score4_description": "answer and ground truth are similar",
    "score5_description": "answer and ground truth are exactly the same",
}

rubrics_score = RubricsScore(rubrics=my_custom_rubrics)
```


