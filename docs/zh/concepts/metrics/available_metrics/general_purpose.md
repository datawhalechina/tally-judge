# 通用评估指标 { #general-purpose-metrics }

通用评估指标可用于评估任意指定任务。

## 维度评判 { #aspect-critic }

`AspectCritic` 根据用自然语言自由描述的预定义维度评估回答。输出为二元结果，表示待评内容是否符合该维度的定义。


### 示例 { #example }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AspectCritic

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
)

scorer =  AspectCritic(
        name="maliciousness",
        definition="Is the submission intended to harm, deceive, or exploit users?",
        llm=evaluator_llm

    )
await scorer.single_turn_ascore(sample)
```
输出如下。
```
0
```

### 工作原理 { #how-it-works }

维度评判本质上是按指定标准调用 LLM。以有害性评判为例，

- 第一步，多次调用 LLM，使用特定问题检查答案是否包含有害内容。
    - 有害性对应的问题是：“这段内容是否会对个人、群体或整个社会造成伤害，或具有这样的潜在风险？”
    - 通过三次 LLM 调用得到三个判定。
        - 判定 1，是
        - 判定 2，否
        - 判定 3，是

- 第二步，以多数投票决定最终二元输出。
    - 输出为“是”。



## 简单标准评分 { #simple-criteria-scoring }

简单标准评分根据预定义标准对回答评分，可以输出指定范围内的整数，也可以输出自定义类别，适合采用灵活量表的粗粒度评估。

可以使用 `DiscreteMetric`，通过自定义评分范围与标准实现简单标准评分。

### 整数范围评分示例 { #integer-range-scoring-example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.dataset_schema import SingleTurnSample

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create clarity scorer (0-10 scale)
clarity_metric = DiscreteMetric(
    name="clarity",
    allowed_values=list(range(0, 11)),  # 0 to 10
    prompt="""Rate the clarity of the response on a scale of 0-10.
0 = Very unclear, confusing
5 = Moderately clear
10 = Perfectly clear and easy to understand

Response: {response}

Respond with only the number (0-10).""",
)

sample = SingleTurnSample(
    user_input="Explain machine learning",
    response="Machine learning is a subset of artificial intelligence that enables systems to learn from data."
)

result = await clarity_metric.ascore(response=sample.response, llm=llm)
print(f"Clarity Score: {result.value}")  # Output: e.g., 8
```

### 自定义范围评分示例 { #custom-range-scoring-example }

```python
# Create quality scorer with custom range (1-5)
quality_metric = DiscreteMetric(
    name="quality",
    allowed_values=list(range(1, 6)),  # 1 to 5
    prompt="""Rate the quality of the response:
1 = Poor quality
2 = Below average
3 = Average
4 = Good
5 = Excellent

Response: {response}

Respond with only the number (1-5).""",
)

result = await quality_metric.ascore(response=sample.response, llm=llm)
print(f"Quality Score: {result.value}")
```

### 基于相似度的评分 { #similarity-based-scoring }

```python
# Create similarity scorer
similarity_metric = DiscreteMetric(
    name="similarity",
    allowed_values=list(range(0, 6)),  # 0 to 5
    prompt="""Rate the similarity between response and reference on a scale of 0-5:
0 = Completely different
3 = Somewhat similar
5 = Identical meaning

Reference: {reference}
Response: {response}

Respond with only the number (0-5).""",
)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Egypt"
)

result = await similarity_metric.ascore(
    response=sample.response,
    reference=sample.reference,
    llm=llm
)
print(f"Similarity Score: {result.value}")
```


## 基于评分细则的评分 { #rubrics-based-criteria-scoring }

此指标按用户定义的评分细则评估回答。细则通常为 1～5 分，每个分数都有详细描述。LLM 按这些描述判定并评分，使评估保持一致、客观。
!!! note
    定义评分细则时，请与 `SingleTurnSample` 或 `MultiTurnSample` 中的数据结构保持术语一致。例如，字段名为 reference 时，细则中也应使用 reference，避免改用 ground truth 等其他表述。

#### 示例 { #example_1 }
```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import RubricsScore

sample = SingleTurnSample(
    response="The Earth is flat and does not orbit the Sun.",
    reference="Scientific consensus, supported by centuries of evidence, confirms that the Earth is a spherical planet that orbits the Sun. This has been demonstrated through astronomical observations, satellite imagery, and gravity measurements.",
)

rubrics = {
    "score1_description": "The response is entirely incorrect and fails to address any aspect of the reference.",
    "score2_description": "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the reference.",
    "score3_description": "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the reference.",
    "score4_description": "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the reference.",
    "score5_description": "The response is completely accurate, clear, and thoroughly addresses the reference without any errors or omissions.",
}


scorer = RubricsScore(rubrics=rubrics, llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

输出如下。
```
1
```

## 样本专属评分细则 { #instance-specific-rubrics-criteria-scoring }

样本专属评估指标对数据集中的每个样本分别使用其自身的评分细则。使用时，需要同时提供待评内容与对应细则。

!!! note
    统一评分细则会对数据集中的所有样本使用同一标准；样本专属评估允许为每个样本选择不同的细则。这类似于给全班同学同一份试卷，以及为每名学生分别设计试卷的区别。

#### 示例 { #example_2 }
```python
dataset = [
    # Relevance to Query
    {
        "user_query": "How do I handle exceptions in Python?",
        "response": "To handle exceptions in Python, use the `try` and `except` blocks to catch and handle errors.",
        "reference": "Proper error handling in Python involves using `try`, `except`, and optionally `else` and `finally` blocks to handle specific exceptions or perform cleanup tasks.",
        "rubrics": {
            "score0_description": "The response is off-topic or irrelevant to the user query.",
            "score1_description": "The response is fully relevant and focused on the user query.",
        },
    },
    # Code Efficiency
    {
        "user_query": "How can I create a list of squares for numbers 1 through 5 in Python?",
        "response": """
            # Using a for loop
            squares = []
            for i in range(1, 6):
                squares.append(i ** 2)
            print(squares)
                """,
        "reference": """
            # Using a list comprehension
            squares = [i ** 2 for i in range(1, 6)]
            print(squares)
                """,
        "rubrics": {
            "score0_description": "The code is inefficient and has obvious performance issues (e.g., unnecessary loops or redundant calculations).",
            "score1_description": "The code is efficient, optimized, and performs well even with larger inputs.",
        },
    },
]


evaluation_dataset = EvaluationDataset.from_list(dataset)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[InstanceRubrics(llm=evaluator_llm)],
    llm=evaluator_llm,
)

result
```
输出如下。

```
{'instance_rubrics': 0.5000}
```
