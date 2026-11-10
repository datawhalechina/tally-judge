# NVIDIA 指标 { #nvidia-metrics }

## 答案准确性 { #answer-accuracy }

**答案准确性（Answer Accuracy）**衡量模型回答与给定问题的标准答案是否一致。它通过两个不同的“LLM 作为裁判”提示词，分别给出 0、2 或 4 分，再将其转换到 [0,1] 范围并取平均。分数越高，表示模型答案越接近参考答案。

- **0** → **回答**不准确，或与**参考答案**所回答的问题不一致。
- **2** → **回答**与**参考答案**部分一致。
- **4** → **回答**与**参考答案**完全一致。


### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = AnswerAccuracy(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="When was Einstein born?",
    response="Albert Einstein was born in 1879.",
    reference="Albert Einstein was born in 1879."
)
print(f"Answer Accuracy Score: {result.value}")
```

输出如下。

```
Answer Accuracy Score: 1.0
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        user_input="When was Einstein born?",
        response="Albert Einstein was born in 1879.",
        reference="Albert Einstein was born in 1879."
    )
    ```

### 计算过程 { #how-its-calculated }

**第一步**，使用两个不同的模板让 LLM 评分，提高稳健性。

- **模板 1**，比较**回答**与**参考答案**，给出 **0、2 或 4** 分。
- **模板 2**，再次评估同一问题，但交换**回答**与**参考答案**的位置。

这种双视角方式有助于公平评价答案准确性。

**第二步**，如果两次评分都有效，取其平均值作为最终得分；如果仅有一次有效，则使用该次得分。

**计算示例**

- **用户输入**：“爱因斯坦出生于哪一年？”
- **回答**：“阿尔伯特·爱因斯坦出生于 1879 年。”
- **参考答案**：“阿尔伯特·爱因斯坦出生于 1879 年。”

假设两个模板均给出 **4** 分，即完全匹配，转换过程如下。

- 原始评分 **4** 对应 [0,1] 范围内的 **1**。
- 两次得分的平均值为 (1 + 1) / 2 = **1**。

因此，最终的**答案准确性**得分为 **1**。

### 相近的 Ragas 指标 { #similar-ragas-metrics }

1. [答案正确性](answer_correctness.md)，同时考虑语义与事实相似度，衡量生成答案与标准答案的接近程度。

2. [评分细则评分](general_purpose.md#rubrics-based-criteria-scoring)，根据用户定义的评分细则评估回答。每一级分数都有具体描述，LLM 据此评分，使过程一致、客观。

### 指标比较 { #comparison-of-metrics }

#### Answer Correctness 与 Answer Accuracy { #answer-correctness-vs-answer-accuracy }

- **LLM 调用次数**。Answer Correctness 需要三次调用，两次拆分回答与参考答案中的陈述，一次进行分类。Answer Accuracy 使用两次独立的 LLM 评判。
- **Token 用量**。Answer Correctness 包含详细拆分与分类过程，消耗更多 token。
- **可解释性**。Answer Correctness 能提供事实正确性和语义相似度的详细信息，可解释性更强。Answer Accuracy 直接提供分数。
- **评估稳健性**。Answer Accuracy 通过双重评判提高一致性，Answer Correctness 则深入分析回答质量，提供更全面的视角。

#### Answer Accuracy 与评分细则评分 { #answer-accuracy-vs-rubric-score }

- **LLM 调用次数**。Answer Accuracy 调用两次，每位裁判一次；评分细则评分只需一次。
- **Token 用量**。Answer Accuracy 仅输出分数，用量较少；评分细则评分还会生成推理，消耗更多 token。
- **可解释性**。Answer Accuracy 只提供原始分数；评分细则评分同时给出理由与判定。
- **效率**。Answer Accuracy 较轻量，在较小模型上也能取得良好效果。

### 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerAccuracy

sample = SingleTurnSample(
    user_input="When was Einstein born?",
    response="Albert Einstein was born in 1879.",
    reference="Albert Einstein was born in 1879."
)

scorer = AnswerAccuracy(llm=evaluator_llm) # evaluator_llm wrapped with ragas LLM Wrapper
score = await scorer.single_turn_ascore(sample)
print(score)
```

输出如下。

```
1.0
```

## 上下文相关性 { #context-relevance }

**上下文相关性（Context Relevance）**衡量 **retrieved_contexts**（片段或段落）与 **user_input** 的相关程度。通过两次独立的 LLM 裁判调用分别给出 **0、1 或 2** 分，再归一化到 [0,1] 并取平均。分数越高，表示上下文越切合用户查询。

- **0** → 检索上下文与用户查询完全无关。
- **1** → 上下文部分相关。
- **2** → 上下文完全相关。

### 示例 { #example_1 }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextRelevance(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="When and Where Albert Einstein was born?",
    retrieved_contexts=[
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
)
print(f"Context Relevance Score: {result.value}")
```

输出如下。

```
Context Relevance Score: 1.0
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        user_input="When and Where Albert Einstein was born?",
        retrieved_contexts=[...]
    )
    ```

### 实现说明 { #implementation-note }

**与原始论文的区别**。原始 Ragas 论文基于句子提取定义上下文相关性（CR = 相关句子数量 / 句子总数）。当前实现采用离散评判，让 LLM 对整体上下文相关性给出 0～2 分，提高效率并减少句子边界划分错误。这一设计旨在保持评估目标的同时，提高可靠性并降低计算开销。

### 计算过程 { #how-its-calculated_1 }

**第一步**，使用两个不同模板（template_relevance1 和 template_relevance2），评估检索上下文与用户查询的相关性。每次返回 **0、1 或 2** 分。独立评估有助于增强稳健性，减轻单次 LLM 判断的偏差。

**第二步**，将每个分数除以 2，归一化到 [0,1]。两次评分均有效时取平均；只有一次有效时使用该次得分。

**计算示例**

- **用户输入**：“阿尔伯特·爱因斯坦出生于何时何地？”
- **检索上下文**
  - “阿尔伯特·爱因斯坦出生于 1879 年 3 月 14 日。”
  - “阿尔伯特·爱因斯坦出生于德国符腾堡的乌尔姆。”

这两段上下文共同提供了出生日期和地点，完整回答用户查询。因此，两个提示词都会给出 **2** 分（完全相关）。归一化后均为 **1.0**（2/2），最终平均得分为 **1**。

### 相近的 Ragas 指标 { #similar-ragas-metrics_1 }

1. [上下文精确率](context_precision.md)，衡量检索上下文对回答用户查询的相关程度，通过相关片段所在位置的 precision@k 均值，反映检索系统将相关信息排在前面的能力。

2. [上下文召回率](context_recall.md)，衡量相关信息被成功检索的程度。计算检索结果所覆盖的相关陈述或上下文占参考信息总数的比例，检查重要信息是否遗漏。

3. [评分细则评分](general_purpose.md#rubrics-based-criteria-scoring)，按用户定义的评分细则评估回答。评分标准和量表均可按需定制。

#### 上下文精确率、召回率与上下文相关性 { #context-precision-and-context-recall-vs-context-relevance }

- **LLM 调用次数**。上下文精确率判断上下文是否有助于获得参考答案，返回 1 或 0；上下文召回率判断答案陈述是否有上下文依据，返回是（1）或否（0）。此处所述两者各进行一次调用，上下文相关性使用两次调用以增强稳健性。
- **Token 用量**。上下文精确率与召回率消耗更多 token，上下文相关性更节省。
- **可解释性**。上下文精确率与召回率提供详细理由，上下文相关性直接提供分数。
- **评估稳健性**。相比一次调用的评估方式，上下文相关性通过两次 LLM 评判增强稳健性。

### 旧版指标 API { #legacy-metrics-api_1 }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample_1 }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextRelevance

sample = SingleTurnSample(
    user_input="When and Where Albert Einstein was born?",
    retrieved_contexts=[
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
)

scorer = ContextRelevance(llm=evaluator_llm)
score = await scorer.single_turn_ascore(sample)
print(score)
```

输出如下。

```
1.0
```

## 回答依据充分性 { #response-groundedness }

**回答依据充分性（Response Groundedness）**衡量回答是否得到检索上下文的支持，检查回答中的每条陈述是否能在给定上下文中找到完整或部分依据。

- **0** → 回答完全没有上下文依据。
- **1** → 回答部分有依据。
- **2** → 回答完全有依据，每条陈述都能在检索上下文中找到或推断出来。

### 示例 { #example_2 }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ResponseGroundedness

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ResponseGroundedness(llm=llm)

# Evaluate
result = await scorer.ascore(
    response="Albert Einstein was born in 1879.",
    retrieved_contexts=[
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
)
print(f"Response Groundedness Score: {result.value}")
```

输出如下。

```
Response Groundedness Score: 1.0
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        response="Albert Einstein was born in 1879.",
        retrieved_contexts=[...]
    )
    ```

### 计算过程 { #how-its-calculated_2 }

**第一步**，使用两个不同模板，让 LLM 评估回答相对于检索上下文的依据充分程度，每次返回 **0、1 或 2** 分。

**第二步**，将每个评分除以 2，转换到 [0,1] 范围，即 0 转为 0.0，1 转为 0.5，2 转为 1.0。两次均有效时取平均；只有一次有效时使用该次得分。

**计算示例**

- **回答**：“阿尔伯特·爱因斯坦出生于 1879 年。”
- **检索上下文**
  - “阿尔伯特·爱因斯坦出生于 1879 年 3 月 14 日。”
  - “阿尔伯特·爱因斯坦出生于德国符腾堡的乌尔姆。”

上下文提供了爱因斯坦的出生日期与地点。虽然回答只包含年份，但其陈述有上下文支持，因此两个提示词很可能都给出 **2** 分。归一化后均为 **1.0**（2/2），最终平均得分为 **1**。

### 相近的 Ragas 指标 { #similar-ragas-metrics_2 }

1. [忠实度](faithfulness.md)，衡量回答与检索上下文在事实层面的一致性，检查每条陈述是否得到支持。得分为 0～1，越高表示一致性越好。

2. [评分细则评分](general_purpose.md#rubrics-based-criteria-scoring)，依据用户指定标准评估回答。调整细则即可用于评估答案准确性、上下文相关性或回答依据充分性。

### 指标比较 { #comparison-of-metrics_1 }

#### 忠实度与回答依据充分性 { #faithfulness-vs-response-groundedness }

- **LLM 调用次数**。忠实度通过两次调用完成详细的陈述拆分和判断；回答依据充分性使用两次独立的 LLM 评判。
- **Token 用量**。忠实度消耗更多 token，回答依据充分性更节省。
- **可解释性**。忠实度为每条陈述提供可查看的判断理由；回答依据充分性直接提供分数。
- **评估稳健性**。忠实度结合用户输入作全面评估，回答依据充分性通过双重评判提高一致性。

### 旧版指标 API { #legacy-metrics-api_2 }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample_2 }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseGroundedness

sample = SingleTurnSample(
    response="Albert Einstein was born in 1879.",
    retrieved_contexts=[
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
)

scorer = ResponseGroundedness(llm=evaluator_llm)
score = await scorer.single_turn_ascore(sample)
print(score)
```

输出如下。

```
1.0
```
