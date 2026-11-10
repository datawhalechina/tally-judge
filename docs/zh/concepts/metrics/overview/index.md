# 评估指标概览 { #overview-of-metrics }

## 指标为什么重要 { #why-metrics-matter }

衡量是改进的前提。指标构成反馈闭环，让持续迭代成为可能。

AI 系统的进步依赖大量实验，每次实验都在验证一个改善系统表现的假设。如果缺少清晰可靠的指标，就难以区分成功的实验（新得分高于旧得分）与失败的实验。

指标为优化提供方向，帮助你量化改进、发现退化，并将优化工作与用户体验和业务价值联系起来。

指标是衡量 AI 应用表现的定量度量，用于评价应用整体及其各个组件在给定测试数据上的表现。它为应用开发与部署过程中的比较、优化和决策提供数值依据，主要用于以下方面。

1. **组件选择**。使用自己的数据比较 LLM、检索器和智能体配置等不同组件，选择最适合的方案。
2. **错误诊断与调试**。定位导致错误或表现不佳的组件，便于调试和改进。
3. **持续监控与维护**。跟踪应用表现随时间的变化，发现并应对数据漂移、模型退化和用户需求变化。

## AI 应用中的指标类型 { #types-of-metrics-in-ai-applications }

### 1. 端到端指标 { #1-end-to-end-metrics }

端到端指标从用户视角衡量系统的整体表现，将 AI 应用视为黑盒，仅根据最终输出量化用户关心的结果。

例如，

- 答案正确性，衡量检索增强生成（RAG）系统给出的答案是否准确。
- 引用准确性，衡量 RAG 系统所引用的资料是否正确且相关。

优化端到端指标，可以带来与用户期望直接相关的实际改进。

### 2. 组件级指标 { #2-component-level-metrics }

组件级指标分别评估 AI 系统的各个组成部分，便于采取具体行动并开展针对性优化，但其变化未必与终端用户满意度直接对应。

例如，

- 检索准确率，衡量 RAG 系统检索相关信息的能力。较低的准确率（例如 50%）意味着改进检索组件可能提升整体表现。不过，单独提升某个组件并不能保证端到端效果一定改善。

### 3. 业务指标 { #3-business-metrics }

业务指标将 AI 系统表现与组织目标联系起来，量化实际业务成果。它们通常属于滞后指标，需要在部署一段时间（数天、数周或数月）后计算。

例如，

- 客服工单分流率，衡量部署 AI 助手后减少的人工客服工单比例。

## Ragas 中的指标类型 { #types-of-metrics-in-ragas }

<figure markdown="span">
  ![按组件评估](../../../../_static/imgs/metrics_mindmap.png){width="600"}
  <figcaption>指标思维导图</figcaption>
</figure>

**按底层评估机制，指标可以分为两类。**

&nbsp;&nbsp;&nbsp;&nbsp; **基于 LLM 的指标**。调用 LLM 完成评估，可能通过一次或多次模型调用得到分数或结果。由于 LLM 对相同输入未必始终生成相同结果，这些指标具有一定的不确定性。研究和实践表明，它们通常更准确，也更接近人工评价。

Ragas 中所有基于 LLM 的指标都继承自 `MetricWithLLM` 类，评分前需要设置 LLM 对象。

```python
from ragas.metrics import FactualCorrectness
scorer = FactualCorrectness(llm=evaluation_llm)
```

每个基于 LLM 的指标都有通过[提示词对象](../../components/prompt.md)编写的相关提示词。你可以按领域和场景定制这些提示词，详情参阅[修改指标提示词](../../../howtos/customizations/metrics/modifying-prompts-metrics.md)。


&nbsp;&nbsp;&nbsp;&nbsp; **不依赖 LLM 的指标**。通过字符串相似度、BLEU 等传统方法评估应用。这些指标具有确定性，无须调用 LLM，但与人工评价的相关性通常较低。

Ragas 中所有不依赖 LLM 的指标都继承自 `Metric` 类。

**按评估数据的交互形式，指标还可以分为两类。**

&nbsp;&nbsp;&nbsp;&nbsp; **单轮指标**。评估用户与 AI 之间的一轮交互。Ragas 中支持单轮评估的指标继承自 [SingleTurnMetric][ragas.metrics.base.SingleTurnMetric] 类，使用 `single_turn_ascore` 方法评分，并接收[单轮样本][ragas.dataset_schema.SingleTurnSample]对象作为输入。

```python
from ragas.metrics import FactualCorrectness

scorer = FactualCorrectness()
await scorer.single_turn_ascore(sample)
```

&nbsp;&nbsp;&nbsp;&nbsp; **多轮指标**。评估用户与 AI 之间的多轮交互。Ragas 中支持多轮评估的指标继承自 [MultiTurnMetric][ragas.metrics.base.MultiTurnMetric] 类，使用 `multi_turn_ascore` 方法评分，并接收[多轮样本][ragas.dataset_schema.MultiTurnSample]对象作为输入。

```python
from ragas.metrics import AgentGoalAccuracy
from ragas import MultiTurnSample

scorer = AgentGoalAccuracy()
await scorer.multi_turn_ascore(sample)
```

### 输出类型 { #output-types }

Ragas 也根据指标输出的类型进行分类，便于理解各类指标的行为，以及如何解释和汇总结果，共有三类。

#### 1. 离散指标 { #1-discrete-metrics }

从预先定义的类别列表中返回一个值，类别本身不隐含顺序。常见用途包括将输出判定为通过/失败，或好/一般/差。离散指标可以直接接收自定义提示词，适合快速实现自定义评估。

例如，
```python
from ragas.metrics import discrete_metric

@discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
def my_metric(predicted: str, expected: str) -> str:
    return "pass" if predicted.lower() == expected.lower() else "fail"
```

要修改现有 collections 指标（如 Faithfulness、FactualCorrectness）的提示词，请参阅[修改指标提示词](../../../howtos/customizations/metrics/modifying-prompts-metrics.md)。

#### 2. 数值指标 { #2-numeric-metrics }

返回指定范围内的整数或浮点数，支持均值、总和和众数等聚合运算，适合统计分析。

```python
from ragas.metrics import numeric_metric

@numeric_metric(name="response_accuracy", allowed_values=(0, 1))
def my_metric(predicted: float, expected: float) -> float:
    return abs(predicted - expected) / max(expected, 1e-5)

my_metric.score(predicted=0.8, expected=1.0)  # Returns a float value
```

#### 3. 排序指标 { #3-ranking-metrics }

一次评估多个输出，并按指定标准返回排序结果，适合比较同一流水线生成的多个输出。

```python
from ragas.metrics import ranking_metric
@ranking_metric(name="response_ranking", allowed_values=[0,1])
def my_metric(responses: list) -> list:
    response_lengths = [len(response) for response in responses]
    sorted_indices = sorted(range(len(response_lengths)), key=lambda i: response_lengths[i])
    return sorted_indices

my_metric.score(responses=["short", "a bit longer", "the longest response"])  # Returns a ranked list of indices
```

## 指标设计原则 { #metric-design-principles }

为 AI 应用设计有效指标，需要确保其可靠、可解释且与任务相关。Ragas 遵循以下五项原则。

**1. 聚焦单一维度**
一个指标应只衡量应用表现的某个具体维度，使其含义清晰，并能指导实际改进。

**2. 直观且易于解释**
指标应易于理解和解释，让团队能够清楚地交流结果，并得出有意义的结论。

**3. 有效的提示词流程**
使用 LLM 构建指标时，应设计贴近人工评估方式的提示词流程。将复杂任务拆分为较小的子任务，并分别设置提示词，可以提高指标的准确性与相关性。

**4. 稳健性**
为基于 LLM 的指标提供足够的少样本示例，展示期望结果，为模型提供上下文和指导，从而增强指标的稳健性。

**5. 统一评分范围**
将分数归一化，或限制在统一范围内（例如 0～1），有助于比较不同指标，保持评估框架的一致性和可解释性。

这些原则有助于构建有效、实用且有意义的 AI 应用评估指标。

## 为应用选择合适的指标 { #choosing-the-right-metrics-for-your-application }

### 1. 优先考虑端到端指标 { #1-prioritize-end-to-end-metrics }

先关注反映整体用户满意度的指标。事实正确性、回答语气和解释深度都会影响满意度，初期应集中于最有用户价值的少数维度，例如 RAG 助手的答案准确性和引用准确性。

### 2. 确保可解释性 { #2-ensure-interpretability }

指标应足够清晰，让整个团队都能理解并据此推理。例如，

- 文本转 SQL 系统中的执行准确率，检查生成的 SQL 查询是否返回与领域专家编写的标准查询完全相同的数据集。

### 3. 优先采用客观标准 { #3-emphasize-objective-over-subjective-metrics }

优先选择评判标准客观的指标，减少主观判断。可以让团队成员独立标注样本，再衡量一致程度。较高的标注者一致率（≥80%）意味着标准更客观。

### 4. 选择信号明确的少数指标 { #4-few-strong-signals-over-many-weak-signals }

大量信号微弱的指标会增加决策难度。应选择数量较少、信号清晰可靠的指标。例如，

- 对于对话 AI，目标达成率（用户与 AI 交互的目的是否实现）这一项指标，有时比连贯性、有帮助程度等多个较弱的替代指标更能反映系统表现。
