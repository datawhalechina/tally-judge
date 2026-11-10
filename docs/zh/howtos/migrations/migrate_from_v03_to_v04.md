# 从 v0.3 迁移到 v0.4 { #migration-from-v03-to-v04 }

Ragas v0.4 转向**以实验为基础的架构**。这是自 v0.2 以来的重要改动，将评估、分析与迭代整合到统一的实验框架中。

这一架构调整带来以下改进。

1. **基于 Collections 的指标系统**，统一指标用法，便于在实验中使用
2. **统一的 LLM 工厂**，简化初始化，支持多种服务商
3. **新版提示词系统**，支持更易组合和复用的提示词

本指南逐步介绍关键变化及迁移方法。

## 主要变化概览 { #overview-of-major-changes }

以实验为基础的架构主要关注以下三个方面。

1. **实验中心设计**，将单次指标运行整合为包含分析的结构化实验工作流
2. **Collections 指标**，返回结构化结果，便于分析和跟踪
3. **改进 LLM 与提示词系统**，统一服务商支持与提示词模式

### 主要数据 { #key-statistics }

- **指标迁移**，20 多个核心指标迁入 collections 系统
- **不兼容变更**，7 项以上主要 API 改动
- **弃用项**，旧版封装类和提示词定义
- **新增功能**，支持 GPT-5/o 系列、自动处理参数约束及多服务商

## 理解以实验为基础的架构 { #understanding-the-experiment-based-architecture }

迁移前，先了解使用思路的变化。

**v0.3，以指标为中心**
```
Data → Individual Metric → Score → Analysis
```

各次指标评估相对独立，运行后得到浮点数，跟踪与分析通常在外部完成。

**v0.4，以实验为中心**
```
Data → Experiment → [Metrics Collection] → Structured Results → Integrated Analysis
```

指标在实验上下文中运行，评估、分析与迭代相互衔接，支持以下能力。

- 跟踪包含解释的指标结果
- 比较多次实验
- 使用内置能力分析指标行为
- 更方便地迭代系统


## 迁移顺序 { #migration-path }

建议按以下顺序迁移。

1. **更新评估方式**，从 `evaluate()` 转向 `experiment()`，参阅[从评估到实验](#evaluation-to-experiment)
2. **更新 LLM 配置**，参阅 [LLM 初始化](#llm-initialization)
3. **迁移指标**，参阅[指标迁移](#metrics-migration)
4. **迁移嵌入模型**，参阅[嵌入模型迁移](#embeddings-migration)
5. **更新自定义提示词**，参阅[提示词系统迁移](#prompt-system-migration)
6. **更新数据结构**，参阅[数据结构变化](#data-schema-changes)
7. **重构自定义指标**，参阅[自定义指标](#custom-metrics)

---

## 从评估到实验 { #evaluation-to-experiment }

v0.4 推荐以 `experiment()` 为基础的方式，支持迭代评估和结构化结果跟踪。

### 主要变化 { #what-changed }

从直接返回分数的**评估函数** `evaluate()`，转向内置跟踪和版本管理的**实验装饰器** `@experiment()`，组织结构化工作流。

### 迁移前（v0.3） { #before-v03 }

```python
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

# Setup
dataset = ...  # Your dataset
metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm)]

# Simple evaluation
result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings
)

print(result)  # Returns EvaluationResult with scores
```

### 迁移后（v0.4） { #after-v04 }

```python
from ragas import experiment
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from pydantic import BaseModel

# Define experiment result structure
class ExperimentResult(BaseModel):
    faithfulness: float
    answer_relevancy: float

# Create experiment function
@experiment(ExperimentResult)
async def run_evaluation(row):
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm)

    faith_result = await faithfulness.ascore(
        response=row.response,
        retrieved_contexts=row.contexts
    )

    relevancy_result = await answer_relevancy.ascore(
        user_input=row.user_input,
        response=row.response
    )

    return ExperimentResult(
        faithfulness=faith_result.value,
        answer_relevancy=relevancy_result.value
    )

# Run experiment
exp_results = await run_evaluation(dataset)
```

### 使用 `experiment()` 的收益 { #benefits-of-using-experiment }

1. **结构化结果**，明确指定需要记录的内容
2. **逐行控制**，按需定制单条样本的评估
3. **版本跟踪**，通过 `version_experiment()` 可选集成 Git
4. **迭代工作流**，方便修改并重新运行实验
5. **更好集成**，结合新版指标和数据集使用

---

## LLM 初始化 { #llm-initialization }

### 主要变化 { #what-changed_1 }

v0.3 根据用途使用不同工厂函数。

- `instructor_llm_factory()`，用于需要 Instructor 的指标
- `llm_factory()`，用于通用 LLM 操作
- 各种 LangChain 和 LlamaIndex 封装类

v0.4 将其整合为**统一工厂**。

```python
from ragas.llms import llm_factory
```

该工厂提供以下功能。

- 返回支持结构化输出的 `InstructorBaseRagasLLM`
- 自动检测并配置服务商特定约束
- 支持 GPT-5 与 o 系列，自动处理 `temperature` 和 `top_p` 约束
- 支持 OpenAI、Anthropic、Cohere、Google、Azure、Bedrock 等主要服务商

### 迁移前（v0.3） { #before-v03_1 }

```python
from ragas.llms import instructor_llm_factory, llm_factory
from openai import AsyncOpenAI

# For metrics that need instructor
llm = instructor_llm_factory("openai", model="gpt-4o-mini", client=AsyncOpenAI(api_key="..."))

# Or, the old way (not recommended, still supported in 0.3)
client = AsyncOpenAI(api_key="sk-...")
llm = llm_factory("openai", model="gpt-4o-mini", client=client)
```

### 迁移后（v0.4） { #after-v04_1 }

```python
from ragas.llms import llm_factory
from openai import AsyncOpenAI

# Single unified approach - works everywhere
client = AsyncOpenAI(api_key="sk-...")
llm = llm_factory("gpt-4o-mini", client=client)
```

**主要区别**

| 维度 | v0.3 | v0.4 |
|--------|------|------|
| **工厂函数** | `instructor_llm_factory()` 或 `llm_factory()` | `llm_factory()` |
| **服务商检测** | 手动指定服务商字符串 | 根据模型名称自动检测 |
| **返回类型** | 多种 `BaseRagasLLM` 实现 | `InstructorBaseRagasLLM` |
| **约束处理** | 手动配置 | 自动处理 GPT-5/o 系列约束 |
| **是否需要异步客户端** | 是 | 是 |

### 迁移步骤 { #migration-steps }

1. **更新导入**

    ```python
    # Remove this
    from ragas.llms import instructor_llm_factory

    # Use this instead
    from ragas.llms import llm_factory
    ```

2. **替换工厂调用**

    ```python
    # Old - v0.3
    llm = instructor_llm_factory("openai", model="gpt-4o", client=client)

    # New - v0.4
    llm = llm_factory("gpt-4o", client=client)
    ```

3. **更新其他服务商的用法**，模型名会自动识别。

    ```python
    # OpenAI
    llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI(api_key="..."))

    # Anthropic
    llm = llm_factory("claude-3-sonnet-20240229", client=AsyncAnthropic(api_key="..."))

    # Google
    llm = llm_factory("gemini-2.0-flash", client=...)
    ```

### LLM 封装类（已弃用） { #llm-wrapper-classes-deprecated }

以下封装类已弃用，将在未来移除。

```python
# Deprecated - will be removed
from ragas.llms import LangchainLLMWrapper, LlamaIndexLLMWrapper
```

```python
# Recommended - use llm_factory directly
from ragas.llms import llm_factory
```

**迁移方法**。直接调用 `llm_factory()`，由工厂自动识别服务商。

---

## 指标迁移 { #metrics-migration }

### 为什么调整指标 { #why-metrics-changed }

实验架构要求指标更好地融入工作流。

- **结构化结果**，返回包含得分和理由的 `MetricResult`，支持更丰富的分析与跟踪
- **关键字参数**，直接传入各项数据，便于组合指标并集成实验流水线
- **统一输入/输出**，collections 指标采用一致模式，便于进一步构建汇总分析与实验功能

### 架构变化 { #architectural-changes }

指标系统围绕实验工作流重新设计，核心区别如下。

#### 基类变化 { #base-class-changes }

| 维度 | v0.3 | v0.4 |
|--------|------|------|
| **导入路径** | `from ragas.metrics import Metric` | `from ragas.metrics.collections import Metric` |
| **基类** | `MetricWithLLM`, `SingleTurnMetric` | collections 中的 `BaseMetric` |
| **评分方法** | `async def single_turn_ascore(sample: SingleTurnSample)` | `async def ascore(**kwargs)` |
| **输入类型** | `SingleTurnSample` 对象 | 各字段的关键字参数 |
| **输出类型** | `float` 分数 | `MetricResult`，含 `.value` 和可选 `.reason` |
| **LLM 参数** | 初始化时提供 | 初始化时提供 |

#### 评分流程 { #scoring-workflow }

**v0.3 用法**
```python
# 1. Create a sample object containing all data
sample = SingleTurnSample(
    user_input="What is AI?",
    response="AI is artificial intelligence...",
    retrieved_contexts=["Context 1", "Context 2"],
    ground_truths=["AI definition"]
)

# 2. Call metric with the sample
metric = Faithfulness(llm=llm)
score = await metric.single_turn_ascore(sample)  # Returns: 0.85
```

**v0.4 用法**
```python
# 1. Call metric with individual arguments
metric = Faithfulness(llm=llm)
result = await metric.ascore(
    user_input="What is AI?",
    response="AI is artificial intelligence...",
    retrieved_contexts=["Context 1", "Context 2"]
)

# 2. Access result properties
print(result.value)      # Score: 0.85 (float)
print(result.reason)     # Optional explanation
```

### v0.4 可用指标 { #available-metrics-in-v04 }

以下指标已迁入 v0.4 collections 系统。

#### RAG 评估指标 { #rag-evaluation-metrics }
- **Faithfulness**，回答是否得到检索上下文支持（v0.3.9+）
- **AnswerRelevancy**，回答是否切合用户问题（v0.3.9+）
- **AnswerCorrectness**，回答是否符合参考答案（v0.3.9+）
- **AnswerAccuracy**，答案事实是否准确
- **ContextPrecision**，检索结果是否优先排列相关上下文（v0.3.9+）
  - 有参考版本为 `ContextPrecisionWithReference`
  - 无参考版本为 `ContextPrecisionWithoutReference`
  - 旧名称 `ContextUtilization`，现在封装 ContextPrecisionWithoutReference
- **ContextRecall**，相关信息是否被充分检索（v0.3.9+）
- **ContextRelevance**，检索上下文与查询是否相关（v0.3.9+）
- **ContextEntityRecall**，上下文是否覆盖参考答案的重要实体（v0.3.9+）
- **NoiseSensitivity**，系统对上下文噪声的敏感程度（v0.3.9+）
- **ResponseGroundedness**，回答中的陈述是否有上下文依据

#### 文本比较指标 { #text-comparison-metrics }
- **SemanticSimilarity**，两个文本语义是否接近（v0.3.9+）
- **FactualCorrectness**，事实陈述是否正确（v0.3.9+）
- **BleuScore**，BLEU 得分（v0.3.9+）
- **RougeScore**，ROUGE 得分（v0.3.9+）

#### 基于字符串的指标（无须 LLM） { #string-based-metrics-non-llm }
- **ExactMatch**，精确字符串匹配
- **StringPresence**，检查子串是否存在
- **LevenshteinDistance**，基于编辑距离的相似度
- **MatchingSubstrings**，匹配子串数量
- **NonLLMStringSimilarity**，多种字符串相似度算法

#### 摘要指标 { #summary-metrics }
- **SummaryScore**，摘要整体质量评估（v0.3.9+）

#### 已移除的指标 { #removed-metrics-no-longer-available }
- **AspectCritic**，改用 `@discrete_metric()` 装饰器
- **SimpleCriteria**，改用 `@discrete_metric()` 装饰器
- **AnswerSimilarity**，改用 `SemanticSimilarity`

#### 智能体与工具指标（已迁移） { #agent-tool-metrics-migrated }
- **ToolCallAccuracy** - `ragas.metrics.collections.ToolCallAccuracy`
- **ToolCallF1** - `ragas.metrics.collections.ToolCallF1`
- **TopicAdherence** - `ragas.metrics.collections.TopicAdherence`
- **AgentGoalAccuracy**，对应 collections 中的目标达成指标

#### SQL 与数据指标（已迁移） { #sql-data-metrics-migrated }
- **DataCompy 得分**，`ragas.metrics.collections.DataCompyScore`
- **SQL 查询等价性**，`ragas.metrics.collections.SQLSemanticEquivalence`

#### 评分细则指标（已迁移） { #rubric-metrics-migrated }
- **DomainSpecificRubrics** - `ragas.metrics.collections.DomainSpecificRubrics`
- **InstanceSpecificRubrics** - `ragas.metrics.collections.InstanceSpecificRubrics`

#### 字符串与自然语言处理指标（已迁移） { #string-nlp-metrics-migrated }
- **CHRF 得分**，`ragas.metrics.collections.CHRFScore`，字符 n-gram F 分数
- **引文对齐**，`ragas.metrics.collections.QuotedSpansAlignment`，验证引用

#### 专门指标的迁移状态 { #specialized-metrics-not-yet-migrated }
- **多模态忠实度**，原文在此记录为待迁移
- **多模态相关性**，原文在此记录为待迁移

!!! note "迁移状态"

    本段反映原文撰写时的迁移状态。仓库现有文档已提供多模态指标的 collections 用法，请结合对应指标页面与当前版本确认。

    旧版指标仍可能显示弃用提示，后续版本的迁移情况以实际导出和指标文档为准。

### 逐步迁移 { #step-by-step-migration }

#### 第一步，更新导入 { #step-1-update-imports }

```python
# v0.3
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
```

```python
# v0.4
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
```

#### 第二步，初始化指标（方式相同） { #step-2-initialize-metrics-no-change-required }

```python
# v0.3
metric = Faithfulness(llm=llm)
```

```python
# v0.4 - Same initialization
metric = Faithfulness(llm=llm)
```

#### 第三步，更新评分调用 { #step-3-update-metric-scoring-calls }

将 `single_turn_ascore(sample)` 替换为 `ascore(**kwargs)`。

```python
# v0.3
sample = SingleTurnSample(
    user_input="What is AI?",
    response="AI is artificial intelligence.",
    retrieved_contexts=["AI is a technology..."],
    ground_truths=["AI definition"]
)

score = await metric.single_turn_ascore(sample)
print(score)  # Output: 0.85
```

```python
# v0.4
result = await metric.ascore(
    user_input="What is AI?",
    response="AI is artificial intelligence.",
    retrieved_contexts=["AI is a technology..."]
)

print(result.value)   # Output: 0.85
print(result.reason)  # Optional: "Response is faithful to context"
```

#### 第四步，处理 MetricResult 对象 { #step-4-handle-metricresult-objects }

v0.4 指标返回 `MetricResult` 对象。

```python
from ragas.metrics.collections.base import MetricResult

result = await metric.ascore(...)

# Access the score
score_value = result.value  # float between 0 and 1

# Access the explanation (if available)
if result.reason:
    print(f"Reason: {result.reason}")

# Convert to float for compatibility
score_float = float(result.value)
```

### 各指标的迁移 { #metric-specific-migrations }

#### 忠实度 { #faithfulness }

**迁移前（v0.3）**
```python
sample = SingleTurnSample(
    user_input="What is machine learning?",
    response="ML is a subset of AI.",
    retrieved_contexts=["ML involves algorithms..."]
)
score = await metric.single_turn_ascore(sample)
```

**迁移后（v0.4）**
```python
result = await metric.ascore(
    user_input="What is machine learning?",
    response="ML is a subset of AI.",
    retrieved_contexts=["ML involves algorithms..."]
)
score = result.value
```

#### AnswerRelevancy

**迁移前（v0.3）**
```python
sample = SingleTurnSample(
    user_input="What is Python?",
    response="Python is a programming language..."
)
score = await metric.single_turn_ascore(sample)
```

**迁移后（v0.4）**
```python
result = await metric.ascore(
    user_input="What is Python?",
    response="Python is a programming language..."
)
score = result.value
```

#### AnswerCorrectness

该指标将 `ground_truths` 改为 `reference`。

**迁移前（v0.3）**
```python
sample = SingleTurnSample(
    user_input="What is AI?",
    response="AI is artificial intelligence.",
    ground_truths=["AI is artificial intelligence and machine learning."]
)
score = await metric.single_turn_ascore(sample)
```

**迁移后（v0.4）**
```python
result = await metric.ascore(
    user_input="What is AI?",
    response="AI is artificial intelligence.",
    reference="AI is artificial intelligence and machine learning."
)
score = result.value
```

#### ContextPrecision

**迁移前（v0.3）**
```python
sample = SingleTurnSample(
    user_input="What is RAG?",
    response="RAG improves LLM accuracy.",
    retrieved_contexts=["RAG = Retrieval Augmented Generation...", "..."],
    ground_truths=["RAG definition"]
)
score = await metric.single_turn_ascore(sample)
```

**迁移后（v0.4）**
```python
result = await metric.ascore(
    user_input="What is RAG?",
    response="RAG improves LLM accuracy.",
    retrieved_contexts=["RAG = Retrieval Augmented Generation...", "..."],
    reference="RAG definition"
)
score = result.value
```

---

## 提示词系统迁移 { #prompt-system-migration }

### 为什么调整提示词 { #why-prompts-changed }

模块化架构将提示词作为**独立组件**，提供以下能力。

- **按指标定制**，每个指标都有明确的提示词接口
- **类型安全**，输入/输出模型定义确切的数据结构
- **复用**，不同指标的提示词类采用一致模式
- **测试**，可以独立生成并检查提示词

v0.3 的字符串或 dataclass 提示词分散在指标实现中。v0.4 将其整合为统一的 `BasePrompt` 架构，并提供专门的输入/输出模型。

### 架构变化 { #architectural-changes_1 }

#### 基础提示词系统 { #base-prompt-system }

| 维度 | v0.3 | v0.4 |
|--------|------|------|
| **提示词定义** | `PydanticPrompt` 数据类或字符串 | 具有 `to_string()` 方法的 `BasePrompt` 类 |
| **输入/输出类型** | 通用 Pydantic 模型 | 指标专用输入/输出模型 |
| **访问方式** | 分散在指标代码中 | 集中在指标的 `util.py` 模块 |
| **定制方式** | 需要较深入的修改 | 继承类并调整 `instruction` 与 `examples` |
| **组织方式** | 混合在指标文件中 | 放入独立的 `util.py` |

### v0.4 中的指标提示词 { #available-metric-prompts-in-v04 }

以下指标具有明确、可定制的提示词。

- **忠实度**，`FaithfulnessPrompt`、`FaithfulnessInput`、`FaithfulnessOutput`
- **上下文召回率**，`ContextRecallPrompt`、`ContextRecallInput`、`ContextRecallOutput`
- **上下文精确率**，`ContextPrecisionPrompt`、`ContextPrecisionInput`、`ContextPrecisionOutput`
- **回答相关性**，`AnswerRelevancyPrompt`、`AnswerRelevancyInput`、`AnswerRelevancyOutput`
- **答案正确性**，`AnswerCorrectnessPrompt`、`AnswerCorrectnessInput`、`AnswerCorrectnessOutput`
- **回答依据充分性**，`ResponseGroundednessPrompt`、`ResponseGroundednessInput`、`ResponseGroundednessOutput`
- **答案准确性**，`AnswerAccuracyPrompt`、`AnswerAccuracyInput`、`AnswerAccuracyOutput`
- **上下文相关性**，`ContextRelevancePrompt`、`ContextRelevanceInput`、`ContextRelevanceOutput`
- **上下文实体召回率**，`ContextEntityRecallPrompt`、`ContextEntityRecallInput`、`ContextEntityRecallOutput`
- **事实正确性**，`ClaimDecompositionPrompt`、`VerificationPrompt` 及相关输入/输出模型
- **噪声敏感度**，`NoiseAugmentationPrompt` 及相关模型
- **摘要得分**，`SummaryScorePrompt`、`SummaryScoreInput`、`SummaryScoreOutput`

### 逐步迁移 { #step-by-step-migration_1 }

#### 第一步，访问指标提示词 { #step-1-access-prompts-in-your-metrics }

```python
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory

# Create metric instance
metric = Faithfulness(llm=llm)

# Access the prompt object
print(metric.prompt)  # <ragas.metrics.collections.faithfulness.util.FaithfulnessPrompt>
```

#### 第二步，查看提示词字符串 { #step-2-view-prompt-strings }

```python
from ragas.metrics.collections.faithfulness.util import FaithfulnessInput

# Create sample input
sample_input = FaithfulnessInput(
    response="The Eiffel Tower is in Paris.",
    context="The Eiffel Tower is located in Paris, France."
)

# Generate prompt string
prompt_string = metric.prompt.to_string(sample_input)
print(prompt_string)
```

#### 第三步，按需定制提示词 { #step-3-customize-prompts-if-needed }

**方式 A，继承默认提示词**

```python
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import FaithfulnessPrompt

# Create custom prompt by subclassing
class CustomFaithfulnessPrompt(FaithfulnessPrompt):
    @property
    def instruction(self):
        return """Your custom instruction here."""

# Apply to metric
metric = Faithfulness(llm=llm)
metric.prompt = CustomFaithfulnessPrompt()
```

**方式 B，为领域评估定制示例**

```python
from ragas.metrics.collections.faithfulness.util import (
    FaithfulnessInput,
    FaithfulnessOutput,
    FaithfulnessPrompt,
    StatementFaithfulnessAnswer,
)

class DomainSpecificPrompt(FaithfulnessPrompt):
    examples = [
        (
            FaithfulnessInput(
                response="ML uses statistical techniques.",
                context="Machine learning is a field that uses algorithms to learn from data.",
            ),
            FaithfulnessOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="ML uses statistical techniques.",
                        reason="Related to learning from data, but context doesn't explicitly mention statistical techniques.",
                        verdict=0
                    ),
                ]
            ),
        ),
    ]

# Apply custom prompt
metric = Faithfulness(llm=llm)
metric.prompt = DomainSpecificPrompt()
```

### 常见提示词定制 { #common-prompt-customizations }

#### 修改指令 { #changing-instructions }

多数指标允许覆盖 instruction 属性。

```python
class StrictFaithfulnessPrompt(FaithfulnessPrompt):
    @property
    def instruction(self):
        return """Be very strict when judging faithfulness.
Only mark statements as faithful (verdict=1) if they are directly stated or strongly implied."""
```

#### 添加领域示例 { #adding-domain-examples }

领域示例可以提高指标准确性，原文示例约提升 10～20%。

```python
class MedicalFaithfulnessPrompt(FaithfulnessPrompt):
    examples = [
        # Medical domain examples here
    ]
```

#### 修改输出格式 { #changing-output-format }

进阶定制可以继承提示词并覆盖 `to_string()`。

```python
class CustomPrompt(FaithfulnessPrompt):
    def to_string(self, input: FaithfulnessInput) -> str:
        # Custom prompt generation logic
        return "..."
```

### 验证自定义提示词 { #verifying-custom-prompts }

使用前，先验证自定义提示词。

```python
# Test prompt generation
sample_input = FaithfulnessInput(
    response="Test response.",
    context="Test context."
)

custom_metric = Faithfulness(llm=llm)
custom_metric.prompt = MyCustomPrompt()

# View the generated prompt
prompt_string = custom_metric.prompt.to_string(sample_input)
print(prompt_string)

# Then use it for evaluation
result = await custom_metric.ascore(
    response="Test response.",
    context="Test context."
)
```

### 迁移 v0.3 自定义提示词 { #migration-from-v03-custom-prompts }

如果 v0.3 中使用了 `PydanticPrompt`，可按以下方式迁移。

**迁移前（v0.3），数据类方式**
```python
from ragas.prompt.pydantic_prompt import PydanticPrompt
from pydantic import BaseModel

class MyInput(BaseModel):
    response: str
    context: str

class MyOutput(BaseModel):
    is_faithful: bool

class MyPrompt(PydanticPrompt[MyInput, MyOutput]):
    instruction = "Check if response is faithful to context"
    input_model = MyInput
    output_model = MyOutput
    examples = [...]
```

**迁移后（v0.4），BasePrompt 方式**
```python
from ragas.metrics.collections.base import BasePrompt
from pydantic import BaseModel

class MyInput(BaseModel):
    response: str
    context: str

class MyOutput(BaseModel):
    is_faithful: bool

class MyPrompt(BasePrompt):
    @property
    def instruction(self):
        return "Check if response is faithful to context"

    @property
    def input_model(self):
        return MyInput

    @property
    def output_model(self):
        return MyOutput

    @property
    def examples(self):
        return [...]

    def to_string(self, input: MyInput) -> str:
        # Generate prompt string from input
        return f"Check if this is faithful: {input.response}"
```

### 使用 BasePrompt.adapt() 适配语言 { #language-adaptation-with-basepromptadapt }

v0.4 为 `BasePrompt` 实例提供 `adapt()` 翻译方法，替代已弃用的 `PromptMixin.adapt_prompts()`。

#### 迁移前（v0.3），PromptMixin 方式 { #before-v03-promptmixin-approach }

```python
from ragas.prompt.mixin import PromptMixin
from ragas.metrics import Faithfulness

# Metrics inherited from PromptMixin to use adapt_prompts
class MyFaithfulness(Faithfulness, PromptMixin):
    pass

metric = MyFaithfulness(llm=llm)

# Adapt ALL prompts to another language
adapted_prompts = await metric.adapt_prompts(
    language="spanish",
    llm=llm,
    adapt_instruction=True
)

# Apply all adapted prompts
metric.set_prompts(**adapted_prompts)
```

**旧版方式的局限**
- 需要继承 mixin，耦合较紧
- 所有提示词一起适配，灵活性不足
- mixin 方法分散在代码中

#### 迁移后（v0.4），BasePrompt.adapt() { #after-v04-basepromptadapt-method }

```python
from ragas.metrics.collections import Faithfulness

# Create metric with default prompt
metric = Faithfulness(llm=llm)

# Adapt individual prompt to another language
adapted_prompt = await metric.prompt.adapt(
    target_language="spanish",
    llm=llm,
    adapt_instruction=True
)

# Apply adapted prompt
metric.prompt = adapted_prompt

# Use metric with adapted language
result = await metric.ascore(
    response="...",
    retrieved_contexts=[...]
)
```

!!! note ""
    原文说明，BasePrompt 的提示词保存与加载计划在后续 v0.4.x 提供，当时仅 PromptMixin 支持。

#### 语言适配示例 { #language-adaptation-examples }

**仅适配示例，保留指令文本**
```python
from ragas.metrics.collections import AnswerRelevancy

metric = AnswerRelevancy(llm=llm)

# Only update language field, keep instruction in English
adapted_prompt = await metric.prompt.adapt(
    target_language="french",
    llm=llm,
    adapt_instruction=False  # Default - just updates language
)

metric.prompt = adapted_prompt
print(metric.prompt.language)  # "french"
```

**同时翻译示例与指令**
```python
# Translate both instruction and examples
adapted_prompt = await metric.prompt.adapt(
    target_language="german",
    llm=llm,
    adapt_instruction=True  # Translate instruction text too
)

metric.prompt = adapted_prompt

# Examples are also automatically translated
# Both instruction and examples in German now
```

**适配自定义提示词**
```python
from ragas.metrics.collections.faithfulness.util import FaithfulnessPrompt

class CustomFaithfulnessPrompt(FaithfulnessPrompt):
    @property
    def instruction(self):
        return "Custom instruction in English"

prompt = CustomFaithfulnessPrompt(language="english")

# Adapt to Italian
adapted = await prompt.adapt(
    target_language="italian",
    llm=llm,
    adapt_instruction=True
)

# Check language was updated
assert adapted.language == "italian"
```

#### 从 v0.3 迁移到 v0.4 { #migration-from-v03-to-v04_1 }

**第一步，移除 PromptMixin 继承**

```python
# v0.3
from ragas.prompt.mixin import PromptMixin
from ragas.metrics import Faithfulness

class MyMetric(Faithfulness, PromptMixin):  # ← Remove PromptMixin
    pass

# v0.4
from ragas.metrics.collections import Faithfulness

# No mixin needed - just use the metric directly
metric = Faithfulness(llm=llm)
```

**第二步，将 adapt_prompts() 替换为 adapt()**

```python
# v0.3
adapted_prompts = await metric.adapt_prompts(
    language="spanish",
    llm=llm,
    adapt_instruction=True
)
metric.set_prompts(**adapted_prompts)

# v0.4
adapted_prompt = await metric.prompt.adapt(
    target_language="spanish",
    llm=llm,
    adapt_instruction=True
)
metric.prompt = adapted_prompt
```

#### 完整迁移示例 { #complete-migration-example }

**迁移前（v0.3）**
```python
from ragas.prompt.mixin import PromptMixin
from ragas.metrics import Faithfulness, AnswerRelevancy

class MyMetrics(Faithfulness, AnswerRelevancy, PromptMixin):
    pass

# Setup
metrics = MyMetrics(llm=llm)

# Adapt multiple metrics to Spanish
adapted = await metrics.adapt_prompts(
    language="spanish",
    llm=best_llm,
    adapt_instruction=True
)

metrics.set_prompts(**adapted)
metrics.save_prompts("./spanish_prompts")
```

**迁移后（v0.4）**
```python
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

# Setup individual metrics
faith_metric = Faithfulness(llm=llm)
answer_metric = AnswerRelevancy(llm=llm)

# Adapt each metric's prompt independently
faith_adapted = await faith_metric.prompt.adapt(
    target_language="spanish",
    llm=best_llm,
    adapt_instruction=True
)
faith_metric.prompt = faith_adapted

answer_adapted = await answer_metric.prompt.adapt(
    target_language="spanish",
    llm=best_llm,
    adapt_instruction=True
)
answer_metric.prompt = answer_adapted

# Use metrics with adapted prompts
faith_result = await faith_metric.ascore(...)
answer_result = await answer_metric.ascore(...)
```

---

## 数据结构变化 { #data-schema-changes }

### SingleTurnSample 更新 { #singleturnsample-updates }

`SingleTurnSample` 数据结构包含以下不兼容变化。

#### `ground_truths` → `reference`

参数 `ground_truths` 统一改名为 `reference`。

**迁移前（v0.3）**
```python
sample = SingleTurnSample(
    user_input="...",
    response="...",
    ground_truths=["correct answer"]  # List of strings
)
```

**迁移后（v0.4）**
```python
sample = SingleTurnSample(
    user_input="...",
    response="...",
    reference="correct answer"  # Single string
)
```

!!! tip ""

    - v0.3 的 `ground_truths` 为**列表**
    - v0.4 的 `reference` 为**单个字符串**
    - 多个参考答案可以分别运行评估

#### 更新后的数据结构 { #updated-schema }

```python
from ragas import SingleTurnSample

# v0.4 complete sample
sample = SingleTurnSample(
    user_input="What is AI?",                      # Required
    response="AI is artificial intelligence.",     # Required
    retrieved_contexts=["Context 1", "Context 2"], # Optional
    reference="Correct definition of AI"           # Optional (was ground_truths)
)
```

### EvaluationDataset 更新 { #evaluationdataset-updates }

使用 `EvaluationDataset` 时，更新加载逻辑。

**迁移前（v0.3）**
```python
dataset = EvaluationDataset(
    samples=[
        SingleTurnSample(
            user_input="Q1",
            response="A1",
            ground_truths=["correct"]
        )
    ]
)
```

**迁移后（v0.4）**
```python
dataset = EvaluationDataset(
    samples=[
        SingleTurnSample(
            user_input="Q1",
            response="A1",
            reference="correct"
        )
    ]
)
```

从 CSV/JSON 加载时，更新数据文件。

**迁移前（v0.3）CSV 格式**
```csv
user_input,response,retrieved_contexts,ground_truths
"Q1","A1","[""ctx1""]","[""correct""]"
```

**迁移后（v0.4）CSV 格式**
```csv
user_input,response,retrieved_contexts,reference
"Q1","A1","[""ctx1""]","correct"
```

---

## 自定义指标 { #custom-metrics }

### 基于 Collections 的指标 { #for-metrics-using-collections-based-architecture }

已经继承 collections 中 `BaseMetric` 的自定义指标，只需少量调整。

```python
from ragas.metrics.collections.base import BaseMetric, MetricResult
from pydantic import BaseModel

class MyCustomMetric(BaseMetric):
    name: str = "my_metric"
    dimensions: list[str] = ["my_dimension"]

    async def ascore(self, **kwargs) -> MetricResult:
        # Your metric logic
        score = 0.85
        reason = "Explanation of the score"
        return MetricResult(value=score, reason=reason)
```

**注意以下事项。**

- 继承 `BaseMetric`
- 实现 `async def ascore(**kwargs)`
- 返回 `MetricResult` 对象
- 使用关键字参数传入数据

### 基于旧版架构的指标 { #for-metrics-using-legacy-architecture }

如果自定义指标继承 `SingleTurnMetric` 或 `MetricWithLLM`，

```python
# v0.3 - Legacy approach
from ragas.metrics.base import MetricWithLLM

class MyMetric(MetricWithLLM):
    async def single_turn_ascore(self, sample: SingleTurnSample) -> float:
        # Extract values from sample
        user_input = sample.user_input
        response = sample.response
        contexts = sample.retrieved_contexts or []

        # Your logic
        return 0.85
```

**迁移步骤**

1. 改为继承 collections 的 `BaseMetric`
2. 调整方法签名，接收关键字参数
3. 返回 `MetricResult`
4. 按需添加 `dimensions` 属性

```python
# v0.4 - Collections approach
from ragas.metrics.collections.base import BaseMetric, MetricResult

class MyMetric(BaseMetric):
    name: str = "my_metric"
    dimensions: list[str] = ["quality"]

    async def ascore(self,
                    user_input: str,
                    response: str,
                    retrieved_contexts: list[str] | None = None,
                    **kwargs) -> MetricResult:
        # Use keyword arguments directly
        contexts = retrieved_contexts or []

        # Your logic
        score = 0.85
        return MetricResult(value=score, reason="Optional explanation")
```

### 提示词系统更新 { #prompt-system-updates }

#### v0.3，基于数据类的提示词 { #v03-dataclass-based-prompts }

```python
from ragas.prompt.pydantic_prompt import PydanticPrompt
from pydantic import BaseModel

class Input(BaseModel):
    query: str
    document: str

class Output(BaseModel):
    is_relevant: bool

class RelevancePrompt(PydanticPrompt[Input, Output]):
    instruction = "Is the document relevant to the query?"
    input_model = Input
    output_model = Output
    examples = [...]
```

#### v0.4，基于函数的提示词 { #v04-function-based-prompts }

自定义场景也可以通过简单函数组织提示词。

```python
def relevance_prompt(query: str, document: str) -> str:
    return f"""Determine if the document is relevant to the query.

Query: {query}
Document: {document}

Respond with YES or NO."""
```

**收益**

- 更简单，易于组合
- 减少样板类定义
- 便于测试和修改
- 支持原生 Python 类型提示

**迁移方法**

- 找到自定义指标中的提示词定义
- 将数据类定义改写为函数
- 更新指标，直接使用函数

---

## 已移除功能 { #removed-features }

以下功能在 v0.4 中已移除，继续使用会报错。

### 函数 { #functions }

**`instructor_llm_factory()`**，已移除

- **合并到** `llm_factory()`
- **迁移方法**，将全部调用替换为 `llm_factory()`
- **影响**，属于直接不兼容变化，无回退实现

**迁移前（v0.3），此写法已失效**

```python
llm = instructor_llm_factory("openai", model="gpt-4o", client=client)
```

**迁移后（v0.4），使用以下写法**
```python
llm = llm_factory("gpt-4o", client=client)
```

### 指标 { #metrics }

以下三个指标已从 collections API 移除，可以使用新的模式实现对应功能。

**1. AspectCritic**，已移除

- **原因**，采用更灵活的离散指标模式
- **替代方式**，使用 `@discrete_metric()` 自定义维度评估
- **用法**
  ```python
  # Instead of AspectCritic, use:
  from ragas.metrics import discrete_metric

  @discrete_metric(name="aspect_critic", allowed_values=["positive", "negative", "neutral"])
  def evaluate_aspect(response: str, aspect: str) -> str:
      # Your evaluation logic
      return "positive"
  ```

**2. SimpleCriteria**，已移除

- **原因**，采用更灵活的离散指标模式
- **替代方式**，使用 `@discrete_metric()` 自定义标准
- **用法**
  ```python
  from ragas.metrics import discrete_metric

  @discrete_metric(name="custom_criteria", allowed_values=["pass", "fail"])
  def evaluate_criteria(response: str, criteria: str) -> str:
      return "pass" if criteria in response else "fail"
  ```

**3. AnswerSimilarity**，已移除

- **原因**，功能由 `SemanticSimilarity` 覆盖
- **直接替代**，`SemanticSimilarity`
- **用法**
  ```python
  # v0.3 - No longer available
  from ragas.metrics import AnswerSimilarity  # ERROR

  # v0.4 - Use this instead
  from ragas.metrics.collections import SemanticSimilarity
  metric = SemanticSimilarity(llm=llm)
  result = await metric.ascore(
      reference="Expected answer",
      response="Actual answer"
  )
  ```

### 已弃用方法的移除 { #deprecated-methods-removed-in-v04 }

**旧版 `Metric.ascore()` 与 `Metric.score()`**

- **时间**，v0.3 标记计划移除，本文按 v0.4 迁移说明介绍
- **原因**，新模式为 collections 中的 `ascore(**kwargs)`
- **迁移方法**，改用 collections 指标

**基于样本对象的旧版方法**

- **`single_turn_ascore(sample: SingleTurnSample)`** 仅用于旧版指标
- **替代方式**，使用 collections 的 `ascore(**kwargs)`

---

## 已弃用功能 { #deprecated-features }

以下功能仍可工作，但会显示弃用警告，计划在**未来版本**移除。

### evaluate() 函数（已弃用） { #evaluate-function-deprecated }

- **状态**，仍可使用，建议迁移
- **原因**，`@experiment()` 提供更结构化的工作流
- **迁移方法**，参阅[从评估到实验](#evaluation-to-experiment)

**迁移前（v0.3），已弃用**
```python
from ragas import evaluate

result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
```

**迁移后（v0.4），推荐方式**
```python
from ragas import experiment
from pydantic import BaseModel

class Results(BaseModel):
    score: float

@experiment(Results)
async def run(row):
    result = await metric.ascore(**row.dict())
    return Results(score=result.value)

result = await run(dataset)
```

### LLM 封装类 { #llm-wrapper-classes }

#### LangchainLLMWrapper（已弃用） { #langchainllmwrapper-deprecated }

- **状态**，仍可使用，建议迁移
- **弃用警告**
  ```
  Direct usage of LangChain LLMs with Ragas prompts is deprecated and will be
  removed in a future version. Use Ragas LLM interfaces instead
  ```
- **迁移方法**，原生客户端结合 `llm_factory()`

**迁移前（v0.3），已弃用**
```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

langchain_llm = ChatOpenAI(model="gpt-4o")
ragas_llm = LangchainLLMWrapper(langchain_llm)
```

**迁移后（v0.4），推荐方式**
```python
from ragas.llms import llm_factory
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="...")
ragas_llm = llm_factory("gpt-4o", client=client)
```

#### LlamaIndexLLMWrapper（已弃用） { #llamaindexllmwrapper-deprecated }

- **状态**，仍可使用，建议迁移
- **警告内容**与 LangchainLLMWrapper 类似
- **迁移方法**，原生客户端结合 `llm_factory()`

**迁移前（v0.3），已弃用**
```python
from ragas.llms import LlamaIndexLLMWrapper
from llama_index.llms.openai import OpenAI

llamaindex_llm = OpenAI(model="gpt-4o")
ragas_llm = LlamaIndexLLMWrapper(llamaindex_llm)
```

**迁移后（v0.4），推荐方式**
```python
from ragas.llms import llm_factory
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="...")
ragas_llm = llm_factory("gpt-4o", client=client)
```

### 嵌入模型迁移 { #embeddings-migration }

#### LangchainEmbeddingsWrapper 与 LlamaIndexEmbeddingsWrapper（已弃用） { #langchainembeddingswrapper-llamaindexembeddingswrapper-deprecated }

- **状态**，仍可使用，但会显示弃用警告
- **原因**，改用直接集成客户端的原生嵌入提供器
- **迁移方法**，参阅[嵌入模型迁移](#embeddings-migration)

v0.4 使用直接集成客户端库的**原生嵌入提供器**替换原有封装方式。

### 主要变化 { #what-changed_2 }

| 维度 | v0.3 | v0.4 |
|--------|------|------|
| **类** | `LangchainEmbeddingsWrapper`, `LlamaIndexEmbeddingsWrapper` | `OpenAIEmbeddings`, `GoogleEmbeddings`, `HuggingFaceEmbeddings` |
| **客户端** | LangChain/LlamaIndex 封装 | OpenAI、Google 等原生客户端 |
| **方法** | `embed_query()`, `embed_documents()` | `embed_text()`, `embed_texts()` |
| **配置** | 包装现有 LangChain 对象 | 直接传入原生客户端 |

#### OpenAI 迁移 { #openai-migration }

**迁移前（v0.3）**
```python
from langchain_openai import OpenAIEmbeddings as LangChainEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

embeddings = LangchainEmbeddingsWrapper(
    LangChainEmbeddings(api_key="sk-...")
)
embedding = embeddings.embed_query("text")
```

**迁移后（v0.4）**
```python
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    client=AsyncOpenAI(api_key="sk-..."),
    model="text-embedding-3-small"
)
embedding = embeddings.embed_text("text")  # Different method name
```

#### Google 嵌入迁移 { #google-embeddings-migration }

**迁移前（v0.3）**
```python
from langchain_community.embeddings import VertexAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

embeddings = LangchainEmbeddingsWrapper(
    VertexAIEmbeddings(model_name="textembedding-gecko@001", project="my-project")
)
```

**迁移后（v0.4）**
```python
from ragas.embeddings import GoogleEmbeddings

embeddings = GoogleEmbeddings(
    model="text-embedding-004",
    use_vertex=True,
    project_id="my-project"
)
```

#### HuggingFace 迁移 { #huggingface-migration }

**迁移前（v0.3）**
```python
from ragas.embeddings import HuggingfaceEmbeddings

embeddings = HuggingfaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**迁移后（v0.4）**
```python
from ragas.embeddings import HuggingFaceEmbeddings  # Capitalization changed

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # Optional GPU acceleration
)
```

### 使用 embedding_factory() { #using-embedding_factory }

**迁移前（v0.3）**
```python
from ragas.embeddings import embedding_factory

embeddings = embedding_factory()  # Defaults to OpenAI
```

**迁移后（v0.4）**
```python
from ragas.embeddings import embedding_factory
from openai import AsyncOpenAI

embeddings = embedding_factory(
    provider="openai",
    model="text-embedding-3-small",
    client=AsyncOpenAI(api_key="sk-...")
)
```

### 提示词系统 { #prompt-system }

#### 数据类提示词 PydanticPrompt（已弃用） { #dataclass-based-prompts-pydanticprompt-deprecated }

- **状态**，旧版提示词仍可用，建议迁移
- **弃用方向**，优先使用模块化 BasePrompt 架构
- **迁移方法**，参阅[提示词系统迁移](#prompt-system-migration)

**迁移前（v0.3），已弃用方式**
```python
from ragas.prompt.pydantic_prompt import PydanticPrompt
from pydantic import BaseModel

class Input(BaseModel):
    query: str

class Output(BaseModel):
    is_relevant: bool

class RelevancePrompt(PydanticPrompt[Input, Output]):
    instruction = "Is this relevant?"
    input_model = Input
    output_model = Output
```

**迁移后（v0.4），推荐方式**
```python
# Use BasePrompt classes instead - see Prompt System Migration section
from ragas.metrics.collections.faithfulness.util import FaithfulnessPrompt

class CustomPrompt(FaithfulnessPrompt):
    @property
    def instruction(self):
        return "Your custom instruction here"
```

### 旧版指标方法 { #legacy-metric-methods }

#### `single_turn_ascore(sample)`（已弃用） { #single_turn_ascoresample-deprecated }

- **状态**，仅用于旧版非 collections 指标
- **替代方式**，使用 collections 的 `ascore()`
- **时间计划**，所有指标迁移后，在未来版本移除

**迁移前（v0.3），已弃用**
```python
sample = SingleTurnSample(user_input="...", response="...", ...)
score = await metric.single_turn_ascore(sample)
```

**迁移后（v0.4），推荐方式**
```python
result = await metric.ascore(user_input="...", response="...")
score = result.value
```

#### ContextUtilization

`ContextUtilization` 现在为 `ContextPrecisionWithoutReference` 的兼容封装。

**迁移前（v0.3）**
```python
from ragas.metrics import ContextUtilization
metric = ContextUtilization(llm=llm)
score = await metric.single_turn_ascore(sample)
```

**迁移后（v0.4）**
```python
from ragas.metrics.collections import ContextUtilization
# or use the modern name directly:
from ragas.metrics.collections import ContextPrecisionWithoutReference

metric = ContextUtilization(llm=llm)  # Still works (wrapper)
# or
metric = ContextPrecisionWithoutReference(llm=llm)  # Preferred

result = await metric.ascore(
    user_input="...",
    response="...",
    retrieved_contexts=[...]
)
score = result.value
```

---

## 不兼容变化汇总 { #breaking-changes-summary }

v0.3 到 v0.4 的不兼容变化如下。

| 变化 | v0.3 | v0.4 | 迁移方法 |
|--------|------|------|-----------|
| **评估方式** | `evaluate()` 函数 | `@experiment()` 装饰器 | 参阅[从评估到实验](#evaluation-to-experiment) |
| **指标位置** | `ragas.metrics` | `ragas.metrics.collections` | 更新导入路径 |
| **评分方法** | `single_turn_ascore(sample)` | `ascore(**kwargs)` | 更新调用 |
| **返回类型** | `float` | `MetricResult` | 使用 `.value` |
| **LLM 工厂** | `instructor_llm_factory()` | `llm_factory()` | 使用统一工厂 |
| **嵌入方式** | LangChain 封装类 | 原生提供器 | 参阅[嵌入模型迁移](#embeddings-migration) |
| **嵌入方法** | `embed_query()`、`embed_documents()` | `embed_text()`、`embed_texts()` | 更新调用 |
| **ground_truths 参数** | `ground_truths: list[str]` | `reference: str` | 改名并调整类型 |
| **样本类型** | `SingleTurnSample` | 更新后的 `SingleTurnSample` | 更新创建方式 |
| **提示词系统** | 数据类 | 支持函数式定义 | 重构自定义提示词 |

---

## 弃用与移除 { #deprecations-and-removals }

### v0.4 中移除的功能 { #removed-in-v04 }

以下功能已移除，继续使用会报错。

- **`instructor_llm_factory()`**，改用 `llm_factory()`
- collections 中的 **AspectCritic**，通过离散指标实现
- collections 中的 **SimpleCriteriaScore**，通过离散指标实现
- **AnswerSimilarity**，改用 `SemanticSimilarity`

### 已弃用，计划未来移除 { #deprecated-will-be-removed-in-future-releases }

以下功能仍可工作，但会显示警告。

- **`LangchainLLMWrapper`**，改用 `llm_factory()`
- **`LlamaIndexLLMWrapper`**，改用 `llm_factory()`
- **旧版提示词类**，迁移到新的提示词模式
- 旧版 **`single_turn_ascore()`**，改用 collections 的 `ascore()`

---

## v0.4 新增功能参考 { #new-features-in-v04-reference }

除迁移所需变化外，v0.4 还提供以下能力，可在升级后使用。

- **GPT-5 与 o 系列支持**，自动处理模型参数约束
- **多服务商支持**，单个 `llm_factory()` 支持 Anthropic、Google、Azure 等
- **函数式提示词**，更灵活、易于组合
- **指标装饰器**，通过 `@discrete_metric`、`@numeric_metric`、`@ranking_metric` 简化定制
- **包含理由的 MetricResult**，结构化结果与可选解释
- **指标保存与加载**，方便序列化配置
- **嵌入支持改进**，支持同步与异步操作

详情参阅[上游发布记录](https://github.com/vibrantlabsai/ragas/releases)中的 v0.4 说明。

---

## 自定义指标迁移 { #custom-metrics-migration }

对于已移除的 AspectCritic 和 SimpleCriteria，可以使用 v0.4 装饰器实现替代，也可用于其他自定义指标。

### 离散指标，输出类别 { #discrete-metrics-categorical-outputs }

**迁移前（v0.3），AspectCritic**
```python
from ragas.metrics import AspectCritic
metric = AspectCritic(name="clarity", allowed_values=["clear", "unclear"])
result = await metric.single_turn_ascore(sample)
```

**迁移后（v0.4），@discrete_metric 装饰器**
```python
from ragas.metrics import discrete_metric

@discrete_metric(name="clarity", allowed_values=["clear", "unclear"])
def clarity(response: str) -> str:
    return "clear" if len(response) > 50 else "unclear"

metric = clarity()
result = await metric.ascore(response="...")
print(result.value)  # "clear" or "unclear"
```

离散指标适用于各类分类任务，可替代 AspectCritic 与 SimpleCriteria。

### 数值指标，输出连续值 { #numeric-metrics-continuous-values }

使用 `@numeric_metric`，按数值量表评分。

```python
from ragas.metrics import numeric_metric

@numeric_metric(name="length_score", allowed_values=(0.0, 1.0))
def length_score(response: str) -> float:
    return min(len(response) / 500, 1.0)

# Custom range
@numeric_metric(name="quality_score", allowed_values=(0.0, 10.0))
def quality_score(response: str) -> float:
    return 7.5

metric = length_score()
result = await metric.ascore(response="...")
print(result.value)  # float between 0 and 1
```

### 排序指标，输出有序列表 { #ranking-metrics-ordered-lists }

使用 `@ranking_metric` 为多个条目排序。

```python
from ragas.metrics import ranking_metric

@ranking_metric(name="context_rank", allowed_values=5)
def context_ranking(question: str, contexts: list[str]) -> list[str]:
    """Rank contexts by relevance."""
    scored = [(len(set(question.split()) & set(c.split())), c) for c in contexts]
    return [c for _, c in sorted(scored, reverse=True)]

metric = context_ranking()
result = await metric.ascore(question="...", contexts=[...])
print(result.value)  # Ranked list
```

### 小结 { #summary }

装饰器自动提供验证、类型安全、错误处理和结果封装，将旧版 50 多行的自定义代码简化为约 5～10 行。

---

## 常见问题与解决方法 { #common-issues-and-solutions }

### 问题，导入 instructor_llm_factory 报错 { #issue-importerror-for-instructor_llm_factory }

**错误**
```
ImportError: cannot import name 'instructor_llm_factory' from 'ragas.llms'
```

**解决方法**
```python
# Instead of this
from ragas.llms import instructor_llm_factory

# Use this
from ragas.llms import llm_factory
```

### 问题，指标返回 MetricResult { #issue-metric-returns-metricresult-instead-of-float }

**错误**
```python
score = await metric.ascore(...)
print(score)  # Prints: MetricResult(value=0.85, reason=None)
```

**解决方法**
```python
result = await metric.ascore(...)
score = result.value  # Access the float value
print(score)  # Prints: 0.85
```

### 问题，SingleTurnSample 缺少 ground_truths { #issue-singleturnsample-missing-ground_truths }

**错误**
```
TypeError: ground_truths is not a valid keyword
```

**解决方法**
```python
# Change from
sample = SingleTurnSample(..., ground_truths=["correct"])

# To
sample = SingleTurnSample(..., reference="correct")
```

## 获取帮助 { #getting-help }

迁移遇到问题时，可通过以下方式获取帮助。

1. **查阅文档**
    - [指标文档](../../concepts/metrics/available_metrics/index.md)
    - [Collections API](../../concepts/metrics/overview/index.md)
    - [LLM 配置](../../references/llms.md)

2. **GitHub issues**
    - 搜索[已有 issue](https://github.com/explodinggradients/ragas/issues)
    - 新建 issue，提供迁移相关细节

3. **社区支持**
    - [加入 Discord 社区](https://discord.gg/5djav8GGNZ)
    - 与维护者[预约交流](https://cal.com/shahul-ragas/30min)

---

## 总结 { #summary_1 }

v0.4 以实验为基础，将评估、分析与迭代紧密结合。不兼容变更旨在使 Ragas 更适合系统化实验。

迁移的主要步骤如下。

1. 使用 `llm_factory()` 初始化 LLM
2. 从 `ragas.metrics.collections` 导入指标
3. 将 `single_turn_ascore()` 替换为 `ascore()`
4. 将 `ground_truths` 改为 `reference`
5. 处理 `MetricResult` 对象

这些变化带来以下能力。

- **更好的实验分析**，结构化结果与理由支持深入检查
- **更易组合的 API**，通过关键字参数传入数据
- **集成工作流**，在实验流水线中使用指标
- **功能增强**，支持多服务商与自动约束
- **便于演进**，基于 Instructor 和统一模式

后续版本将继续完善实验架构，提供更多评估管理、分析和迭代功能。

祝迁移顺利！如有问题，欢迎向社区求助。🎉
