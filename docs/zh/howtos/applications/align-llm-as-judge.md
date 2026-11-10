# 对齐 LLM 裁判 { #how-to-align-an-llm-as-a-judge }

本指南介绍如何使用 Ragas 系统化评估 LLM 裁判，并使其判断与领域专家对齐。

- 构建可复用的裁判对齐评估流水线
- 分析裁判与人工标签不一致的规律
- 迭代裁判提示词，提高与专家判断的一致性

## 为什么先对齐 LLM 裁判？ { #why-align-your-llm-judge-first }

评估实验前，需要让 LLM 裁判贴合具体场景。如果裁判与目标不一致，根据它的分数作出的改进可能偏离真正目标。使其与专家判断一致，才能确保优化的是重要的质量维度。这是可靠评估的基础。

!!! tip "深入分析数据的价值"
    对齐过程中，系统化分析数据与失败模式能带来实际业务价值。它促使你检查边界情况、明确评判标准，并理解回答质量的差别。裁判可以扩大这类分析的规模，专家仍需审阅和判断。

## 配置环境 { #setup-your-environment }

我们提供可安装运行的简单模块，便于你专注理解评估过程。

```bash
uv pip install "ragas[examples]"
export OPENAI_API_KEY="your-api-key-here"
```

!!! note "完整代码"
    完整裁判对齐代码见[此目录](https://github.com/vibrantlabsai/ragas/tree/main/examples/ragas_examples/judge_alignment)。

## 理解数据集 { #understand-the-dataset }

使用 [EvalsBench 数据集](https://github.com/vibrantlabsai/EvalsBench/blob/main/data/benchmark_df.csv)，其中包含专家标注的 LLM 业务问题回答。每行包含以下内容。

- `question`，原始问题
- `grading_notes`，优质回答应覆盖的关键要点
- `response`，LLM 生成的回答
- `target`，专家给出的通过/失败二元判断

**下载数据集。**

```bash
# Create datasets folder and download the dataset
mkdir -p datasets
curl -o datasets/benchmark_df.csv https://raw.githubusercontent.com/vibrantlabsai/EvalsBench/main/data/benchmark_df.csv
```

**加载并检查数据集。**

```python
import pandas as pd
from ragas import Dataset

def load_dataset(csv_path: str = None) -> Dataset:
    """Load annotated dataset with human judgments.

    Expected columns: question, grading_notes, response, target (pass/fail)
    """
    path = csv_path or "datasets/benchmark_df.csv"
    df = pd.read_csv(path)

    dataset = Dataset(name="llm_judge_alignment", backend="local/csv")

    for _, row in df.iterrows():
        dataset.append({
            "question": row["question"],
            "grading_notes": row["grading_notes"],
            "response": row["response"],
            "target": (row["target"]),
        })

    return dataset

# Load the dataset
dataset = load_dataset()
print(f"Dataset loaded with {len(dataset)} samples")
```

**数据集示例行**

| question | grading_notes | response | target |
|----------|---------------|----------|---------|
| A 轮融资前，科技初创公司的投前估值主要有哪些方法，区别是什么？ | DCF：未来现金流，需要预测；可比分析：相似公司倍数；VC 方法：营收乘倍数，考虑投后估值；创始人持股重要；战略买家愿意支付更多。 | 确定 A 轮融资前的投前估值是关键步骤……（涵盖 DCF、可比分析和 VC 方法） | pass |
| 订阅制初创公司应优先采用哪些指标和策略，以管理并降低流失率？ | 每月监测流失率，理想低于 5%；提高参与度与新用户引导；平衡 CAC 和 LTV，比例 3:1 以上；尽早建立反馈闭环；主动及时的客户支持。 | 管理并降低订阅业务流失率非常重要……（缺少具体指标和策略） | fail |

同一问题包含多个回答，有些通过，有些失败，有助于裁判区分可接受与不可接受回答之间的细微差别。

!!! info "理解标准标签"
    对齐质量取决于标准标签的质量。生产场景应邀请**主要领域专家**，即最能代表具体场景质量标准的人，例如心理健康 AI 的心理学家、法律 AI 的律师，或客服机器人的客服负责人。其一致判断构成裁判对齐的标准。无须标注全部数据，覆盖多种场景的代表性样本（100～200 条）即可支持可靠对齐。

## 理解评估方法 { #understand-the-evaluation-approach }

本指南评估数据集中已有的回答，使多次评估结果可复现，并将重点放在裁判对齐。

评估流程为：**数据行（问题 + 回答）→ 裁判 → 与人工 target 比较**。

## 定义评估指标 { #define-evaluation-metrics }

裁判对齐需要两个指标。

**主要指标 `accuracy`（LLM 裁判）**，评价回答并返回通过/失败以及理由。

**对齐指标 `judge_alignment`**，检查裁判决定是否与专家判断一致。

### 设置裁判指标 { #setting-up-the-judge-metric }

先定义简单的基线裁判，对照评分说明评估回答。

```python
from ragas.metrics import DiscreteMetric

# Define the judge metric with a simple baseline prompt
accuracy_metric = DiscreteMetric(
    name="accuracy",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\n\nResponse: {response}\nGrading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)
```

### 对齐指标 { #the-alignment-metric }

将裁判决定与人工判断比较。

```python
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

@discrete_metric(name="judge_alignment", allowed_values=["pass", "fail"])
def judge_alignment(judge_label: str, human_label: str) -> MetricResult:
    """Compare judge decision with human label."""
    judge = judge_label.strip().lower()
    human = human_label.strip().lower()

    if judge == human:
        return MetricResult(value="pass", reason=f"Judge={judge}; Human={human}")

    return MetricResult(value="fail", reason=f"Judge={judge}; Human={human}")
```

## 实验函数 { #the-experiment-function }

[实验函数](../../concepts/experimentation.md)调度完整流程，使用裁判评估回答，并衡量与人工判断的一致性。

```python
from typing import Dict, Any
from ragas import experiment
from ragas.metrics import DiscreteMetric
from ragas_examples.judge_alignment import judge_alignment  # The metric we created above

@experiment()
async def judge_experiment(
    row: Dict[str, Any],
    accuracy_metric: DiscreteMetric,
    llm,
):
    """Run complete evaluation: Judge → Compare with human."""
    # Step 1: Get response (in production, this is where you'd call your LLM app)
    # For this evaluation, we use pre-existing responses from the dataset
    app_response = row["response"]

    # Step 2: Judge evaluates the response
    judge_score = await accuracy_metric.ascore(
        question=row["question"],
        grading_notes=row["grading_notes"],
        response=app_response,
        llm=llm,
    )

    # Step 3: Compare judge decision with human target
    alignment = judge_alignment.score(
        judge_label=judge_score.value,
        human_label=row["target"]
    )

    return {
        **row,
        "judge_label": judge_score.value,
        "judge_reason": judge_score.reason,
        "alignment": alignment.value,
        "alignment_reason": alignment.reason,
    }
```

## 运行基线评估 { #run-baseline-evaluation }

### 执行评估流水线并收集结果 { #execute-evaluation-pipeline-and-collect-results }

```python
import os
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas_examples.judge_alignment import load_dataset

# Load dataset
dataset = load_dataset()
print(f"Dataset loaded with {len(dataset)} samples")

# Initialize LLM client
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = llm_factory("gpt-4o-mini", client=openai_client)

# Run the experiment
results = await judge_experiment.arun(
    dataset,
    name="judge_baseline_v1_gpt-4o-mini",
    accuracy_metric=accuracy_metric,
    llm=llm,
)

# Calculate alignment rate
passed = sum(1 for r in results if r["alignment"] == "pass")
total = len(results)
print(f"✅ Baseline alignment: {passed}/{total} passed ({passed/total:.1%})")
```

??? example "📋 基线 v1 输出"

    ```text
    2025-10-08 22:40:00,334 - Loaded dataset with 160 samples
    2025-10-08 22:40:00,334 - Initializing LLM client with model: gpt-4o-mini
    2025-10-08 22:40:01,858 - Running baseline evaluation...
    Running experiment: 100%|████████████████████████| 160/160 [04:35<00:00,  1.72s/it]
    2025-10-08 22:44:37,149 - ✅ Baseline alignment: 121/160 passed (75.6%)
    ```

### 初步表现分析 { #initial-performance-analysis }

CSV 结果包含全部输入（question、grading_notes、response）、人工标签、裁判决定与理由，以及对齐结果。

## 分析错误与失败规律 { #analyze-errors-and-failure-patterns }

基线运行后，分析不一致的样本，理解裁判在哪些情况下偏离专家判断。

**基线一致率为 75.6%，160 条中有 121 条正确。**

查看错误分布。

??? admonition "📋 代码"

    ```python
    import pandas as pd

    # Load results
    df = pd.read_csv('experiments/judge_baseline_v1_gpt-4o-mini.csv')

    # Analyze misalignments
    false_positives = len(df[(df['judge_label'] == 'pass') & (df['target'] == 'fail')])
    false_negatives = len(df[(df['judge_label'] == 'fail') & (df['target'] == 'pass')])

    print(f"False positives (judge too lenient): {false_positives}")
    print(f"False negatives (judge too strict): {false_negatives}")
    ```

    📋 输出

    ```text
    False positives (judge too lenient): 39
    False negatives (judge too strict): 0
    ```

**主要发现**。39 条不一致结果（24.4%）全部为假正例，即裁判判为 pass，而专家判为 fail。基线裁判过于宽松，未识别出遗漏评分说明中关键概念的回答。

### 失败样本 { #sample-failure-cases }

以下回答遗漏关键概念，却被裁判错误地判为通过。

| 评分说明 | 人工标签 | 裁判标签 | 遗漏内容 |
|---------------|-------------|-------------|----------------|
| `估值上限、金额和投后估值是关键；清算优先权通常为 1 倍以上；反稀释需区分完全棘轮与加权平均；投资方董事席位 1～2 个；员工期权池 10～20%。` | fail | pass | 回答看似全面，但人工标注者发现细微遗漏 |
| `对估值的影响：可扩展性、开发成本、集成难度；开源与专有技术；技术债风险；讨论 AWS/GCP/Azure……` | fail | pass | 缺少对投后估值影响的具体讨论 |
| `历史及预测营收；自上而下与自下而上方法；业务进展证据；无偏假设；未来 12～24 个月预测……` | fail | pass | 未明确提到业务进展证据 |

**常见错误规律**

1. 大部分内容已覆盖，但**遗漏一两个具体概念**。
2. **隐含与明确表达的区别**。裁判接受隐含表达，而标准要求明确提及。
3. **未正确理解缩略语**，如 mkt demand 表示市场需求，post-$ 表示投后估值。
4. **忽略重点标记**，`*` 或 `!` 标出的内容通常是必需的。

## 改进裁判提示词 { #improve-the-judge-prompt }

根据错误分析，新提示词需要满足以下要求。

1. **理解评分说明中的缩略语**
2. **识别重点标记**，包括 `*`、`!` 和具体数值
3. **要求覆盖全部概念**
4. **接受语义等价表达**，允许不同措辞表达相同概念
5. **把握严格程度**，避免过于宽松或苛刻

### 创建改进的 v2 提示词 { #create-the-improved-v2-prompt }

定义具有完整评判标准的增强版裁判指标。

```python
from ragas.metrics import DiscreteMetric

# Define improved judge metric with enhanced evaluation criteria
accuracy_metric_v2 = DiscreteMetric(
    name="accuracy",
    prompt="""Evaluate if the response covers ALL the key concepts from the grading notes. Accept semantic equivalents but carefully check for missing concepts.

ABBREVIATION GUIDE - decode these correctly:

• Financial: val=valuation, post-$=post-money, rev=revenue, ARR/MRR=Annual/Monthly Recurring Revenue, COGS=Cost of Goods Sold, Opex=Operating Expenses, LTV=Lifetime Value, CAC=Customer Acquisition Cost
• Business: mkt=market, reg/regs=regulation/regulatory, corp gov=corporate governance, integr=integration, S&M=Sales & Marketing, R&D=Research & Development, acq=acquisition
• Technical: sys=system, elim=elimination, IP=Intellectual Property, TAM=Total Addressable Market, diff=differentiation
• Metrics: NPS=Net Promoter Score, SROI=Social Return on Investment, proj=projection, cert=certification

EVALUATION APPROACH:

Step 1 - Parse grading notes into distinct concepts:

- Separate by commas, semicolons, or line breaks
- Each item is a concept that must be verified
- Example: "*Gross Margin* >40%, CAC, LTV:CAC >3:1" = 3 concepts

Step 2 - For each concept, check if it's addressed:

- Accept semantic equivalents (e.g., "customer acquisition cost" = "CAC")
- Accept implicit coverage when it's clear (e.g., "revenue forecasting" covers "historical vs forecasted rev")
- Be flexible on exact numbers (e.g., "around 40%" acceptable for ">40%")

Step 3 - Count missing concepts:

- Missing 0 concepts = PASS
- Missing 1+ concepts = FAIL (even one genuinely missing concept should fail)
- Exception: If a long list (10+ items) has 1 very minor detail missing but all major points covered, use judgment

CRITICAL RULES:

1. Do NOT require exact wording - "market demand" = "mkt demand" = "demand analysis"

2. Markers (* or !) mean important, not mandatory exact phrases:
   - "*traction evidence*" can be satisfied by discussing metrics, growth, or validation
   - "!unbiased assumptions" can be satisfied by discussing assumption methodology

3. Numbers should be mentioned but accept approximations:
   - "$47B to $10B" can be "$47 billion dropped to around $10 billion"
   - "LTV:CAC >3:1" can be "LTV to CAC ratio of at least 3 to 1" or "3x or higher"

4. FAIL only when concepts are genuinely absent:
   - If notes mention "liquidation prefs, anti-dilution, board seats" but response only has board seats → FAIL
   - If notes mention "scalability, tech debt, IP" but response never discusses technical risks → FAIL
   - If notes mention "GDPR compliance" and response never mentions GDPR or EU regulations → FAIL

5. PASS when ALL concepts present:
   - All concepts covered, even with different wording → PASS
   - Concepts addressed implicitly when clearly implied → PASS
   - Minor phrasing differences → PASS
   - One or more concepts genuinely absent → FAIL

Response: {response}

Grading Notes: {grading_notes}

Are ALL distinct concepts from the grading notes covered in the response (accepting semantic equivalents and implicit coverage)?""",
    allowed_values=["pass", "fail"],
)
```

!!! tip "使用 LLM 优化提示词"
    明确错误规律后，可以使用 LLM 优化提示词。也可以让 LLM 辅助发现错误，并人工复核，确保与标准标签一致。Cursor、Claude Code 等编码智能体，以及 [DSPy](https://github.com/stanfordnlp/dspy) 等框架，都可辅助系统化优化。

## 使用改进提示词重新评估 { #re-run-evaluation-with-improved-prompt }

保持基线配置，只替换为增强版 v2 指标，再次运行。

```python
# Use the same dataset and LLM setup from the baseline evaluation above
results = await judge_experiment.arun(
    dataset,
    name="judge_accuracy_v2_gpt-4o-mini",
    accuracy_metric=accuracy_metric_v2,  # ← Using improved v2 prompt
    llm=llm,
)

passed = sum(1 for r in results if r["alignment"] == "pass")
total = len(results)
print(f"✅ V2 alignment: {passed}/{total} passed ({passed/total:.1%})")
```

??? example "📋 改进版 v2 输出"

    ```text
    2025-10-08 23:42:11,650 - Loaded dataset with 160 samples
    2025-10-08 23:42:11,650 - Initializing LLM client with model: gpt-4o-mini
    2025-10-08 23:42:12,730 - Running v2 evaluation with improved prompt...
    Running experiment: 100%|██████████| 160/160 [04:39<00:00,  1.75s/it]
    2025-10-08 23:46:52,740 - ✅ V2 alignment: 139/160 passed (86.9%)
    ```

**一致率从 75.6% 提升至 86.9%。**

如需继续迭代，可以采取以下步骤。

- 分析剩余错误，判断是假正例还是假负例，并找出规律。
- 标注标签时同时记录理由，帮助改进裁判，也可以作为少样本示例。
- **尝试更强的模型**，如 GPT-5 或 Claude 4.5 Sonnet，通常更适合作为裁判。
- **使用 AI 助手**，本指南通过 Cursor 智能体分析错误并迭代提示词，也可以使用其他编码智能体或 [DSPy](https://github.com/stanfordnlp/dspy) 等框架。
- 一致率连续两三轮不再提高，或达到业务要求时停止。

## 已完成的工作 { #what-youve-accomplished }

你已通过 Ragas 建立系统化评估流水线，能够完成以下操作。

- 用明确指标衡量裁判与专家判断的一致性
- 通过结构化错误分析识别失败规律
- 通过可复现实验跟踪多轮改进

对齐后的裁判为可靠 AI 评估提供基础，可以用于 RAG、智能体工作流和其他 LLM 应用，使指标改进更能反映实际质量提升。
