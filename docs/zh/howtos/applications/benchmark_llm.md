# 为具体场景评估新的 LLM { #how-to-evaluate-a-new-llm-for-your-use-case }

新 LLM 发布后，你可能希望判断它在特定场景中是否优于当前模型。本指南介绍如何使用 Ragas 比较两个模型的准确率。

## 你将完成的工作 { #what-youll-accomplish }

完成本指南后，你将获得以下内容。

- 比较两个 LLM 的结构化评估流程
- 模型在真实业务任务上的表现评估
- 支持模型选型的详细结果
- 可在新模型发布时重复运行的评估闭环

## 评估场景 { #the-evaluation-scenario }

我们以折扣计算为例，根据客户资料计算适用折扣，并说明理由。任务需要应用规则和推理，能够反映模型能力差异。

*你可以将这种方法适配到应用中关心的任意场景。*

> **📁 完整代码**。示例源码见 [GitHub](https://github.com/vibrantlabsai/ragas/tree/main/examples/benchmark_llm)。

## 配置环境与 API 访问 { #set-up-your-environment-and-api-access }

先安装包含基准测试示例代码的 ragas-examples 包。

```bash
pip install ragas[examples]
```

然后配置 API 凭据。

```bash
export OPENAI_API_KEY=your_actual_api_key
```

## LLM 应用 { #the-llm-application }

示例包已提供根据业务规则计算客户折扣的简单应用，方便你专注于评估。

系统提示词定义了以下折扣计算规则。

```python
SYSTEM_PROMPT = """
You are a discount calculation assistant. I will provide a customer profile and you must calculate their discount percentage and explain your reasoning.

Discount rules:
- Age 65+ OR student status: 15% discount
- Annual income < $30,000: 20% discount
- Premium member for 2+ years: 10% discount
- New customer (< 6 months): 5% discount

Rules can stack up to a maximum of 35% discount.

Respond in JSON format only:
{
  "discount_percentage": number,
  "reason": "clear explanation of which rules apply and calculations",
  "applied_rules": ["list", "of", "applied", "rule", "names"]
}
"""
```

使用一份示例客户资料测试应用。

```python
from ragas_examples.benchmark_llm.prompt import run_prompt

# Test with a sample customer profile
customer_profile = """
Customer Profile:
- Name: Sarah Johnson
- Age: 67
- Student: No
- Annual Income: $45,000
- Premium Member: Yes, for 3 years
- Account Age: 3 years
"""

result = await run_prompt(customer_profile)
print(result)
```

??? example "📋 输出"
    ```json
    {
      "discount_percentage": 25,
      "reason": "Sarah qualifies for a 15% discount due to age (67). She also gets a 10% discount for being a premium member for over 2 years. The total stacking of 15% and 10% discounts results in 25%. No other discounts apply based on income or account age.",
      "applied_rules": ["Age 65+", "Premium member for 2+ years"]
    }
    ```

## 检查评估数据集 { #examine-the-evaluation-dataset }

本次评估使用合成数据集，涵盖以下情况。

- 结果明确的简单样本
- 接近规则边界的样本
- 信息含糊的复杂场景

每个样本包含以下字段。

- `customer_profile`，输入的客户资料
- `expected_discount`，预期折扣百分比
- `description`，样本复杂程度说明

示例数据结构如下，建议添加 `id` 列便于比较。

| ID | 客户资料 | 预期折扣 | 说明 |
|----|------------------|-------------------|-------------|
| 1 | Martha 是 70 岁的退休人员，喜欢园艺。她近期没有就读课程，年养老金为 50,000 美元，九年前注册服务，从未升级为高级会员。 | 15 | 仅符合老年优惠 |
| 2 | Arjun 19 岁，是全日制计算机科学本科生，兼职年收入约 45,000 美元，一年前开户，没有高级会员资格。 | 15 | 仅符合学生优惠 |
| 3 | Cynthia 是 40 岁的自由艺术家，年收入约 25,000 美元，目前没有在读课程，五年前订阅基础套餐，从未升级为高级会员。 | 20 | 仅符合低收入优惠 |

为自己的场景创建 `datasets/` 目录并添加 CSV 文件。详情参阅[评估数据集](../../concepts/components/eval_dataset.md)。

优先从应用真实数据中采样。如果暂无真实数据，可以用 LLM 生成合成数据。本例任务稍复杂，建议使用 gpt-5-high 等模型生成更准确的数据，并始终进行人工复核。

!!! note
    为便于展示，本例只有约 10 条样本。真实评估可以从 20～30 条开始，逐步扩展到 50～100 条，以获得更可靠的结果。应覆盖常见场景、边界情况和复杂问题。初始准确率无须达到 100%，可以根据结果分析错误，迭代提示词、数据和工具。

### 加载数据集 { #load-dataset }

```python
def load_dataset():
    """Load the dataset from CSV file. Downloads from GitHub if not found locally."""
    import urllib.request
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "datasets", "discount_benchmark.csv")
    # Download dataset from GitHub if it doesn't exist locally
    if not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/vibrantlabsai/ragas/main/examples/ragas_examples/benchmark_llm/datasets/discount_benchmark.csv", dataset_path)
    return Dataset.load(name="discount_benchmark", backend="local/csv", root_dir=current_dir)
```

加载器先检查本地是否已有 CSV，如果没有，会自动从 GitHub 下载。

### 指标函数 { #metrics-function }

选择简单且与场景相关的指标。更多背景参阅[评估指标](../../concepts/metrics/index.md)。下面使用准确性指标给每条回答评分。

```python
@discrete_metric(name="discount_accuracy", allowed_values=["correct", "incorrect"])
def discount_accuracy(prediction: str, expected_discount):
    """Check if the discount prediction is correct."""
    import json

    parsed_json = json.loads(prediction)
    predicted_discount = parsed_json.get("discount_percentage")
    expected_discount_int = int(expected_discount)

    if predicted_discount == expected_discount_int:
        return MetricResult(
            value="correct",
            reason=f"Correctly calculated discount={expected_discount_int}%"
        )
    else:
        return MetricResult(
            value="incorrect",
            reason=f"Expected discount={expected_discount_int}%; Got discount={predicted_discount}%"
        )
```

### 实验结构 { #experiment-structure }

每次模型评估采用以下实验模式。

```python
@experiment()
async def benchmark_experiment(row, model_name: str):
    # Get model response
    response = await run_prompt(row["customer_profile"], model=model_name)

    # Parse response (strict JSON mode expected)
    try:
        parsed_json = json.loads(response)
        predicted_discount = parsed_json.get('discount_percentage')
    except Exception:
        predicted_discount = None

    # Score the response
    score = discount_accuracy.score(
        prediction=response,
        expected_discount=row["expected_discount"]
    )

    return {
        **row,
        "model": model_name,
        "response": response,
        "predicted_discount": predicted_discount,
        "score": score.value,
        "score_reason": score.reason
    }
```

## 运行实验 { #run-experiments }

分别使用基线模型和候选模型运行实验，本例比较以下两个模型。

- 基线模型：`"gpt-4.1-nano-2025-04-14"`
- 候选模型：`"gpt-5-nano-2025-08-07"`

```python
from ragas_examples.benchmark_llm.evals import benchmark_experiment, load_dataset

# Load dataset
dataset = load_dataset()
print(f"Dataset loaded with {len(dataset)} samples")

# Run baseline experiment
baseline_results = await benchmark_experiment.arun(
    dataset,
    name="gpt-4.1-nano-2025-04-14",
    model_name="gpt-4.1-nano-2025-04-14"
)

# Calculate and display accuracy
baseline_accuracy = sum(1 for r in baseline_results if r["score"] == "correct") / len(baseline_results)
print(f"Baseline Accuracy: {baseline_accuracy:.2%}")

# Run candidate experiment
candidate_results = await benchmark_experiment.arun(
    dataset,
    name="gpt-5-nano-2025-08-07",
    model_name="gpt-5-nano-2025-08-07"
)

# Calculate and display accuracy
candidate_accuracy = sum(1 for r in candidate_results if r["score"] == "correct") / len(candidate_results)
print(f"Candidate Accuracy: {candidate_accuracy:.2%}")
```

每次实验在 `experiments/` 下保存一份 CSV，包含逐行结果。

- 字段包括 id、model、response、predicted_discount、score 和 score_reason

??? example "实验输出示例（为便于阅读，仅展示部分列）"
    | ID | 说明 | 预期 | 预测 | 得分 | 评分理由 |
    |----|-------------|----------|-----------|-------|--------------|
    | 1 | 仅老年优惠 | 15 | 15 | correct | 正确计算折扣为 15% |
    | 2 | 仅学生优惠 | 15 | 5 | incorrect | 预期 15%，实际 5% |
    | 3 | 仅低收入优惠 | 20 | 20 | correct | 正确计算折扣为 20% |
    | 4 | 老年、低收入、新客户（触及上限） | 35 | 35 | correct | 正确计算折扣为 35% |
    | 6 | 仅高级会员满两年优惠 | 10 | 15 | incorrect | 预期 10%，实际 15% |


!!! note
    尽可能固定并记录确切的模型快照或版本，例如 `gpt-4o-2024-08-06`。服务商可能更新模型别名对应的版本，表现也会随之变化。可在服务商文档中查看可用快照，例如 [OpenAI 模型目录](https://platform.openai.com/docs/models)。将快照写入结果，便于公平比较和复现。


## 比较结果 { #compare-results }

运行不同模型后，并列比较结果。

```python
from ragas_examples.benchmark_llm.evals import compare_inputs_to_output

# Compare the two experiment results
# Update these paths to match your actual experiment output files
output_path = compare_inputs_to_output(
    inputs=[
        "experiments/gpt-4.1-nano-2025-04-14.csv",
        "experiments/gpt-5-nano-2025-08-07.csv"
    ]
)

print(f"Comparison saved to: {output_path}")
```

比较工具会完成以下操作。

- 读取两份实验文件
- 输出每个模型的准确率
- 创建并列展示结果的新 CSV

比较文件展示以下信息。

- 测试用例详情，包括客户资料与预期折扣
- 每个模型的回答、是否正确及评分理由

??? example "📋 输出"
    ```
    gpt-4.1-nano-2025-04-14 Accuracy: 50.00%
    gpt-5-nano-2025-08-07 Accuracy: 90.00%
    Comparison saved to: experiments/20250820-150548-comparison.csv
    ```

### 使用合并 CSV 分析结果 { #analyze-results-with-the-combined-csv }

在本次示例运行中，

- 筛选两模型表现不同的样本，可发现“老年且为新客户”“学生且为新客户”“仅学生优惠”“仅高级会员满两年优惠”等情况。
- 模型回答中的 reason 字段展示生成结果的原因。

??? example "比较 CSV 的示例行（为便于阅读，仅展示部分列）"
    | id | customer_profile | description | expected_discount | gpt-4.1-nano-2025-04-14_score | gpt-5-nano-2025-08-07_score | gpt-4.1-nano-2025-04-14_score_reason | gpt-5-nano-2025-08-07_score_reason | gpt-4.1-nano-2025-04-14_response | gpt-5-nano-2025-08-07_response |
    |---:|---|---|---:|---|---|---|---|---|---|
    | 2 | Arjun 19 岁，全日制计算机科学本科生，兼职年收入约 45,000 美元，一年前开户，没有高级会员资格。 | 仅学生优惠 | 15 | incorrect | correct | 预期折扣 15%，实际 0% | 正确计算折扣为 15% | ...reason="Arjun 19 岁，不符合老年优惠；年收入超过 30,000 美元；开户已超过六个月；没有高级会员及其他适用优惠。"... | ...reason="Arjun 为 19 岁本科生，符合 15% 学生折扣。"... |
    | 6 | Leonardo 64 岁，下月满 65 岁，收入恰为 30,000 美元，高级会员已持续两年七个月，使用服务五年。 | 仅高级会员满两年优惠 | 10 | incorrect | correct | 预期折扣 10%，实际 25% | 正确计算折扣为 10% | ...reason="Leonardo 即将满 65 岁，符合 15% 老年优惠；另有高级会员满两年优惠。"... | ...reason="Leonardo 目前 64 岁，下月满 65 岁。高级会员满两年，折扣 10%。"... |

!!! tip "新模型发布后重新运行"
    将评估流程加入项目后，就能重复检查。新 LLM 发布时，将其设为候选模型，运行相同评估，与当前基线比较。


## 解读结果并作出决策 { #interpret-results-and-make-your-decision }

### 关注哪些数据 { #what-to-look-at }
- **基线准确率**、**候选准确率**以及**两者差值**。
  - 本次示例为基线 50%（5/10），候选 90%（9/10），提高 40 个百分点。

### 如何阅读结果行 { #how-to-read-the-rows }
- 优先查看两模型判断不同的样本。
- 根据 score_reason 理解为何被判为正确或错误。
- 寻找规律，例如遗漏优惠叠加、“接近 65 岁”这类边界情况，或收入恰好达到阈值。

### 准确率之外 { #beyond-accuracy }
- 同时检查**成本**与**延迟**，确认更高准确率带来的收益能够覆盖代价。

### 作出选择 { #decide }
- 新模型在关键样本上明显更准确，且符合成本和延迟要求时，可以切换。
- 提升很小、关键场景出现失败，或成本与延迟不符合要求时，继续使用当前模型。

在本例中，
- 可以考虑切换到 `gpt-5-nano-2025-08-07`。它将准确率从 50% 提升到 90%，提高 40 个百分点，并修复规则叠加和边界判断等主要问题。如果成本和延迟符合约束，它是更合适的默认选择。

## 适配自己的场景 { #adapting-to-your-use-case }

使用 [GitHub 示例代码](https://github.com/vibrantlabsai/ragas/tree/main/examples/benchmark_llm)作为模板，替换为应用所需的任务和数据。

Ragas 自动处理调度、并行执行与结果汇总，让你专注于具体场景的评估。

