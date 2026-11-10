# 评估并改进提示词 { #how-to-evaluate-your-prompt-and-improve-it }

本指南介绍如何使用 Ragas 评估提示词，并通过迭代持续改进。

## 你将完成的工作 { #what-youll-accomplish }
- 根据评估中的错误分析迭代提示词
- 建立明确的提示词选择标准
- 为数据集构建可复用的评估流水线
- 使用 Ragas 实现评估流程

!!! note "完整代码"
    - 数据集与脚本位于仓库的 `examples/iterate_prompt/` 目录
    - 完整代码见 [GitHub](https://github.com/vibrantlabsai/ragas/tree/main/examples/iterate_prompt)

## 任务定义 { #task-definition }
这里使用客服工单分类任务。

- 标签（可多选）为 `Billing`、`Account`、`ProductIssue`、`HowTo`、`Feature`、`RefundCancel`
- 优先级（只选一个）为 `P0`、`P1` 或 `P2`


## 数据集 { #dataset }

我们为该场景创建了合成数据集，每行包含 `id, text, labels, priority`。示例如下。

| id | text（工单内容） | labels（标签） | priority（优先级） |
|----|---------------------------------------------------------------------------------------------------------------------|------------------------|----------|
| 1 | 升级到 Plus 后，银行记录显示同一天扣了两次款，希望退回重复扣款。 | Billing;RefundCancel | P1 |
| 2 | 通过 Okta 完成 SSO 后又跳回 /login；同事可以登录；提示 state 不匹配，无法访问看板。 | Account;ProductIssue | P0 |
| 3 | 需要将看板导出为带评论和页码的 PDF 供审计使用，截止日期是下周。 | HowTo | P2 |

要适配自己的场景，可以创建 `datasets/` 目录并加入 CSV 文件，也可以连接其他后端。详情参阅[核心概念中的评估数据集](../../concepts/components/eval_dataset.md)。

建议优先从应用真实数据中采样。如果暂无真实数据，可以使用 LLM 生成合成数据。推理模型（如配置高推理强度的 gpt-5）有助于生成更准确、复杂的数据。使用前始终应人工检查和验证。

## 在数据集上评估提示词 { #evaluate-your-prompt-on-a-dataset }

### 提示词运行器 { #prompt-runner }

先在一个样本上运行提示词，检查整个流程是否正常。

??? example "查看完整的 v1 提示词"
    ```text
    You categorize a short customer support ticket into (a) one or more labels and (b) a single priority.

    Allowed labels (multi-label):
    - Billing: charges, taxes (GST/VAT), invoices, plans, credits.
    - Account: login/SSO, password reset, identity/email/account merges.
    - ProductIssue: malfunction (crash, error code, won't load, data loss, loops, outages).
    - HowTo: usage questions ("where/how do I…", "where to find…").
    - Feature: new capability or improvement request.
    - RefundCancel: cancel/terminate and/or refund requests.
    - AbuseSpam: insults/profanity/spam (not mild frustration).

    Priority (exactly one):
    - P0 (High): blocked from core action or money/data at risk.
    - P1 (Normal): degraded/needs timely help, not fully blocked.
    - P2 (Low): minor/info/how-to/feature.

    Return exactly in JSON:
    {"labels":[<labels>], "priority":"P0"|"P1"|"P2"}
    ```

```bash
cd examples/iterate_prompt
export OPENAI_API_KEY=your_openai_api_key
uv run run_prompt.py
```

这会对示例样本运行提示词，并打印结果。

??? example "示例输出"
    ```
    $ uv run run_prompt.py

    Test ticket:
    "SSO via Okta succeeds then bounces me back to /login with no session. Colleagues can sign in. I tried clearing cookies; same result. Error in devtools: state mismatch. I'm blocked from our boards."

    Response:
    {"labels":["Account","ProductIssue"], "priority":"P0"}
    ```


### 评分指标 { #metrics-for-scoring }

优先采用简单、与场景相关的指标。更多背景参阅[核心概念中的评估指标](../../concepts/metrics/index.md)。这里分别使用 `labels_exact_match` 和 `priority_accuracy` 两个离散指标，便于分析与修复不同类型的错误。

- `priority_accuracy`，检查预测优先级是否符合预期，确保正确判断紧急程度。
- `labels_exact_match`，检查预测标签集合是否与预期完全一致，帮助发现多标、漏标问题，并衡量分类准确率。

```python
# examples/iterate_prompt/evals.py
import json
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

@discrete_metric(name="labels_exact_match", allowed_values=["correct", "incorrect"])
def labels_exact_match(prediction: str, expected_labels: str):
    try:
        predicted = set(json.loads(prediction).get("labels", []))
        expected = set(expected_labels.split(";")) if expected_labels else set()
        return MetricResult(
            value="correct" if predicted == expected else "incorrect",
            reason=f"Expected={sorted(expected)}; Got={sorted(predicted)}",
        )
    except Exception as e:
        return MetricResult(value="incorrect", reason=f"Parse error: {e}")

@discrete_metric(name="priority_accuracy", allowed_values=["correct", "incorrect"])
def priority_accuracy(prediction: str, expected_priority: str):
    try:
        predicted = json.loads(prediction).get("priority")
        return MetricResult(
            value="correct" if predicted == expected_priority else "incorrect",
            reason=f"Expected={expected_priority}; Got={predicted}",
        )
    except Exception as e:
        return MetricResult(value="incorrect", reason=f"Parse error: {e}")
```

### 实验函数 { #the-experiment-function }

实验函数负责在数据集上运行提示词。更多详情参阅[核心概念中的实验](../../concepts/experimentation.md)。

这里将 `prompt_file` 作为参数，便于比较不同提示词。也可以传入模型、温度等其他参数，测试不同配置。实验时建议每次只修改一个参数。

```python
# examples/iterate_prompt/evals.py
import asyncio, json
from ragas import experiment
from run_prompt import run_prompt

@experiment()
async def support_triage_experiment(row, prompt_file: str, experiment_name: str):
    response = await asyncio.to_thread(run_prompt, row["text"], prompt_file=prompt_file)
    try:
        parsed = json.loads(response)
        predicted_labels = ";".join(parsed.get("labels", [])) or ""
        predicted_priority = parsed.get("priority")
    except Exception:
        predicted_labels, predicted_priority = "", None

    return {
        "id": row["id"],
        "text": row["text"],
        "response": response,
        "experiment_name": experiment_name,
        "expected_labels": row["labels"],
        "predicted_labels": predicted_labels,
        "expected_priority": row["priority"],
        "predicted_priority": predicted_priority,
        "labels_score": labels_exact_match.score(prediction=response, expected_labels=row["labels"]).value,
        "priority_score": priority_accuracy.score(prediction=response, expected_priority=row["priority"]).value,
    }
```

### CSV 数据集加载器 { #dataset-loader-csv }

加载器将数据加载为 Ragas 数据集对象。详情参阅[核心概念中的评估数据集](../../concepts/components/eval_dataset.md)。

```python
# examples/iterate_prompt/evals.py
import os, pandas as pd
from ragas import Dataset

def load_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, "datasets", "support_triage.csv"))
    dataset = Dataset(name="support_triage", backend="local/csv", root_dir=".")
    for _, row in df.iterrows():
        dataset.append({
            "id": str(row["id"]),
            "text": row["text"],
            "labels": row["labels"],
            "priority": row["priority"],
        })
    return dataset
```

### 使用当前提示词运行实验 { #run-the-experiment-using-current-prompt }

```bash
uv run evals.py run --prompt_file promptv1.txt
```

这会在数据集上运行指定提示词，并将结果保存到 `experiments/` 目录。

??? example "示例输出"
    ```
    $ uv run evals.py run --prompt_file promptv1.txt

    Loading dataset...
    Dataset loaded with 20 samples
    Running evaluation with prompt file: promptv1.txt
    Running experiment: 100%|██████████████████████████████████████████████████████████████████| 20/20 [00:11<00:00,  1.79it/s]
    ✅ promptv1: 20 cases evaluated
    Results saved to: experiments/20250826-041332-promptv1.csv
    promptv1 Labels Accuracy: 80.00%
    promptv1 Priority Accuracy: 75.00%
    ```

## 改进提示词 { #improve-the-prompt }

### 分析结果中的错误 { #analyze-errors-from-the-result }

使用表格编辑器打开 `experiments/{timestamp}-promptv1.csv`，查看 `labels_score` 或 `priority_score` 判断错误的样本。

从 promptv1 实验可以看出以下错误规律。

#### 优先级错误，优先级判得过高（P1 → P0） { #priority-errors-over-prioritization-p1-p0 }
模型反复将本应为 P1 的账单问题判为最高优先级 P0。

| 样本 | 问题 | 预期 | 实际 | 规律 |
|------|-------|----------|-----|---------|
| ID 19 | 暂停工作区后自动扣款 | P1 | P0 | 将账单争议判为紧急 |
| ID 1 | 同一天重复扣款 | P1 | P0 | 将账单争议判为紧急 |
| ID 5 | 取消服务并申请退款 | P1 | P0 | 将常规取消判为紧急 |
| ID 13 | 跟进取消请求 | P1 | P0 | 将后续跟进判为紧急 |

**规律**。模型将所有账单、退款和取消问题都视为紧急的 P0，而多数属于常规业务操作，应为 P1。

#### 标签错误，多标与混淆 { #label-errors-over-labeling-and-confusion }

| 样本 | 问题 | 预期 | 实际 | 规律 |
|------|-------|----------|-----|---------|
| ID 9 | 美国用户询问 GST 税务问题 | `Billing;HowTo` | `Billing;Account` | 将信息咨询混淆为账户操作 |
| ID 10 | 转移账户所有权 | `Account` | `Account;Billing` | 提到金额或套餐就加上 Billing |
| ID 20 | 询问 API 速率限制 | `ProductIssue;HowTo` | `ProductIssue;Billing;HowTo` | 提到套餐就加上 Billing |
| ID 16 | 请求离线模式功能 | `Feature` | `Feature;HowTo` | 给功能请求添加 HowTo |

**发现的规律如下。**

1. **多标 Billing**。主要问题与账单无关时也添加 Billing。
2. **混淆 HowTo 与 Account**。将信息咨询误判为账户管理操作。
3. **多标 HowTo**。用户以“如何”提问、实际在请求开发新功能时，也添加 HowTo。



### 改进提示词 { #improve-the-prompt_1 }

根据错误分析，创建 `promptv2_fewshot.txt` 并作针对性改进。可以手动编辑，也可以交给 LLM。本例将错误规律与原始提示词一同提供给 LLM，生成含少样本示例的修订版。

#### promptv2_fewshot 的主要补充 { #key-additions-in-promptv2_fewshot }

**1. 以业务影响为重点，完善优先级判断规则**
```
- P0: Blocked from core functionality OR money/data at risk OR business operations halted
- P1: Degraded experience OR needs timely help BUT has workarounds OR not fully blocked
- P2: Minor issues OR information requests OR feature requests OR non-urgent how-to
```

**2. 采用保守的多标签规则，减少多标**
```
## Multi-label Guidelines
Use single label for PRIMARY issue unless both aspects are equally important:
- Billing + RefundCancel: Always co-label. Cancellation/refund requests must include Billing.
- Account + ProductIssue: For auth/login malfunctions (loops, "invalid_token", state mismatch, bounce-backs)
- Avoid adding Billing to account-only administration unless there is an explicit billing operation

Avoid over-tagging: Focus on which department should handle this ticket first.
```

**3. 用具体场景细化优先级指南**
```
## Priority Guidelines
- Ignore emotional tone - focus on business impact and available workarounds
- Billing disputes/adjustments (refunds, duplicate charges, incorrect taxes/pricing) = P1 unless causing an operational block
- Login workarounds: If Incognito/another account works, prefer P1; if cannot access at all, P0
- Core business functions failing (webhooks, API, sync) = P0
```

**4. 添加带有推理说明的完整示例**
新增 7 个覆盖不同场景的示例，并明确解释分类理由。

```md
## Examples with Reasoning

Input: "My colleague left and I need to change the team lead role to my email address."
Output: {"labels":["Account"], "priority":"P1"}
Reasoning: Administrative role change; avoid adding Billing unless a concrete billing action is requested.

Input: "Dashboard crashes when I click reports tab, but works fine in mobile app."
Output: {"labels":["ProductIssue"], "priority":"P1"}
Reasoning: Malfunction exists but workaround available (mobile app works); single label since primary issue is product malfunction.
```


!!! tip "避免直接将数据集中的样本加入提示词，以免过拟合，影响其他场景的效果。"


### 评估新版提示词 { #evaluate-new-prompt }

创建改进后的 `promptv2_fewshot.txt` 后，使用新提示词运行实验。

```bash
uv run evals.py run --prompt_file promptv2_fewshot.txt
```

这会在相同数据集上评估改进版本，并将结果保存到新的带时间戳文件。

??? example "示例输出"
    ```
    $ uv run evals.py run --prompt_file promptv2_fewshot.txt

    Loading dataset...
    Dataset loaded with 20 samples
    Running evaluation with prompt file: promptv2_fewshot.txt
    Running experiment: 100%|██████████████████████████████████████████████████████████████| 20/20 [00:11<00:00,  1.75it/s]
    ✅ promptv2_fewshot: 20 cases evaluated
    Results saved to: experiments/20250826-231414-promptv2_fewshot.csv
    promptv2_fewshot Labels Accuracy: 90.00%
    promptv2_fewshot Priority Accuracy: 95.00%
    ```

实验在 `experiments/` 中创建与首次运行结构相同的 CSV，便于直接比较。


### 分析并比较结果 { #analyze-and-compare-results }

我们创建了一个简单工具函数，合并多个 CSV 以便比较。

```bash
uv run evals.py compare --inputs experiments/20250826-041332-promptv1.csv experiments/20250826-231414-promptv2_fewshot.csv
```
它会输出每次实验的准确率，并将合并后的 CSV 保存到 `experiments/`。

??? example "示例输出"
    ```bash
    $ uv run evals.py compare --inputs experiments/20250826-041332-promptv1.csv experiments/20250826-231414-promptv2_fewshot.csv

    promptv1 Labels Accuracy: 80.00%
    promptv1 Priority Accuracy: 75.00%
    promptv2_fewshot Labels Accuracy: 90.00%
    promptv2_fewshot Priority Accuracy: 95.00%
    Combined comparison saved to: experiments/20250826-231545-comparison.csv
    ```

可以看到，promptv2_fewshot 的标签和优先级准确率都提高了，但仍有失败样本。继续分析这些错误，可以进一步改进提示词。

当改进趋于停滞，或准确率已达到业务要求时，可以结束迭代。

!!! tip "如果仅调整提示词已难以继续提高准确率，可以尝试使用更强的模型开展实验。"

## 将这一闭环用于自己的场景 { #apply-this-loop-to-your-use-case }
- 创建场景所需的数据集、指标与实验
- 运行评估并分析错误
- 根据错误分析改进提示词
- 重新评估并比较结果
- 改进趋于停滞或达到业务准确率要求时停止

建立数据集与评估闭环后，还可以进一步测试模型等其他参数。

Ragas 自动处理调度、并行执行与结果汇总，让你专注于具体场景的评估。

!!! tip "进阶，对齐 LLM 裁判"
    如果使用基于 LLM 的指标，建议先使裁判与领域专家的判断对齐，以保证评估可靠。参阅[如何对齐 LLM 裁判](align-llm-as-judge.md)。
