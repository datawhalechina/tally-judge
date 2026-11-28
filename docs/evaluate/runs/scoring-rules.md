---
title: 用业务规则给回答评分
description: 用 Python 核对回复中的工单号，再用有参考量表评估同一条回复给出的处理建议。
---

客服回复有两项验收要求：写对工单号，让用户知道接下来该做什么。第一项可以逐字核对。第二项允许多种措辞，适合把处理政策和评分量表交给裁判模型。两项分开打分后，可以看出是哪条要求没有满足。

## 用函数核对工单号

下面的规则检查回复是否包含完整工单号。`TJ-420` 和 `TJ-42` 是不同的单号。运行这个例子只需安装 `tallyjudge`。

```python
import re

from tallyjudge import Dataset, Evaluator, numeric_metric


@numeric_metric("metric.support.ticket_id", revision="1", value_range=(0.0, 1.0))
def mentions_ticket(response: str, ticket_id: str) -> float:
    return float(ticket_id in re.findall(r"TJ-[0-9]+", response))


dataset = Dataset.from_records([
    {"case_id": "right", "response": "已登记为 TJ-42，请保留凭证。", "ticket_id": "TJ-42"},
    {"case_id": "wrong", "response": "已登记为 TJ-420。", "ticket_id": "TJ-42"},
])
metric = mentions_ticket()
result = Evaluator([metric]).run(dataset)
for row in result.rows:
    print(row.case_id, row.metrics[metric.spec.id].value)
# right 1.0
# wrong 0.0
```

函数的两个参数对应样本中的同名列，类型注解声明接受的数据类型。如果业务改为允许旧单号或使用新的编号格式，要相应地修改规则并更新 `revision`，让缓存和复跑记录能区分两版评分。

## 用五级量表核对处理建议

假设退款政策要求客服说清两件事：退款的到账时限，以及超时后如何查询。下面的函数接收已配置的语言模型，返回同一条回答的等级分和裁判理由。需要接入自己的模型时，可以沿用[快速开始中的裁判配置](/evaluate/rag/quickstart)。

```python
from tallyjudge import Dataset, Evaluator, create_metric
from tallyjudge.sdk import LanguageModel


async def grade_refund_advice(judge: LanguageModel):
    metric = create_metric(
        "metric.custom.rubric_with_reference",
        llm=judge,
        rubrics={
            "score1_description": "处理建议与退款政策相矛盾。",
            "score2_description": "没有错误承诺，但到账时限和超时查询办法均未交代。",
            "score3_description": "准确交代到账时限或超时查询办法其中一项，另一项遗漏。",
            "score4_description": "两项都提到，且无事实错误，但时限或查询步骤不够具体。",
            "score5_description": "准确给出三个工作日的时限，以及超时后凭工单号联系客服的办法。",
        },
    )
    sample = {
        "case_id": "refund-advice",
        "user_input": "退款什么时候到？如果没到怎么办？",
        "response": "会在三个工作日内到账。",
        "reference": "三个工作日内到账。超时请凭工单号联系客服查询。",
    }
    run = await Evaluator([metric]).arun(Dataset.from_records([sample]))
    score = run.row_for("refund-advice").metrics[metric.spec.id]
    return score.value, score.reason
```

在异步调用处使用 `value, reason = await grade_refund_advice(judge)`。按这份量表，示例回答符合第 3 级：时限正确，遗漏了超时查询办法。实际分数和反馈由裁判返回，分数保留 1–5 的原始等级。

同一批使用一套量表时，用构造参数 `rubrics`。如果每道题的验收条件不同，使用 `metric.custom.instance_rubric`，把五档规则放进各行的 `rubrics` 字段。更多可选输出形式见[指标速查](/evaluate/reference/metrics)。需要让其他项目也能按 ID 加载你的函数时，见[实现指标插件](/plugins/metrics)。
