---
title: 评估 Agent 的工具调用和最终状态
sidebar:
  label: 工具调用与最终状态
description: 用一次退款任务的运行记录核对工具调用和最终业务状态。
---

用户要求给订单退款。Agent 成功查询了订单，也成功提交了退款请求。验收时怎样区分“请求已经提交”和“订单已经退款”？可以用同一条样本分别给工具轨迹和订单状态评分。这两项指标都在本地计算，可以直接使用已有的运行记录。

## 给一次退款操作评分

下面的样本记录了两次成功调用。任务要求订单最终变为 `refunded`，而运行结束后从订单系统读到的状态仍是 `pending`。

```python
from tallyjudge import Dataset, Evaluator, create_metric

sample = {
    "case_id": "refund-42",
    "trace": {
        "run_id": "order-42",
        "steps": [
            {"step_id": "1", "name": "lookup_order", "kind": "tool", "status": "success"},
            {"step_id": "2", "name": "submit_refund", "kind": "tool", "status": "success"},
        ],
    },
    "expected_tools": ["lookup_order", "submit_refund"],
    "expected_state": {"refund_status": "refunded"},
    "final_state": {"refund_status": "pending"},
}

metrics = [
    create_metric("metric.workflow.tool_recall"),
    create_metric("metric.workflow.final_state"),
]
result = Evaluator(metrics).run(Dataset.from_records([sample]))
for name, score in result.row_for("refund-42").metrics.items():
    print(name, score.value, score.details)
```

工具召回率为 `1.0`，两个期望工具都至少成功调用过一次。最终状态得分为 `0.0`，明细中的 `mismatched` 列出 `refund_status`。如果业务目标只是提交退款请求，验收值就应取提交成功后可观察到的状态。换成自己的任务时，由被测应用在验收时读取真实状态，`expected_state` 只写任务关心的字段。

## 还可以检查什么

轨迹中保留了参数时，可以[从日志核对工具参数](/evaluate/agent/traces)，检查退款是否发到了正确的订单。手头是多轮聊天记录时，把消息放进 `messages`，用 `metric.conversation.criteria` 和裁判模型检查整段对话是否满足给定标准。数据格式见[样本字段约定](/evaluate/reference/fields)，指标的输入见[内置指标速查](/evaluate/reference/metrics)。
