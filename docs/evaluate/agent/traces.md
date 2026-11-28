---
title: 从日志核对工具参数
description: 把逐步记录的 JSONL 日志导入为 Agent 轨迹，对同一笔退款分别核对工具名称与订单号。
---

用户要退的是订单 42。Agent 调用了查询和退款两个工具，两次都执行成功。如果它把退款发到了订单 24，只核对工具名称就会漏掉这个错误。

## 导入一次运行的日志

把下面两行存成 UTF-8 文件 `refund-42.jsonl`。每行是一次调用，`arguments` 保留工具实际收到的参数。一次运行内 `step_id` 要唯一，有父子关系的步骤按父步骤在前的顺序写入。

```jsonl
{"step_id":"1","kind":"tool","name":"lookup_order","status":"success","arguments":{"order_id":"42"}}
{"step_id":"2","kind":"tool","name":"create_refund","status":"success","arguments":{"order_id":"24"}}
```

## 对照这笔订单应有的调用

在文件所在目录运行下面的脚本。`expected_tools` 列出任务需要的工具，`expected_calls` 还给出每次调用应传的完整参数。两者都来自验收样本。导入器用文件名去掉扩展名作为 `run_id`，这里也用它定位评分结果。

```python
from tallyjudge import Dataset, Evaluator, create_importer, create_metric

trace = create_importer("importer.jsonl.trace").convert("refund-42.jsonl")
sample = {
    "case_id": trace.run_id,
    "trace": trace.model_dump(mode="json"),
    "expected_tools": ["lookup_order", "create_refund"],
    "expected_calls": [
        {"name": "lookup_order", "arguments": {"order_id": "42"}},
        {"name": "create_refund", "arguments": {"order_id": "42"}},
    ],
}
recall = create_metric("metric.workflow.tool_recall")
arguments = create_metric("metric.workflow.tool_arguments")
run = Evaluator([recall, arguments]).run(Dataset.from_records([sample]))
scores = run.row_for("refund-42").metrics
print(scores[recall.spec.id].value)         # 1.0
print(scores[arguments.spec.id].value)      # 0.5
print(scores[arguments.spec.id].details)    # {'unmatched': ['create_refund']}
```

两个期望工具都成功调用过，所以工具召回率是 1。两次调用里只有查询的工具名和参数同时匹配，所以参数得分是 ½。重复的期望调用需要分别匹配。如果已记录的成功工具调用缺少参数，参数指标返回 `undefined`，原因码为 `arguments_not_recorded`。把未采集的参数补成空字典会改变这条日志的含义。

## 手头是其他格式时

| 已有记录 | 导入入口 | 评分时可用的信息 |
| --- | --- | --- |
| LangChain v2 事件 | `create_importer("importer.langchain.trace", include_content=True).convert(events)` | 要核对工具参数和返回内容时开启 `include_content`。 |
| 使用 OpenInference 属性的 OTLP JSON | `create_importer("importer.otlp.openinference").convert(export)` | 从一次导出的轨迹中读取层级、工具和模型步骤，以及已记录的时间和 token 数。 |

两种入口都返回 `WorkflowTrace`，可以接入上面同一段评分代码。若要判定退款是否真的入账，另需运行结束后的业务状态，见[评估最终状态](/evaluate/agent)。聊天消息、轨迹和验收字段的区别见[样本字段约定](/evaluate/reference/fields)。
