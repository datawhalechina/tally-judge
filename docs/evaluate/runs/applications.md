---
title: 接入自己的应用
description: 用同一份题集调用应用，把真实回答和召回结果交给指标评分。
---

题集通常只有问题、标准答案和参考证据。要测一次新的检索策略，可以用 `@experiment` 对每道题调用应用，让应用返回这次实际找到的材料和给出的答案。已有这些输出的读者可以直接用 `Evaluator` 评分，参见[给第一条 RAG 回答评分](/evaluate/rag/quickstart)。

## 用两道题跑一次应用

下面的图书馆问答应用用本地索引回答问题。一条借期规定已经过时，题集保留了现行规定和它的文档 ID。运行示例只需安装 TallyJudge。

```python
from tallyjudge import Dataset, create_metric, experiment

index = {
    "哪天闭馆？": ("hours", "周一"),
    "可以借多久？": ("old-loans", "7天"),
}
cases = Dataset.from_records([
    {"case_id": "hours", "user_input": "哪天闭馆？",
     "reference": "周一", "reference_context_ids": ["hours"]},
    {"case_id": "loans", "user_input": "可以借多久？",
     "reference": "14天", "reference_context_ids": ["loans"]},
])


@experiment
def answer(row):
    doc_id, passage = index[row["user_input"]]
    return {
        "response": passage,
        "retrieved_contexts": [passage],
        "retrieved_context_ids": [doc_id],
    }


exact = create_metric("metric.text.exact_match")
recall = create_metric("metric.retrieval.id_context_recall")
run = answer.run(cases, metrics=[exact, recall])
for row in run.rows:
    print(row.case_id, row.task.output["response"],
          row.metrics[exact.spec.id].value, row.metrics[recall.spec.id].value)
```

| 题目 ID | 应用回答 | 答案完全匹配 | 参考文档 ID 召回率 |
| --- | --- | ---: | ---: |
| hours | 周一 | 1.0 | 1.0 |
| loans | 7天 | 0.0 | 0.0 |

把 `index[...]` 换成检索和作答的调用即可。向应用传入问题，按召回顺序返回实际使用的文本、文档 ID 和最终答案。`reference` 和 `reference_context_ids` 留在题集中供指标评分。为每题指定长期稳定的 `case_id`，换应用版本或重排题目后还能找到同一道题。

## 选择接入方式

| 应用入口 | 调用方式 |
| --- | --- |
| Python 函数 | 用 `@experiment` 包装函数。函数返回字段字典，返回 `None` 表示跳过该题。异步客户端可写为 `async def` 并用 `await answer.arun(...)` 运行。 |
| HTTP 服务 | 把 `create_runner("runner.http.json", url=...)` 传给 `experiment`。运行器用 POST 发送 JSON 对象，读取响应里的 JSON 对象。 |
| 命令行程序 | 把 `create_runner("runner.command.json", command=[...])` 传给 `experiment`。程序从标准输入读取一个 JSON 对象，将结果对象写到标准输出。 |

HTTP 和命令行运行器默认只发送 `user_input`。应用还需要语言等信息时，用 `input_fields=["user_input", "language"]` 指定要传的全部字段。运行结束后，`run.row_for("loans").task` 保留该题的执行状态和应用输出。要看改版后哪些题变差，接着阅读[保存与比较结果](/evaluate/runs/results)。
