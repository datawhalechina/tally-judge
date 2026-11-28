---
title: 给第一条 RAG 回答评分
description: 配置裁判模型，给一条有参考答案和召回块的回答评分。
---

顾客问“签收后多久可以申请无理由退货？寄回运费由谁承担？”系统找到了退货期限和运费条款，回答时却把“买家承担”写成了“商家承担”。这个例子可以用三项分数分清楚：材料找全了吗，相关材料排在前面吗，回答有材料支持吗？

## 运行示例

准备 Python 3.12 或更高版本和 uv，从仓库根目录运行。打开[完整的单文件示例](/examples/rag_openai_judge.py)，在脚本开头设置 `OPENAI_API_KEY` 和 `JUDGE_MODEL`。使用 OpenAI 兼容服务时，再设置 `OPENAI_BASE_URL`。脚本优先使用终端里已有的同名环境变量。这一步会调用你配置的裁判模型。

```bash
uv run --extra openai examples/rag_openai_judge.py
```

脚本用 `Dataset.from_records` 建立一条样本，把同一个裁判传给三个指标，然后调用 `await Evaluator(metrics).arun(dataset)`。换成自己的样本时，按下面的含义填写。

| 字段 | 示例里放的是什么 |
| --- | --- |
| `case_id` | 这道题的稳定标识 `return-shipping`。 |
| `user_input` | 顾客实际提出的问题。 |
| `reference` | 参考答案，包含“7 天内”和“买家承担运费”两个事实。 |
| `response` | 待测系统实际给出的回答，本例答为“7 天内，商家承担运费”。 |
| `retrieved_contexts` | 检索器实际返回的文本列表，依次是客服时间、退货期限、运费条款。保留原始排名。 |

## 读懂输出

脚本逐项打印指标 ID、状态和分数或诊断信息。如果裁判把期限和运费拆成两项事实，仅判定后两块材料相关，下面是一组可能的结果。

| 指标 | 分数 | 在这个例子里的含义 |
| --- | ---: | --- |
| `metric.rag.context_recall` | 1.0 | 两个参考事实都有材料支持。 |
| `metric.rag.context_precision_with_reference` | 约 0.5833 | 两个相关块排在第 2、3 位，第一条客服时间拉低了排序得分。 |
| `metric.rag.faithfulness` | 0.5 | 回答中的退货期限有依据，运费承担方与材料相反。 |

这道题可以先查回答生成环节：模型拿到了正确的运费条款，仍然答错了。换用其他裁判时，事实的拆分和判定可能不同。

样本增加后，在 `main()` 中用下面的循环替换读取 `result.row_for("return-shipping")` 的循环。保留状态和原因，便于找出哪些样本没有取得分数。

```python
for row in result.rows:
    for metric_id, score in row.metrics.items():
        print(row.case_id, metric_id, score.status, score.value, score.error)
```

如果手头只有检索结果和相关块的 ID，可以先在本地评分，无需模型密钥。

```bash
uv run examples/rag_retrieval_ids.py
```

如果有召回文本、还没有系统回答，可直接运行[先作答再评分的示例](/examples/rag_answer_then_score.py)。只有知识库文件时，从[生成题集并接入检索器](/evaluate/rag/workflow)开始。
