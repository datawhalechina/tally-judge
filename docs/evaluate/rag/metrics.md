---
title: 怎样选择 RAG 评测指标
description: 按评测数据选择检索或回答指标，解读逐题分数和覆盖率。
---

同一道退货问题，检索器可能找齐了规则，回答却把承担运费的一方写反了。要定位问题，可以分别检查召回了哪些块、块排在什么位置，以及回答有没有用对证据。

## 有标注的证据块 ID

将检索器实际返回的 `retrieved_context_ids` 与标注的 `reference_context_ids` 比对。下面两个指标都在本地计算，重复 ID 只算一次，交换顺序也不影响分数。

| 指标 ID | 回答的问题 | 计算方法 |
| --- | --- | --- |
| `metric.retrieval.id_context_precision` | 返回的块里有多少是标注证据？ | 命中的 ID 数 ÷ 返回的 ID 数 |
| `metric.retrieval.id_context_recall` | 标注证据找回了多少？ | 命中的 ID 数 ÷ 标注的 ID 数 |

例如，问题只问退货期限，返回了「期限」和「客服时间」，精度为 1/2，召回率为 1。问题同时问期限和运费，只返回「期限」，精度为 1，召回率为 1/2。[运行 ID 检索示例](/examples/rag_retrieval_ids.py)

## 有召回文本或参考答案

以下指标都需要语言模型，`user_input` 都是原问题。按实际具备的数据选择指标，并通过 `create_metric(指标 ID, llm=judge)` 创建。

| 想查什么 | 指标 ID | 除问题外还需要 |
| --- | --- | --- |
| 对参考答案有用的块是否排在前面 | `metric.rag.context_precision_with_reference` | `retrieved_contexts`、`reference` |
| 缺少参考答案时，哪些排在前面的块对系统回答有用 | `metric.rag.context_precision_without_reference` | `retrieved_contexts`、`response` |
| 参考答案里的事实能否从召回块找到依据 | `metric.rag.context_recall` | `retrieved_contexts`、`reference` |
| 系统回答里的陈述能否从召回块找到依据 | `metric.rag.faithfulness` | `retrieved_contexts`、`response` |
| 系统回答是否答对了问题 | `metric.rag.answer_accuracy` | `response`、`reference` |

退货示例按顺序返回了「客服时间」「签收后 7 天内申请」「运费由买家承担」。参考答案包含后两条，系统回答却写成了「7 天内申请，运费由商家承担」。

- **排序精度**。如果裁判对三个块依次判为 `[0, 1, 1]`，两个相关块所在位置的精度分别是 1/2 和 2/3，平均精度为 7/12。把客服时间移到末尾，判定变为 `[1, 1, 0]`，得分为 1。
- **证据覆盖与回答忠实度**。如果裁判把期限和运费各判为一条事实，参考答案的两条都有证据，`context_recall` 为 1。系统回答只有期限得到支持，`faithfulness` 为 1/2。[运行退货评分示例](/examples/rag_openai_judge.py)
- **答案准确度**。`answer_accuracy` 直接比较系统回答和参考答案，交换两者的位置各评一次，每次给 0、2 或 4，再把有效评分的均值除以 4。若一侧用尽尝试仍给出量表外的等级，会采用另一侧的有效评分。因此即使运费答错，也不能从两条事实机械推得准确度为 1/2。

## 怎样读汇总结果

每个指标单独汇总。先看 `result.summary[key]["coverage"]`，再看 `mean`，并用 `result.row_for(case_id).metrics[key]` 查看单题状态。

| 单题结果 | 含义 |
| --- | --- |
| `success`，值为 `0.0` | 有效零分，计入均值。例如两组 ID 都非空却完全没有交集。 |
| `undefined` | 该题没有可计算的分数，原因见 `error.code`。例如返回 ID 为空时精度的原因是 `no_retrieved_ids`，标注 ID 为空时召回率的原因是 `no_reference_ids`。模型排序精度对所有块都判为无关时，原因是 `no_relevant_contexts`。 |
| `error` | 输入校验或评分执行失败，详情见 `error`。例如误把 `retrieved_contexts` 传成了单个字符串。 |

同一指标在四道题上分别得到 1、0、`undefined`、`error`，有效题数就是 2，`coverage` 为 2/4，`mean` 为 (1 + 0)/2。这里的覆盖率表示有多少题拿到了有效分数。检索证据找全了多少要看召回率。

[快速开始](/evaluate/rag/quickstart) · [评测数据字段](/evaluate/reference/fields) · [从文档到评测](/evaluate/rag/workflow)
