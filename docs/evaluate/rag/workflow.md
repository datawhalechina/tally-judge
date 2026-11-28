---
title: 从知识库生成题集并评测 RAG
description: 从知识库切片生成带参考答案的题集，接入实际检索和作答函数并评测。
---

只有知识库、还没有标准问答时，可以先从切片生成问题和参考答案。然后让被测系统逐题检索、作答，分别检查它找回了多少参考证据，以及回答是否有实际召回内容作为依据。已有题集的读者可以直接从第二步开始。

## 1. 从切片生成题集

下面两条切片来自同一份退货指南。`chunk_id` 使用检索索引中的唯一标识。之后评测按 ID 比对，如果一侧用文件名、另一侧用切片编号，同一段原文也会被算成漏召回。

```python
from tallyjudge import create_synthesizer

chunks = [
    {"document_id": "returns", "chunk_id": "window", "text": "签收后七天内可申请退货。"},
    {"document_id": "returns", "chunk_id": "refund", "text": "退款退回原支付账户。"},
]


async def make_testset(chunks, judge):
    generator = create_synthesizer("synthesis.testset.single_hop", llm=judge, seed=7)
    return await generator.arun(chunks, testset_size=2)
```

`judge` 是已配置的语言模型。调用 `testset = await make_testset(chunks, judge)` 后，`testset.to_records()` 给出成功生成的记录。每条记录都有问题 `user_input`、参考答案 `reference`、出题所用的原文 `reference_contexts` 和对应的 `reference_context_ids`。单跳生成器每个切片最多尝试一次，这里最多两题。若实际题数较少，用 `testset.candidates` 查看每次尝试及其错误。[完整出题示例](/examples/rag_generate_testset.py)

首次生成题集时，可以逐题对照原文，检查问题能否回答、参考答案是否受原文支持。确认后复用同一份题集，比较不同检索器或回答模型。

## 2. 接入实际检索和作答

下面的 `ask(question)` 是你提供的异步函数。它接收一个问题字符串，用自己的索引检索，再根据命中片段作答，返回包含以下三个字段的字典。

| 字段 | 从被测系统取什么 |
| --- | --- |
| `retrieved_context_ids` | 实际命中的切片 ID 列表，按检索结果排序 |
| `retrieved_contexts` | 同一批切片的正文列表，与 ID 逐项对应 |
| `response` | 被测系统根据这些切片给出的回答字符串 |

```python
from tallyjudge import create_metric, experiment


async def score_testset(dataset, ask, judge):
    @experiment
    async def answer(row):
        return await ask(row["user_input"])

    return await answer.arun(
        dataset,
        metrics=[
            create_metric("metric.retrieval.id_context_recall"),
            create_metric("metric.rag.faithfulness", llm=judge),
        ],
    )
```

在同一个异步入口中调用 `run = await score_testset(testset.to_dataset(), ask, judge)`。实验把原题的参考字段和 `ask` 的返回值配在一起评分。只把问题传给 `ask`，这样即使检索漏掉出题用的切片，也能看到这次漏召回对回答的影响。手头只有现成的召回块时，可以用它们生成回答，再评测回答质量。[根据已召回片段作答的示例](/examples/rag_answer_then_score.py)

## 3. 如何读分数

| 想知道什么 | 怎么看 |
| --- | --- |
| 找齐了出题证据吗？ | `id_context_recall` 比较参考 ID 和实际召回 ID。例如参考证据为 A、B，只命中 A，召回率就是 0.5。|
| 回答里有多少陈述有依据？ | `faithfulness` 把回答拆成陈述，判断每条是否得到实际召回文本的支持。某些要点即使漏答，已给出的陈述仍可能全部有依据。|
| 哪一题拉低了得分？ | 用 `run.row_for(case_id).metrics` 看该题各指标的 `status`、`value` 和 `error`。汇总时同时看 `run.summary` 里的 `mean` 和 `coverage`，前者只统计有效得分，后者给出有效得分占样本的比例。|

要检验回答是否答对参考答案，可再加 `create_metric("metric.rag.answer_accuracy", llm=judge)`。比较两种检索设置时，复用同一份 `dataset`，分别传入各自的 `ask`，便能按同一个 `case_id` 对比。[指标选择](/evaluate/rag/metrics)

## 何时需要建图

如果一块就能回答一道题，上面的单跳流程只需语言模型。要检验系统能否把分散在几段文字里的线索找齐，可以用证据图生成多跳题。例如第一段说明谁设计了某件物品，第二段说明这件物品在哪里，多跳题就可以把两条线索连起来。

从 TallyJudge 仓库运行时，用 `uv sync --extra synthesis-graph --extra openai` 安装建图依赖和示例使用的 OpenAI 客户端，并准备一个语言模型和一个嵌入模型。已有索引切片时，使用 `build_chunk_graph(chunks, llm=judge, embeddings=embeddings)` 保留现有 ID。手头是完整文档时，将正文和文档 ID 封装为 `SourceDocument`，交给 `build_document_graph`。两个构建函数均需 `await`，都从 `tallyjudge.plugins.synthesis.transforms` 导入。

把生成的图传给 `synthesis.testset.rag` 生成器的 `arun`，再将题集交给同一个 `score_testset`。这种生成器按图中可用的题型分配名额，分别向上取整，所以 `testset_size` 是目标值，实际题数以 `generated_count` 为准。[完整文档与切片的建图示例](https://github.com/datawhalechina/tally-judge/blob/main/README.md#从知识库构建单轮-rag-评测集)
