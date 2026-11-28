---
title: 评测框架使用指南
sidebar:
  label: 评测入门
description: 安装 TallyJudge，按样本选择 RAG 或 Agent 指标，运行自己的应用并比较结果。
---

TallyJudge 支持 Python 3.12 及以上版本。在已有的 uv 项目中安装：

```bash
uv add tallyjudge
```

手头已经有回答或运行日志，可以直接评分。想在同一批问题上测试自己的系统，可以先[接入应用](/evaluate/runs/applications)，让它逐题返回实际输出。

## RAG 评测

| 手头有什么 | 从哪里开始 |
| --- | --- |
| 已有问题、参考答案、实际回答和召回块 | [给第一条 RAG 回答评分](/evaluate/rag/quickstart)，用一个运费答错的例子看懂三项分数。 |
| 已有排序后的召回块 ID 和标注的相关块 ID | 直接运行[本地检索示例](/examples/rag_retrieval_ids.py)，计算 ID 精度和召回率。 |
| 只有知识库，还没有评测题和参考答案 | 阅读[从知识库到评测结果](/evaluate/rag/workflow)，从生成题集开始，再接入自己的检索和作答函数。 |
| 已经有了分数，想知道该改检索还是改回答 | 查阅[RAG 指标选择与分数解读](/evaluate/rag/metrics)，按手头的数据选择指标。 |

## Agent 评测

如果已经有一次 Agent 的运行记录，可以直接计算工具调用指标。对于退款、建单等会改变业务状态的任务，还可以对照执行后的实际状态，检查操作是否生效。[用运行记录评测 Agent](/evaluate/agent) · [从日志核对工具参数](/evaluate/agent/traces)

## Text2SQL 评测（规划中）

这个方向关注模型能否把自然语言问题转成符合意图的 SQL。专用指标和示例仍在规划中。

## 配置、运行与结果

| 想完成的事 | 阅读页面 |
| --- | --- |
| 用同一批题测试自己的应用，保留每题的输出 | [接入应用](/evaluate/runs/applications) |
| 把业务验收条件写成函数或五级量表 | [业务评分规则](/evaluate/runs/scoring-rules) |
| 在命令行复跑同一份数据和指标配置 | [用配置文件复跑](/evaluate/runs/suites) |
| 找到两个版本之间哪些题变好了、哪些退步了 | [保存和比较结果](/evaluate/runs/results) |

## 字段与指标速查

接入已有数据时，可用字段映射沿用自己的列名。RAG 样本、Agent 轨迹和预期状态的字段含义见[样本字段约定](/evaluate/reference/fields)。已知指标名称时，可在[内置指标速查](/evaluate/reference/metrics)查它需要哪些列、是否调用模型，以及分数该如何解释。
