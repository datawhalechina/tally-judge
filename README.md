# TallyJudge

<p align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/tally-judge/ec003395bb37c223fc981e6567162513aec52a71/banner.png" alt="TallyJudge" width="860">
</p>

TallyJudge 是一个评估检索增强生成（RAG）回答的 Python 库。它可以衡量回答中的说法是否有检索资料支持，以及回答是否切题。有参考答案时，也可以直接在本地做文本匹配。

> [!NOTE]
> TallyJudge 源自 Ragas，由 Datawhale 社区独立维护，沿用 Apache License 2.0。[Ragas 项目](https://github.com/vibrantlabsai/ragas) · [许可证](LICENSE)

## 安装

使用 Python 3.12 或更高版本，在当前 Python 环境中从源码安装。

```bash
git clone https://github.com/datawhalechina/tally-judge.git
cd tally-judge
python -m pip install .
```

## 用参考答案给一批回答评分

下面的例子比较图书馆问答系统的两条回答。`response` 是系统给出的回答，`reference` 是参考答案。两个文本指标都在本地计算，无需 API 密钥。

```python
from tallyjudge import Dataset, Evaluator, create_metric

dataset = Dataset.from_records([
    {"case_id": "闭馆日", "response": "周一", "reference": "周一"},
    {"case_id": "开馆时间", "response": "上午九点开放。", "reference": "九点"},
])
metrics = [
    create_metric("metric.text.exact_match"),
    create_metric("metric.text.string_presence"),
]
result = Evaluator(metrics).run(dataset)

for name, stats in result.summary.items():
    print(name, stats["mean"])
```

```text
metric.text.exact_match 0.5
metric.text.string_presence 1.0
```

两条回答都包含参考答案，只有第一条完全相同。`mean` 是有效得分的平均值，`coverage` 是取得有效得分的样本比例。用 `result.row_for("开馆时间").metrics` 可以查看第二条回答的逐项得分。

## 检查回答是否有检索依据

假设图书馆公告只写了工作日的开放时间，而系统回答“周末也开放”。忠实度指标会把回答拆成陈述，请裁判模型逐条判断资料是否支持。

下例使用支持 JSON 输出的 Chat Completions 服务。先运行 `python -m pip install openai`，再设置环境变量 `OPENAI_API_KEY`（服务密钥）和 `JUDGE_MODEL`（裁判模型名）。使用其他兼容服务时，可通过 `OPENAI_BASE_URL` 指定地址。

```python
import asyncio
import os

from openai import AsyncOpenAI
from tallyjudge import create_language_model, create_metric


async def main():
    async with AsyncOpenAI() as client:
        judge = create_language_model(
            "model.openai_compatible.chat_completions",
            client=client,
            model=os.environ["JUDGE_MODEL"],
        )
        faithfulness = create_metric("metric.rag.faithfulness", llm=judge)
        score = await faithfulness.ascore(
            user_input="图书馆什么时候开放？",
            response="图书馆工作日上午九点开放，周末也开放。",
            retrieved_contexts=["图书馆公告：工作日上午九点开放。"],
        )
        print(score.value)


asyncio.run(main())
```

如果裁判把“工作日上午九点开放”和“周末也开放”识别为两条陈述，只判定第一条有依据，这条回答的忠实度就是 `0.5`。

## 可用指标

| 指标 | 所需字段 | 评分方式 |
|---|---|---|
| `metric.text.exact_match` | `response`、`reference` | 两段文本完全相同得 1 分，否则得 0 分。 |
| `metric.text.string_presence` | `response`、`reference` | 回答中包含参考文本得 1 分，否则得 0 分。 |
| `metric.rag.faithfulness` | `user_input`、`response`、`retrieved_contexts` | 用裁判模型计算有检索依据的陈述占比。检索资料按段落传入字符串列表。 |
| `metric.rag.answer_relevance` | `user_input`、`response` | 裁判模型根据回答反推问题，再用嵌入模型衡量这些问题与原问题的相似度。 |

也可以用 Python 函数定义自己的指标。后续计划增加检索精度、召回率和基于参考答案的正确性评测。[问题反馈与功能建议](https://github.com/datawhalechina/tally-judge/issues)
