# TallyJudge

<p align="center">
  <img src="assets/banner.png" alt="TallyJudge" width="860">
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

默认安装只包含本地指标所需的依赖。需要调用裁判模型时，用 `python -m pip install ".[openai]"` 一并安装 OpenAI Python SDK。

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

下例使用支持 JSON 输出的 Chat Completions 服务。先运行 `python -m pip install ".[openai]"`，再设置环境变量 `OPENAI_API_KEY`（服务密钥）和 `JUDGE_MODEL`（裁判模型名）。使用其他兼容服务时，可通过 `OPENAI_BASE_URL` 指定地址。

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
            temperature=0.0,
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

`temperature` 需要显式指定。不传时插件不发送该参数，由服务端默认值决定（常见为 1.0），裁判的判断会更不稳定。客户端在哪个事件循环里创建，就只能在那个循环里使用：在一个 `asyncio.run(main())` 内创建客户端、连续 `await Evaluator(...).arun(dataset)`，不要用同一个模型反复调用同步的 `run()`。

## 可用指标

| 指标 | 所需字段 | 评分方式 |
|---|---|---|
| `metric.text.exact_match` | `response`、`reference` | 两段文本完全相同得 1 分，否则得 0 分。 |
| `metric.text.string_presence` | `response`、`reference` | 回答中包含参考文本得 1 分，否则得 0 分。 |
| `metric.rag.faithfulness` | `user_input`、`response`、`retrieved_contexts` | 用裁判模型计算有检索依据的陈述占比。检索资料按段落传入字符串列表。 |
| `metric.rag.answer_relevance` | `user_input`、`response` | 裁判模型根据回答反推问题，再用嵌入模型衡量这些问题与原问题的相似度。 |

后续计划增加检索精度、召回率和基于参考答案的正确性评测。

## 用自己的函数评分

带类型注解的函数可以直接变成指标。插件 ID 和评分版本 `revision` 由你声明，修改评分逻辑时提升 `revision`，以免和旧结果混在一起比较。

```python
from tallyjudge import Dataset, Evaluator, numeric_metric


@numeric_metric("metric.library.length_ratio", revision="1", value_range=(0.0, None))
def length_ratio(response: str, reference: str) -> float:
    return len(response) / max(len(reference), 1)


result = Evaluator([length_ratio()]).run(
    Dataset.from_records([{"case_id": "闭馆日", "response": "周一", "reference": "周一"}])
)
print(result.summary["metric.library.length_ratio"]["mean"])  # 1.0
```

`categorical_metric(..., labels=[...])` 用于固定标签的分类结果，`ranking_metric(..., length=n)` 用于固定长度的排序结果。

## 评测自己的应用

`@experiment` 对每行调用一次你的函数（例如调用自己的 RAG 系统），把返回的字段覆盖到这一行上，再交给指标评分。函数返回 `None` 时这一行记为跳过，抛错、超时和取消都保留在结果中。

```python
from tallyjudge import Dataset, create_metric, experiment


@experiment
def answer(row):
    return {"response": "周一"}  # 替换为对你的系统的调用


result = answer.run(
    Dataset.from_records([{"case_id": "闭馆日", "user_input": "哪天闭馆？", "reference": "周一"}]),
    metrics=[create_metric("metric.text.exact_match")],
)
print(result.row_for("闭馆日").task.output, result.summary["metric.text.exact_match"]["mean"])
```

需要等待网络或磁盘的函数请写成 `async def`，并用 `await answer.arun(...)` 运行。

## 保存、读取与对照

结果可以保存到本地目录，在另一个进程中按运行 ID 读回，再按样本 ID 逐行对照两次运行。读取不需要重新创建模型。

```python
from tallyjudge import compare_runs, create_storage_backend

store = create_storage_backend("storage.local.jsonl", root="runs")
result.save(backend=store)  # include_inputs=False 时不保存原始样本
restored = store.load_run(result.run_id)

for case in compare_runs(restored, result).rows:
    print(case.case_id, {key: score.delta for key, score in case.metrics.items()})
```

`compare_runs` 只在输入和评分口径一致、两边都评分成功时给出差值，其余情况在 `changes` 中说明原因。给 `Evaluator` 传入 `backend=store` 可在运行结束时自动保存；`cache=create_cache_backend("cache.local.sqlite", path=..., namespace=...)` 可复用已经算过的分数。

## 命令行

安装后可以用命令行运行本地指标并查看保存的结果。输入文件每行是一个 JSON 样本。

```bash
tallyjudge metrics list
tallyjudge run --input samples.jsonl --metric metric.text.exact_match --store runs
tallyjudge runs show RUN_ID --store runs
```

运行完成时退出码为 0（`record` 模式下个别样本失败仍为 0），运行被停止或读写失败时为 1，参数错误为 2。需要注入模型客户端的指标请使用 Python 接口。

## 参与贡献

欢迎通过 [Issues](https://github.com/datawhalechina/tally-judge/issues) 提交问题和建议，通过 Pull Request 提交改进。开发环境、检查命令和迁移约定见[贡献指南](CONTRIBUTING.md)，设计与路线见[迭代路线](plan/roadmap.zh-CN.md)。原 Ragas 实现保存在 [`legacy-ragas`](https://github.com/datawhalechina/tally-judge/tree/legacy-ragas) 分支，来源与版权说明见 [NOTICE](NOTICE)。
