---
title: 保存与比较结果
description: 读回一次评测，按题比较两版应用的分数和输出，定位变差的答案。
---

图书馆应用更新之后，怎样找出原先答对、现在答错的题？给两版应用使用相同的题集和裁判，把结果写入本地，再按 `case_id` 配对。下面的两版应用只在借期回答上有差别，整个示例可离线运行。

## 保存两次实验，找到退步的那道题

```python
from tallyjudge import (
    Dataset, compare_experiments, create_metric, create_storage_backend, experiment,
)

cases = Dataset.from_records([
    {"case_id": "hours", "topic": "开放时间", "user_input": "哪天闭馆？", "reference": "周一"},
    {"case_id": "loans", "topic": "借阅规则", "user_input": "可以借多久？", "reference": "14天"},
])
exact = create_metric("metric.text.exact_match")
key = exact.spec.id
store = create_storage_backend("storage.local.jsonl", root="runs")


def run_version(version, loan_answer):
    answers = {"哪天闭馆？": "周一", "可以借多久？": loan_answer}

    @experiment(metadata={"app_version": version})
    def answer(row):
        return {"response": answers[row["user_input"]]}

    return answer.run(cases, metrics=[exact], backend=store).run_id


old_id = run_version("v1", "14天")
new_id = run_version("v2", "7天")
old = store.load_run(old_id)
new = store.load_run(new_id)

print(new.run_status, new.summary[key]["mean"], new.summary[key]["coverage"])
report = compare_experiments(old, new)
summary = report.metrics[key]
print(summary.paired, summary.wins, summary.ties, summary.losses)
for row in report.rows:
    delta = row.metrics[key].delta
    if delta is not None and delta < 0:
        print(row.case_id, row.left_task.output["response"],
              row.right_task.output["response"], delta)
```

应看到 `completed 0.5 1.0`，两题都有有效分数，新版答对一题。第二行是 `2 0 1 1`，表示两题配对成功，新版零胜、一平、一负。最后一行 `loans 14天 7天 -1.0` 指出哪道题退步，并保留了两次回答供复核。

## 读懂一次运行

| 想看什么 | 从哪里读取 |
| --- | --- |
| 运行是否走到结束 | `new.run_status`。单题失败也可能出现在正常结束的运行中。 |
| 指标的有效样本数、覆盖率、均值 | `new.summary[key]` 的 `valid_count`、`coverage` 和 `mean`。无有效分数时数值指标的均值为 `None`。各状态的题数在 `status_counts`。 |
| 一道题的实际回答与评分 | `new.row_for("loans").task.output` 和 `new.row_for("loans").metrics[key]`。先看评分的 `status`，再读 `value`。 |
| 按专题排查 | 对保留了输入的结果，用 `create_analyzer("analyzer.summary.by_field", field="topic").analyze([new])["runs"][0]["groups"]` 取得各组的覆盖率和均值。 |

已经拿到的运行结果也可用 `result.save(backend=store)` 保存。记下 `run_id`，另一个进程只要打开同一个存储目录就能读回。`include_inputs=False` 可省略原始题集，应用返回的 `task.output` 仍会保存。要按题集中的专题或语言分组，请保留原始输入。

## 选择比较方式

| 要回答的问题 | 方法 | 如何解释差值 |
| --- | --- | --- |
| 同一批输入用同样的指标重跑，分数有没有变化？ | `compare_runs(left, right)` | 两侧都成功、输入和评分配置等条件一致时才有数值差。条件变化可查看行和指标的 `changes`。 |
| 同一批题换了应用版本，哪些回答变好了？ | `compare_experiments(left, right)` | 用显式 `case_id` 配对，允许两版的回答和召回结果变化。问题、参考数据和评分配置应一致；先看 `paired` 和两侧覆盖率，再看胜平负。 |

分差按右侧减左侧计算。上例的完全匹配指标越高越好，所以负数表示退步。`delta is None` 表示这一对没有可比的数值差，不能计作平局。若要把题集、指标和列映射一起留给同事复跑，可使用[评测套件](/evaluate/runs/suites)。

## 检查重跑波动与比较两版提示词

想知道同一版应用重跑后得分是否稳定，固定题集、业务提示词和评分设置，至少运行两次。接着上例再跑一次 v2：

```python
from tallyjudge import create_analyzer

repeat = store.load_run(run_version("v2", "7天"))
stability = create_analyzer("analyzer.stability.repeats").analyze([new, repeat])
scores = stability["metrics"][key]
print(scores["compared"], scores["max_range"], scores["unstable_cases"])  # 2 0.0 []
```

分析器核对共有样本的输入指纹和共有指标的评分配置。只有每次都成功取得数值分数的题才计入 `compared`。本例两题的最大波动为零；有波动时，`unstable_cases` 列出对应的题目 ID。

要比较两版**业务提示词**，在应用中分别使用各自的提示词，用 `@experiment(metadata={"prompt_version": "v1"})` 等标记版本。保持题目、参考答案和裁判配置一致，按上例调用 `compare_experiments(old, new)`，先读胜平负，再逐题查看回答与分差。
