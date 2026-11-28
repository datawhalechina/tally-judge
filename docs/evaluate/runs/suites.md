---
title: 用配置文件复跑评测
description: 在 JSONL 中保存已有回答，用 TOML 固定指标、列映射和版本，再从命令行或 Python 运行。
---

如果同事要对同一批回答复算分数，可以给他两份文件：`samples.jsonl` 放样本，`suite.toml` 记录评测配置。以下示例只使用本地字符串指标，可以在安装了 TallyJudge 的 Python 环境中直接运行。

## 准备两份文件

在一个目录中新建 UTF-8 编码的 `samples.jsonl`。每行是一个独立的 JSON 对象。这里沿用业务导出中的 `answer` 列名，第二条回答与参考答案不同。

```jsonl
{"case_id":"hours","user_input":"哪天闭馆？","answer":"周一","reference":"周一"}
{"case_id":"loans","user_input":"可以借多久？","answer":"7天","reference":"14天"}
```

在同一目录创建 `suite.toml`。

```toml
id = "suite.library.answers"
revision = "1"

[dataset]
path = "samples.jsonl"

[metrics.exact]
plugin = "metric.text.exact_match"
input_map = { response = "answer" }
```

`dataset.path` 相对于主 TOML 所在目录解析。`exact` 是这次评测中给指标起的名字，运行结果也用它作为键。`input_map` 左边是指标要求的参数名，右边是题集的列名。`reference` 与指标同名，无需映射。

## 从命令行运行

切到这两份文件所在目录。在 uv 管理的项目环境中执行：

```bash
uv run tallyjudge suite show suite.toml
uv run tallyjudge suite run suite.toml --store runs
```

第一条命令显示实际生效的配置。第二条为两道题评分，将结果保存到 `runs`，并在终端输出 JSON。`summary.exact.mean` 应为 `0.5`，`summary.exact.coverage` 应为 `1.0`。记下输出中的 `run_id`，以后用 `uv run tallyjudge runs show RUN_ID --store runs` 重新读取，把 `RUN_ID` 换成这次的值。

## 从 Python 运行同一套配置

```python
from tallyjudge import Suite

result = Suite.load("suite.toml").run()
print(result.metadata["suite"])
print(result.summary["exact"]["mean"], result.summary["exact"]["coverage"])
print(result.row_for("loans").metrics["exact"].value)
```

输出中，评测套件为 `suite.library.answers`、修订号为 `1`，借期题得分为 `0.0`。要保存这次 Python 运行，给 `run(backend=store)` 传入[本地存储后端](/evaluate/runs/results)。

需要切换裁判时，可把模型命名在 `[models.judge]`，让需要模型的指标引用 `llm = "judge"`。另建一个 TOML 并通过 `--overlay` 指定，就能在同一份主配置上换模型。覆盖文件里的同名模型或指标整项替换，需写全这一项的插件和参数。可先用 `suite show` 核对最终配置。
