---
title: 注册与测试
description: 声明插件入口，生成清单，检查安装后的插件发现，并用离线样本验证实际分数。
---

本页从仓库根目录操作，使用已通过 `uv sync` 配好的开发环境和仓库自带的[长度比插件包](https://github.com/datawhalechina/tally-judge/tree/main/examples/length-ratio-plugin)。把插件装进运行评测的同一个 Python 环境后，工厂函数和命令行都能通过 ID 找到它。

## 准备一个可安装的包

```text
examples/length-ratio-plugin/
├── pyproject.toml
└── tallyjudge_length_ratio/
    ├── __init__.py
    ├── metric.py
    └── manifest.py
```

`metric.py` 包含[指标实现](/plugins/metrics)。插件 ID 采用 `<种类>.<包或组织>.<名称>`，本例为 `metric.length_ratio.response_to_reference`。指标在 `MetricSpec.id` 声明 ID，其他种类在实现类上声明 `plugin_id`。额外依赖可以通过 `Plugin(..., requires=("发行包名",))` 声明，并在构造或首次使用时导入。

`pyproject.toml` 声明当前包依赖 TallyJudge，在 `tallyjudge.plugins` 入口组中导出清单。等号左边的 `length_ratio` 是入口名，右边指向 Python 模块及其中的清单对象。

```toml
[project]
name = "tallyjudge-length-ratio"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["tallyjudge"]

[project.entry-points."tallyjudge.plugins"]
length_ratio = "tallyjudge_length_ratio.manifest:PLUGINS"

[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["tallyjudge_length_ratio"]
```

## 生成清单并验证发现

在仓库根目录，以可编辑方式安装示例包。改变实现后，这个环境会直接读取工作区内的代码。命令中的 `--no-sync` 会保留刚安装的本地插件。

```sh
uv pip install -e ./examples/length-ratio-plugin
uv run --no-sync tallyjudge manifest tallyjudge_length_ratio
uv run --no-sync tallyjudge metrics list
```

`manifest` 扫描包的子模块，把类上的声明转成 Python 清单并打印到标准输出。变更插件 ID、种类或依赖后，用命令输出更新 `tallyjudge_length_ratio/manifest.py`。例如在示例包目录运行下面的 Python 片段，先取得生成结果，再以 UTF-8 写入。该脚本及命令行应在同一个已激活的开发环境中运行。

```python
from pathlib import Path
from subprocess import check_output

manifest = check_output(["tallyjudge", "manifest", "tallyjudge_length_ratio"], text=True)
Path("tallyjudge_length_ratio/manifest.py").write_text(manifest, encoding="utf-8")
```

生成的工厂路径类似 `"tallyjudge_length_ratio.metric:LengthRatioMetric"`。这样运行时列举指标只读取清单，到创建实例时才导入实现。包的 `__init__.py` 也应避免主动导入实现和可选 SDK。

在 `metrics list` 的输出中找到该指标后，可以新建 UTF-8 文件 `samples.jsonl`，写入一行 `{"response":"abcd","reference":"ab"}`，然后运行

```sh
uv run --no-sync tallyjudge run --input samples.jsonl --metric metric.length_ratio.response_to_reference --store runs
```

输出中该行应为 `success`，分数为 `2.0`。传给 `--store` 的目录用于读回本次运行。

## 为分数与配置编写测试

下面的测试同时覆盖契约、构造参数和批量评分。显式传入清单可以在单测中隔离本机安装的其他插件；上一节的命令另行验证发行包的入口确实可见。

```python
import asyncio

from tallyjudge import Dataset, Evaluator
from tallyjudge.registry import PluginRegistry
from tallyjudge.sdk.testing import check_metric
from tallyjudge_length_ratio.manifest import PLUGINS


def test_length_ratio():
    plugin_id = "metric.length_ratio.response_to_reference"
    metric = PluginRegistry(PLUGINS).create(plugin_id, cap=1.5)
    row = {"response": "abcd", "reference": "ab"}
    asyncio.run(check_metric(metric, row))

    result = Evaluator([metric]).run(Dataset.from_records([row]))
    score = result.rows[0].metrics[plugin_id]
    assert (score.status, score.value) == ("success", 1.5)
    assert result.metrics[plugin_id]["config_snapshot"] == {"cap": 1.5}
```

将代码保存到 `test_length_ratio.py`，在项目的开发环境运行 `uv run --no-sync pytest test_length_ratio.py`。其他类型插件可使用 `tallyjudge.sdk.testing` 的对应检查。

| 检查 | 覆盖的行为 |
| --- | --- |
| `await check_metric(metric, inputs)` | 返回值满足指标声明，输入未被修改，配置快照可序列化且稳定，缺少必填字段时报错 |
| `await check_language_model(model)` | 模型 ID、用量模式和配置快照有效，能返回给定结构的回答及用量。准备一个能生成 `{"answer":"ok"}` 的本地客户端即可 |
| `check_storage(backend, result)` | 同一运行可重复保存，读回内容一致，缺失时抛出 `FileNotFoundError` |
| `check_cache(cache)` | 未写入的键读回 `None`，写入后的分数可以原样取回 |
| `check_importer(importer, source)` | 转换结果符合声明的 `documents`、`trace` 或 `records` 契约 |

接入 OpenAI 兼容端点的代码可用 `ScriptedClient.chat_reply(...)` 或 `embedding_reply(...)` 排好应答，并通过 `client.calls` 断言请求参数和次数。具体示例见[开发模型适配器](/plugins/model-adapters)。其他契约同样适用，例如配置应拒绝未知字段。事件插件只观察执行，接收器抛异常时框架会记录错误并停止向该接收器发送后续事件。

## 控制加载哪些插件

插件运行时拥有当前 Python 进程的权限，请仅安装可信来源的包。环境变量 `TALLYJUDGE_PLUGINS` 按入口名筛选已安装插件，多个入口用逗号分隔。设为 `length_ratio` 时会发现本例和内置插件，设为空字符串时只使用内置插件。未设置时会查找所有已安装的入口。

如果安装后没有看到插件，先确认命令行和安装命令使用同一环境，再检查入口名、`模块:对象` 路径和上述筛选设置。清单无法加载或 ID 冲突时可查看注册表的警告日志。内置 ID 保留给框架使用。接口版本不受支持时，按 ID 创建插件会抛出 `PluginError`。
