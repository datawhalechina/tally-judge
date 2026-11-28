---
title: 插件开发
sidebar:
  label: 插件开发入门
description: 选择扩展种类，了解 TallyJudge SDK 的接口边界、版本和插件发现方式。
---

给团队加一个评分公式时，可以先写一个 Python 类，直接交给 `Evaluator`。需要在多个项目间复用，或希望用户按 ID 从命令行调用时，再把它打成带插件入口的包。仓库里的[答案长度比示例](https://github.com/datawhalechina/tally-judge/tree/main/examples/length-ratio-plugin)包含这两种用法需要的全部文件。

## 从你要扩展的能力开始

| 要做的事 | 阅读 |
| --- | --- |
| 编写评分公式，声明输入字段、分数范围或裁判模型 | [开发自定义指标](/plugins/metrics) |
| 接入已有语言模型或向量服务，记录请求和用量 | [开发模型适配器](/plugins/model-adapters) |
| 让其他项目按 ID 创建插件，验证安装后的发现与评分 | [注册与测试](/plugins/registration-and-tests) |

## 可扩展的种类

| 种类 | 交付的能力 | 契约 |
| --- | --- | --- |
| `metric` | 给一行样本评分 | `MetricBase`、`MetricSpec` |
| `model` | 按给定结构生成回答 | `LanguageModel`、`LanguageModelBase` |
| `embedding` | 按输入顺序返回文本向量 | `EmbeddingModel` |
| `synthesis` | 从文档生成带参考答案的数据集 | `Synthesizer`，继承 `SynthesizerBase` 实现 `arun` 后即可使用同步入口 |
| `storage` | 保存和读回运行结果 | `StorageBackend` |
| `cache` | 复用已算出的分数 | `ScoreCache` |
| `event` | 观察运行、评分与模型请求事件 | `EventSink` |
| `optimization` | 通过多轮评测改进业务提示词 | `PromptOptimizer`，也可继承 `PromptOptimizerBase` 实现 `arun` |
| `importer` | 把其他框架的数据转成文档、轨迹或样本记录 | `Importer`，声明 `produces="documents"`、`"trace"` 或 `"records"` |
| `runner` | 用一行输入调用被测应用，返回它产生的字段 | `Runner` |
| `analyzer` | 读取一次或多次运行结果，产出报告 | `Analyzer` |

## 稳定的导入入口

插件实现从 `tallyjudge.sdk` 导入公共类型，契约检查从 `tallyjudge.sdk.testing` 导入。两种领域契约由各自的能力包提供，分别位于 `tallyjudge.plugins.synthesis.contract` 和 `tallyjudge.plugins.optimization.contract`。执行引擎、注册表内部实现及其他插件的私有模块都可能随版本调整。

TallyJudge 目前处于 Alpha 阶段。插件 API v1 覆盖 SDK 导出名、`Plugin` 和 `PluginSpec` 等记录的字段，以及公共契约的方法签名。项目约定同一接口版本可增加能力，删除名字或改变签名时提升接口版本。仓库中的 `tests/tallyjudge/contracts/sdk_surface.json` 记录受检查的接口。清单的接口版本与指标自己的评分版本 `revision` 分别维护。

## 自定义种类

如果你的能力需要一套独立的契约，可以在包的子模块中定义可运行时检查的协议，再声明一个 `KindSpec`。例如，模拟器需要实现 `reply`。

```python
from typing import Protocol, runtime_checkable

from tallyjudge.sdk import KindSpec, Plugin


@runtime_checkable
class Simulator(Protocol):
    def reply(self, message: str) -> str: ...


SIMULATOR = KindSpec("simulator", Simulator)


class EchoSimulator:
    plugin = Plugin("simulator")
    plugin_id = "simulator.example.echo"

    def reply(self, message: str) -> str:
        return message
```

生成清单时，命令会一起收集种类声明和实现。安装后，宿主可通过 `registry.create_for_kind("simulator", "simulator.example.echo")` 构造符合协议的实例。种类名须在当前环境中唯一。具体的清单、入口名和加载限制见[注册与测试](/plugins/registration-and-tests)。
