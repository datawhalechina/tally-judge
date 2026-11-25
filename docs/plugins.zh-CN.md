# 编写 TallyJudge 插件

插件是一个独立的 Python 包。它不需要修改 TallyJudge 的代码，安装后就能被 `create_metric` 等工厂函数和命令行使用。完整的最小示例在 [`examples/length-ratio-plugin`](../examples/length-ratio-plugin)。

> [!NOTE]
> 项目处于 Alpha 阶段。插件接口版本目前是 1，在首个正式版本之前仍可能调整。

接口版本 1 覆盖三样东西：`tallyjudge.sdk` 公开的名字、`Plugin` 和 `PluginSpec` 等记录的字段、各契约的方法签名。计划的规则是：在同一个版本内这些只增不改；删除名字或改变签名时把版本提到 2，旧版本的清单会被拒绝并给出明确的报错。这份范围记录在 `tests/tallyjudge/contracts/sdk_surface.json` 里，任何改动都会让测试失败，必须有意更新。

## 插件能导入什么

只导入 `tallyjudge.sdk`。它包含数据类型、通用种类的契约、错误类型和测试工具。`tallyjudge.engine`、`tallyjudge.registry` 和其他插件的内部模块都不在承诺范围内。

有一个例外：种类由某个能力包定义时，契约跟着那个包走。内置的两个是测试集合成（`tallyjudge.plugins.synthesis.contract`）和提示词优化（`tallyjudge.plugins.optimization.contract`），写这两类插件时从这里导入契约。

## 可以写哪些种类

| 种类 | 要实现什么 | 契约 |
|---|---|---|
| `metric` | 声明输入和输出，实现一个评分方法 | `MetricBase`、`MetricSpec` |
| `model` | 按给定的结构生成回答 | `LanguageModel`，通常继承 `LanguageModelBase` |
| `embedding` | 把文本变成向量 | `EmbeddingModel` |
| `synthesis` | 从文档产出带参考答案的数据集 | 由合成能力包定义：`Synthesizer`；继承 `SynthesizerBase` 只需实现 `arun`，同步的 `run` 由基类提供 |
| `storage` | 保存和读取运行结果 | `StorageBackend` |
| `cache` | 复用已经算过的分数 | `ScoreCache` |
| `event` | 接收运行过程中的事件 | `EventSink` |
| `optimization` | 驱动多次运行，改进业务提示词 | 由优化能力包定义：`PromptOptimizer`；继承 `PromptOptimizerBase` 只需实现 `arun` |
| `importer` | 把框架自己的对象转成文档、轨迹或样本记录；声明里用 `produces` 写明是 `documents`、`trace` 还是 `records` | `Importer` |
| `runner` | 打包好的被测应用：输入一行样本，返回它产生的字段 | `Runner` |
| `analyzer` | 读一次或多次运行结果，产出报告 | `Analyzer` |

上面这些种类随 TallyJudge 一起发布，其中合成和优化两个就是用下面的办法由各自的能力包定义的。它们都不合适时，你的包也可以定义自己的种类：写一个契约，再在模块里声明一个 `KindSpec`。

```python
from typing import Protocol, runtime_checkable

from tallyjudge.sdk import KindSpec


@runtime_checkable
class Simulator(Protocol):
    def reply(self, message: str) -> str: ...


SIMULATOR = KindSpec("simulator", Simulator)
```

属于这个种类的插件照常声明 `plugin = Plugin("simulator")` 和 `plugin_id`。`tallyjudge manifest <包名>` 生成的清单会带上这条种类声明；安装后用 `registry.create_for_kind("simulator", plugin_id)` 创建，创建出来的对象必须符合这个契约。种类名不能和已有的重复。

## 写一个指标

```python
from tallyjudge.sdk import MetricBase, MetricResult, MetricSpec


class LengthRatioMetric(MetricBase):
    spec = MetricSpec(
        "metric.length_ratio.response_to_reference",
        "1",
        {"response": str, "reference": str},
        value_range=(0.0, None),
    )

    async def ascore(self, response: str, reference: str) -> MetricResult:
        return MetricResult(len(response) / max(len(reference), 1))
```

- `MetricSpec` 的前两个参数是插件 ID 和评分版本 `revision`。评分逻辑一变就提升 `revision`，不同版本的分数不会被直接相减。
- 第三个参数声明输入字段和类型。评测时内核按它做严格校验，`ascore` 收到的一定是校验过的值。
- 分数没有定义时（例如分母为零）抛 `UndefinedScore("原因代码")`。它会被记为“未定义”，不会算成 0 分。
- 想让使用者看到分数的依据，返回 `MetricResult(value, reason, details={...})`。`details` 是一个 JSON 对象，随运行结果保存。

## 声明配置

把配置写成一个 pydantic 模型，挂在类的 `Config` 属性上，并在 `__init__` 里自己校验。注册表只负责拒绝 `Config` 里没有的字段名。

```python
from pydantic import BaseModel, ConfigDict, Field


class LengthRatioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cap: float | None = Field(default=None, gt=0)


class LengthRatioMetric(MetricBase):
    Config = LengthRatioConfig

    def __init__(self, cap: float | None = None):
        self.cap = LengthRatioConfig(cap=cap).cap

    def config_snapshot(self):
        return {"cap": self.cap}
```

`config_snapshot` 返回会影响分数的全部设置。它随运行结果保存，两次运行的快照不同，就不会给出分差。

## 使用裁判模型

在 `plugin` 里声明依赖，模型会作为同名参数传给构造函数：

```python
class ToneMetric(MetricBase):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})

    def __init__(self, llm):
        self.llm = llm

    async def ascore(self, response: str) -> MetricResult:
        reply = await self.llm.agenerate(PromptInput.from_text(response), Verdict)
        return MetricResult(reply.data.score)
```

`Verdict` 是你定义的 pydantic 模型，模型的回答会按它解析和校验。直接调用 `self.llm.agenerate` 就可以。注入进来的模型已经套上了请求闸门，用量统计、请求预算和取消边界都由它处理，指标里不需要写任何相关代码。如果你的类也会被别人绕过注册表直接构造，保存模型时写 `self.llm = gated(llm)`（`from tallyjudge.sdk import gated`），重复套不会有副作用。

## 清单和入口

在插件类上用 `plugin` 声明它的种类，以及需要注入什么：

```python
from tallyjudge.sdk import MetricBase, MetricSpec, Plugin


class LengthRatioMetric(MetricBase):
    plugin = Plugin("metric")
    spec = MetricSpec("metric.length_ratio.response_to_reference", "1", ...)
```

清单是一个只包含 `PluginSpec` 的模块，由命令从这些声明生成，不要手写：

```bash
tallyjudge manifest tallyjudge_length_ratio > tallyjudge_length_ratio/manifest.py
```

生成的清单里，工厂是 `"模块:类名"` 形式的字符串，所以列出插件时不会导入实现。改了插件的 ID、种类或依赖之后重新生成一次。

在 `pyproject.toml` 里把清单登记到 `tallyjudge.plugins` 入口组：

```toml
[project.entry-points."tallyjudge.plugins"]
length_ratio = "tallyjudge_length_ratio.manifest:PLUGINS"
```

安装这个包之后：

```bash
tallyjudge metrics list
tallyjudge run --input samples.jsonl --metric metric.length_ratio.response_to_reference --store runs
```

需要额外的第三方包时，声明成 `Plugin("metric", requires=("包名",))`，并在实现里用到的地方再导入它，不要在模块顶层导入。

## 插件 ID

- 格式是 `<种类>.<你的包或组织名>.<名称>`，例如 `metric.acme.tone`。
- ID 只在插件类上写一次：指标写在 `MetricSpec` 里，其他种类写成类属性 `plugin_id`。清单里的 ID 是生成出来的。
- 内置插件的 ID 是保留的。与已注册的 ID 冲突的插件会被跳过，并在日志里给出警告。
- 清单读不出来的包也只会产生一条警告，不影响其他插件。

## 测试

不需要 API 密钥。用自己的注册表实例把插件装进去，再用 `Evaluator` 跑：

```python
from tallyjudge import Dataset, Evaluator
from tallyjudge.registry import PluginRegistry
from tallyjudge_length_ratio.manifest import PLUGINS


def test_length_ratio():
    metric = PluginRegistry(PLUGINS).create(PLUGINS[0].id, cap=1.5)
    result = Evaluator([metric]).run(
        Dataset.from_records([{"response": "abcd", "reference": "ab"}])
    )
    assert result.rows[0].metrics[PLUGINS[0].id].value == 1.5
```

要测试调用 OpenAI 兼容接口的代码，用 `tallyjudge.sdk.testing.ScriptedClient` 代替真实客户端：先用 `chat_reply(...)` 排好回答，跑完后检查 `client.calls` 里记录的请求。

## 每种插件要遵守的行为

契约不只是方法签名。`tallyjudge.sdk.testing` 提供了对应的检查函数，把它们放进你的测试里：

| 种类 | 必须做到 | 检查函数 |
|---|---|---|
| 指标 | 返回值符合 `spec` 声明的类型和范围；不修改传入的参数；`config_snapshot` 是 JSON 对象且两次调用结果相同；缺少必填输入时抛 `InputValidationError` | `await check_metric(metric, inputs)` |
| 模型 | `plugin_id` 非空，`usage_tracking` 是 `per_request` 或 `untracked`；回答按给定的结构解析，并附带用量 | `await check_language_model(model)` |
| 存储 | 保存后读回的结果与原结果相同；重复保存同一次运行不报错；读不存在的运行抛 `FileNotFoundError` | `check_storage(backend, result)` |
| 缓存 | 没有的键返回 `None`；存进去的分数原样取回 | `check_cache(cache)` |
| 导入器 | 返回值与声明的 `produces` 一致：`documents` 是源文档，`trace` 是工作流轨迹，`records` 是能建成数据集的记录 | `check_importer(importer, source)` |

另外两条没有检查函数，但同样要遵守：

- 不支持的配置项要报错，不能悄悄忽略。把 `Config` 设成 `extra="forbid"` 就能做到。
- 事件接收器只观察，不能通过抛异常之外的方式影响评分；它抛出的异常会被记录，之后不再向它发送事件。

## 安全

插件是普通的 Python 代码，TallyJudge 不做任何隔离。只安装你信任的插件包。

在生产环境里可以用环境变量 `TALLYJUDGE_PLUGINS` 限定读取哪些插件：值是入口名的逗号分隔列表（即 `pyproject.toml` 里等号左边的名字），设成空字符串表示只用内置插件。不设置时读取全部已安装的插件。
