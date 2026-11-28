---
title: 开发自定义指标
description: 用答案长度比实现一个可配置的指标，再将同样的接口用于模型裁判。
---

下面的指标计算回答与参考答案的字符数之比。它适合检查摘要是否明显过长或过短。长度接近只说明篇幅相似，无法判断答案是否正确。把下面的代码放入 `tallyjudge_length_ratio/metric.py`，可直接用于下一节的安装示例。

## 声明输入和评分

```python
from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.sdk import MetricBase, MetricResult, MetricSpec, Plugin


class LengthRatioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cap: float | None = Field(default=None, gt=0)


class LengthRatioMetric(MetricBase):
    plugin = Plugin("metric")
    Config = LengthRatioConfig
    spec = MetricSpec(
        "metric.length_ratio.response_to_reference",
        "1",
        {"response": str, "reference": str},
        value_range=(0.0, None),
    )

    def __init__(self, cap: float | None = None):
        self.cap = LengthRatioConfig(cap=cap).cap

    def config_snapshot(self):
        return {"cap": self.cap}

    async def ascore(self, response: str, reference: str) -> MetricResult:
        ratio = len(response) / max(len(reference), 1)
        return MetricResult(ratio if self.cap is None else min(ratio, self.cap))
```

`spec.inputs` 的键与 `ascore` 参数同名。框架从样本取出对应字段，严格校验后才进入公式。上例沿用示例包的空参考约定，分母最小为 1。如果你的指标在空分母时没有数学定义，在该分支抛出 `UndefinedScore("empty_reference")`，框架会把该项记为 `undefined`。需要给出判断依据时可返回 `MetricResult(value, reason, details={...})`，其中 `details` 使用可保存为 JSON 的值。

构造函数负责校验配置值，注册表根据 `Config` 中的字段名检查调用者传入的选项。`config_snapshot()` 返回影响分数的设置，因此两次运行采用不同上限时可以识别出来。修改公式、输入语义或空值规则时应提升 `revision`，使旧结果仍能辨认。

## 验证单条与批量调用

以下代码可在示例包目录单独保存为 Python 脚本运行。对两个字符的参考答案和四个字符的回答，原始长度比为 2，应用上限后为 1.5。

```python
from tallyjudge import Dataset, Evaluator
from tallyjudge.sdk import MetricBinding
from tallyjudge_length_ratio.metric import LengthRatioMetric

metric = LengthRatioMetric(cap=1.5)
assert metric.score(response="abcd", reference="ab").value == 1.5

data = Dataset.from_records([{"answer": "abcd", "gold": "ab"}])
binding = MetricBinding(
    metric, key="length_ratio", input_map={"response": "answer", "reference": "gold"}
)
result = Evaluator([binding]).run(data)
score = result.rows[0].metrics["length_ratio"]
assert (score.status, score.value) == ("success", 1.5)
```

`input_map` 左侧是指标参数，右侧是数据列名。进入事件循环后，单条评分用 `await metric.ascore_inputs({...})`，批量评分用 `await Evaluator([binding]).arun(data)`。

## 让裁判模型评分

如果规则依赖模型，可在 `Plugin` 声明中指定 `llm` 依赖，构造函数接收调用方配置好的模型。同一指标因此可以使用生产模型或离线模型。下面定义一个判断回答语气是否礼貌的二元指标。

```python
from pydantic import BaseModel

from tallyjudge.sdk import (
    LanguageModel, MetricBase, MetricResult, MetricSpec, Plugin, PromptInput, gated,
)


class Verdict(BaseModel):
    polite: bool


class PolitenessMetric(MetricBase):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})
    spec = MetricSpec(
        "metric.example.politeness", "1", {"response": str}, value_range=(0.0, 1.0)
    )

    def __init__(self, llm: LanguageModel):
        self.llm = gated(llm)

    def config_snapshot(self):
        return {"llm": {"plugin_id": self.llm.plugin_id, "config": self.llm.config_snapshot()}}

    async def ascore(self, response: str) -> MetricResult:
        prompt = PromptInput.from_text(
            "Decide if the reply is polite. Return JSON with a boolean field "
            f"named polite.\nReply: {response}"
        )
        reply = await self.llm.agenerate(prompt, Verdict)
        return MetricResult(float(reply.data.polite))
```

`Verdict` 声明模型回复的结构。`gated(llm)` 让直接构造指标时也能记录调用；注册表注入的模型已经过相同包装，重复包装会返回同一个对象。指标只需处理评分含义，模型调用的请求计数和用量由适配器记录。如何离线提供结构化回答见[模型适配器](/plugins/model-adapters)。如需让别人按 ID 使用此指标，继续阅读[注册与测试](/plugins/registration-and-tests)。
