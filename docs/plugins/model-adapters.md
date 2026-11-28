---
title: 开发模型适配器
description: 将供应商返回的文本、向量和用量接入公共模型协议，用离线响应验证解析、重试和顺序。
---

裁判指标向语言模型传入提示词和期望的 Pydantic 类型；相似度指标向嵌入模型传入一组文本。适配器负责把一次逻辑调用转换为供应商请求，并把返回值整理成 TallyJudge 的类型。若服务已兼容 OpenAI 的聊天或嵌入接口，可直接使用内置适配器。

| 调用方需要什么 | 实现的方法 | 返回值 |
| --- | --- | --- |
| 从提示词产生结构化回答 | `agenerate(prompt, response_model)` | `GenerationResult`，其中 `data` 是解析好的 `response_model` 实例 |
| 把一批文本嵌入到同一个向量空间 | `aembed_texts(texts)` | `EmbeddingResult`，其中 `vectors[i]` 对应 `texts[i]` |

两类模型都提供 `plugin_id`、`usage_tracking` 和 `config_snapshot()`。快照应能区分影响结果的模型与设置，只放入可公开保存的 JSON 值。客户端和凭据由调用方持有。

## 接入语言模型

继承 `LanguageModelBase` 时，只需实现单次文本请求和结构化重试次数。以下示例的客户端由构造函数传入，`client.complete(messages)` 返回回复文本和实际使用的模型名。替换为自己的 SDK 时，按服务协议转换消息，并在发请求的那一层记录这次尝试。

```python
from collections.abc import Sequence
from typing import Protocol

from tallyjudge.sdk import (
    LanguageModelBase, Message, ModelError, Plugin, RequestRecord, record_request,
)


class ChatClient(Protocol):
    async def complete(self, messages: Sequence[Message]) -> tuple[str, str]: ...


class ExampleChat(LanguageModelBase):
    plugin = Plugin("model", dependencies={"client": "client"})
    plugin_id = "model.example.chat"
    structured_retries = 1

    def __init__(self, *, client: ChatClient):
        self.client = client

    def config_snapshot(self):
        return {"model": "example-chat", "structured_retries": self.structured_retries}

    async def _acomplete(
        self, messages: Sequence[Message], records: list[RequestRecord]
    ) -> tuple[str, str]:
        with record_request("generation", records):
            try:
                return await self.client.complete(messages)
            except ConnectionError as exc:
                raise ModelError("Chat service connection failed") from exc
```

在同一文件中追加以下代码即可离线运行。第一次返回无效 JSON，第二次返回符合模式的回答。这次逻辑调用产生两次请求，服务未报告 token 数，这两次都计入 `requests_without_usage`。

```python
import asyncio

from pydantic import BaseModel

from tallyjudge.sdk import PromptInput


class Answer(BaseModel):
    answer: str


class LocalClient:
    def __init__(self):
        self.replies = iter(["invalid", '{"answer":"Paris"}'])

    async def complete(self, messages):
        return next(self.replies), "example-chat"


async def main():
    model = ExampleChat(client=LocalClient())
    result = await model.agenerate(PromptInput.from_text("Capital of France?"), Answer)
    assert result.data.answer == "Paris"
    assert result.usage.requests == 2
    assert result.usage.requests_without_usage == 2


asyncio.run(main())
```

- 服务能提供实际用量时，用 `with record_request("generation", records) as record` 获取该次记录，收到答复后设置 `record.usage = Usage(input_tokens=..., output_tokens=...)`。只有一侧有数据时，另一侧可保持 `None`。模型内部完全无法观察每次请求时，应声明 `usage_tracking="untracked"`；这类模型无法使用按请求数限制的预算。
- 基类只对 JSON 或结构校验失败追加纠正提示并重试。传输错误用 `ModelError(...) from exc` 保留原因。服务返回拒答或截断时，可用 `ModelOutputError("refusal")` 或 `ModelOutputError("truncated")` 标明原因。取消应继续向上传递。
- 适配器需要一个真实的第三方 SDK 时，在构造客户端或第一次使用时导入。调用方注入的客户端由调用方关闭。内置 OpenAI 兼容客户端在同一个事件循环中复用，连续多轮可在同一个 `asyncio.run(main())` 中等待各次 `arun`。

## 验证嵌入模型

新适配器声明 `Plugin("embedding", dependencies={"client": "client"})` 并实现 `aembed_texts`。如果接口分批发送，建立一个 `RequestCall()`，每批在 `record_request("embedding", records)` 中发送。最后返回 `EmbeddingResult(vectors, UsageSummary.from_records(records), model, records.call_id)`。同一批文本的向量维度应一致；供应商返回索引时按索引还原顺序。空输入返回空向量列表，跳过传输。

以下独立脚本通过内置嵌入适配器演示怎样验证这两项行为。替换为自己的适配器与测试客户端后，保留相同的断言。

```python
import asyncio

from tallyjudge import create_embedding_model
from tallyjudge.sdk.testing import ScriptedClient


async def main():
    client = ScriptedClient()
    client.embedding_reply([(1, [0.0, 1.0]), (0, [1.0, 0.0])])
    model = create_embedding_model(
        "embedding.openai_compatible.embeddings", model="example", client=client
    )
    empty = await model.aembed_texts([])
    assert empty.vectors == [] and empty.usage.requests == 0
    result = await model.aembed_texts(["first", "second"])
    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert len(client.calls) == result.usage.requests == 1


asyncio.run(main())
```

按 ID 提供模型的入口配置见[注册与测试](/plugins/registration-and-tests)。让一个指标接收模型见[开发自定义指标](/plugins/metrics)。
