## 多模态忠实度（MultiModalFaithfulness） { #multimodalfaithfulness }

`MultiModalFaithfulness` 衡量生成答案与图像及文本上下文在事实层面的一致性，使用答案、检索文本上下文和图像上下文计算。分数在 0～1 范围内，越高表示越忠实。

如果答案中的全部陈述都能从图像或文本上下文中推断出来，就认为答案是忠实的。指标直接对照上下文评估回答，忠实度得分为 0 或 1。

### 示例（推荐使用 Collections API） { #example-recommended-collections-api }

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import MultiModalFaithfulness

# Setup - use a vision-capable model
client = AsyncOpenAI()
llm = llm_factory("gpt-4o", client=client)  # Vision-capable model required

# Create metric instance
metric = MultiModalFaithfulness(llm=llm)

# Evaluate faithfulness
result = await metric.ascore(
    response="The Tesla Model X is an electric SUV.",
    retrieved_contexts=[
        "path/to/tesla_image.jpg",  # Image context
        "Tesla manufactures electric vehicles."  # Text context
    ]
)
print(f"Faithfulness Score: {result.value}")  # 1.0 (faithful) or 0.0 (not faithful)
```

### 示例（旧版 API，已弃用） { #example-legacy-api-deprecated }

!!! warning "已弃用"
    旧版 API 已弃用，将在未来版本移除。请迁移到上文的 Collections API。

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import MultiModalFaithfulness

sample = SingleTurnSample(
        user_input="What about the Tesla Model X?",
        response="Cats are cute.",
        retrieved_contexts=[
            "custom_eval/multimodal/images/tesla.jpg"
        ]
    )
scorer = MultiModalFaithfulness()
await scorer.single_turn_ascore(sample)
```

### 计算过程 { #how-its-calculated }

!!! example
    **问题**：介绍一下特斯拉 Model X。

    **上下文（图像）**
    - 一张特斯拉 Model X 的图片（custom_eval/multimodal/images/tesla.jpg）

    **高忠实度回答**：特斯拉 Model X 是特斯拉生产的一款电动 SUV。

    **低忠实度回答**：猫很可爱。

以下以低忠实度回答为例，说明计算过程。

- **第一步**，对照给定上下文评估回答。
    - 回答：“猫很可爱。”

- **第二步**，判断回答能否从给定上下文中推断出来。
    - 判定为否。

- **第三步**，根据判定确定忠实度得分。

    $$
    \text{Faithfulness} = 0
    $$

本例中，无法从特斯拉 Model X 的图片推断出“猫很可爱”，因此忠实度得分为 0。

### 支持的上下文类型 { #supported-context-types }

此指标支持以下上下文输入。

- **文本上下文**，普通文本字符串
- **图片 URL**，指向图片的 HTTP/HTTPS 地址
- **本地图片路径**，本地 jpg、png、gif、webp 或 bmp 文件路径
- **Base64 数据 URI**，以内联方式提供的 Base64 编码图片

### 使用要求 { #requirements }

- 需要支持视觉输入的 LLM，例如 `gpt-4o`、`gpt-4-vision-preview`、`claude-3-opus` 或 `gemini-pro-vision`
- 使用 Collections API 时，通过 `llm_factory` 创建 LLM 实例
