# 提示词 API 参考 { #prompt-api-reference }

Ragas 的提示词系统以灵活、类型安全的方式，为基于 LLM 的指标和其他组件定义提示词。本页介绍核心提示词类及其用法。

## 概览 { #overview }

Ragas 采用基于 `BasePrompt` 的模块化提示词架构，包含以下组成部分。

- **输入/输出模型**，通过 Pydantic BaseModel 类定义提示词输入与输出的结构
- **提示词类**，继承 `BasePrompt`，定义指令、示例和提示词生成逻辑
- **字符串提示词**，用于向后兼容的简单文本提示词

## 核心类 { #core-classes }

::: ragas.prompt
    options:
        members:
            - BasePrompt
            - StringPrompt
            - InputModel
            - OutputModel
            - PydanticPrompt
            - BoolIO
            - StringIO
            - PromptMixin

## Collections 指标提示词 { #metrics-collections-prompts }

Ragas 的新版指标使用专门的提示词类。每个指标模块包含以下部分。

- **输入模型**，定义提示词所需的数据，例如 `FaithfulnessInput`
- **输出模型**，定义预期的 LLM 返回结构，例如 `FaithfulnessOutput`
- **提示词类**，继承 `BasePrompt`，将示例和指令组织为提示词字符串

### 示例，忠实度指标提示词 { #example-faithfulness-metric-prompts }

```python
from ragas.metrics.collections.faithfulness.util import (
    FaithfulnessPrompt,
    FaithfulnessInput,
    FaithfulnessOutput,
)

# The prompt class combines input/output models with instructions and examples
prompt = FaithfulnessPrompt()

# Create input data
input_data = FaithfulnessInput(
    response="The capital of France is Paris.",
    context="Paris is the capital and most populous city of France."
)

# Generate the prompt string for the LLM
prompt_string = prompt.to_string(input_data)

# The output will be structured according to FaithfulnessOutput model
```

### 各指标的提示词 { #available-metric-prompts }

有关提示词的详情，请查阅各指标文档。

- [忠实度](../concepts/metrics/available_metrics/faithfulness.md)
- [上下文召回率](../concepts/metrics/available_metrics/context_recall.md)
- [上下文精确率](../concepts/metrics/available_metrics/context_precision.md)
- [答案正确性](../concepts/metrics/available_metrics/answer_correctness.md)
- [事实正确性](../concepts/metrics/available_metrics/factual_correctness.md)
- [噪声敏感度](../concepts/metrics/available_metrics/noise_sensitivity.md)

## 自定义 { #customization }

定制指标提示词的详细方法，请参阅[修改指标提示词](../howtos/customizations/metrics/modifying-prompts-metrics.md)。
