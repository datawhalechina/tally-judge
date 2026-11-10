# 提示词对象 { #prompt-object }


Ragas 在各类指标和合成数据生成任务中使用提示词，并允许你修改默认提示词，或用自定义提示词替换。本指南介绍 Ragas 中的提示词对象。


## 提示词对象的组成 { #components-of-a-prompt-object }

Ragas 提示词对象包含以下主要部分。

1. **指令**。用自然语言清晰描述大语言模型（LLM）应当执行的任务，是提示词的基本要素。通过提示词对象的 `instruction` 变量指定。

2. **少样本示例**。提供少量示例通常能帮助 LLM 理解任务上下文，生成更准确的回答。通过 `examples` 变量指定，每个示例由输入和相应输出组成。

3. **输入模型**。提示词需要接收输入才能产生输出。`input_model` 使用 Pydantic 模型定义预期的输入结构，便于验证和解析传入提示词的数据。

4. **输出模型**。`output_model` 使用 Pydantic 模型定义 LLM 输出的数据结构，便于验证和解析生成结果。


## 示例 { #example }

以下示例定义了用于文本生成任务的提示词对象。

```python
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field

class MyInput(BaseModel):
    question: str = Field(description="The question to answer")

class MyOutput(BaseModel):
    answer: str = Field(description="The answer to the question")

class MyPrompt(PydanticPrompt[MyInput,MyInput]):
    instruction = "Answer the given question"
    input_model = MyInput
    output_model = MyOutput
    examples = [
        (
            MyInput(question="Who's building the opensource standard for LLM app evals?"),
            MyOutput(answer="Ragas")
        )
    ]

```

## 编写有效提示词的建议 { #guidelines-for-creating-effective-prompts }

在 Ragas 中创建提示词时，可以遵循以下建议，使提示词符合任务要求。

1. **指令清晰简洁**。明确说明 LLM 应执行的任务。含糊的指令可能导致不准确的回答。
2. **提供相关的少样本示例**。示例应覆盖与任务相关的多种场景，通常以 3～5 个为宜，帮助 LLM 理解上下文并准确作答。
3. **简化输入和输出模型**。使用简单直观的模型，准确描述 LLM 接收和生成的数据格式。如果模型结构复杂，可以将任务拆分为较小的子任务，分别设置提示词。
