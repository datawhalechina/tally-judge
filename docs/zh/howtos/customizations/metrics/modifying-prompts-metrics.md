# 修改指标提示词 { #modifying-prompts-in-metrics }

Ragas 中每个基于 LLM 的指标都使用一个或多个提示词，生成中间结果并据此计算得分。提示词可以视为这类指标的超参数。针对领域和场景优化提示词，有时能将准确率提升 10～20%。最佳提示词还取决于所用 LLM，因此值得按指标进行调优。

**快速开始**。如果只需要简单的自定义指标，可以使用直接接收自定义提示词的 [`DiscreteMetric`][ragas.metrics.discrete.DiscreteMetric] 或 [`NumericMetric`][ragas.metrics.numeric.NumericMetric]。示例参见[离散指标](../../../concepts/metrics/overview/index.md#1-discrete-metrics)。

本指南介绍如何修改**现有 collections 指标**（如 Faithfulness、FactualCorrectness）中基于 [`BasePrompt`][ragas.prompt.BasePrompt] 的提示词。继续前，请先了解[提示词对象](../../../concepts/components/prompt.md)。

## 了解指标使用的提示词 { #understand-the-prompts-of-your-metric }

对于支持自定义提示词的指标，可以通过实例访问底层提示词对象。下面以 `Faithfulness` 为例。

```python
from ragas.metrics.collections import Faithfulness
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# Setup dependencies
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric instance
scorer = Faithfulness(llm=llm)

# Faithfulness has two prompts:
# 1. statement_generator_prompt - breaks response into atomic statements
# 2. nli_statement_prompt - evaluates each statement against context
print(scorer.statement_generator_prompt)
print(scorer.nli_statement_prompt)
```

## 生成并查看提示词字符串 { #generating-and-viewing-the-prompt-string }

查看实际发送给 LLM 的提示词。

```python
from ragas.metrics.collections.faithfulness.util import StatementGeneratorInput

# Create sample input
sample_input = StatementGeneratorInput(
    question="What is the Eiffel Tower?",
    answer="The Eiffel Tower is located in Paris."
)

# Generate the prompt string
prompt_string = scorer.statement_generator_prompt.to_string(sample_input)
print(prompt_string)
```

## 修改提示词 { #modifying-prompts }

Ragas 的新版指标采用模块化的 BasePrompt 类，定制过程如下。

1. **访问提示词**，通过指标实例的属性获取
2. **修改提示词类**，扩展或继承提示词类，定制指令与示例
3. **更新指标**，将自定义提示词赋给指标的对应属性

### 示例，定制 FactualCorrectness 提示词 { #example-customizing-factualcorrectness-prompt }

FactualCorrectness 内部使用两个提示词。
- `prompt`，用于拆分陈述的 ClaimDecompositionPrompt
- `nli_prompt`，用于对照上下文验证陈述的 NLIStatementPrompt

可以定制其中一个，也可以同时定制两者。

```python
from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionPrompt,
    NLIStatementPrompt,
)

# Create a custom claim decomposition prompt by subclassing
class CustomClaimDecompositionPrompt(ClaimDecompositionPrompt):
    instruction = """You are an expert at breaking down complex statements into atomic claims.
Break down the input text into clear, verifiable claims.
Only output valid JSON with a "claims" array."""

# Optionally customize the NLI prompt too
class CustomNLIPrompt(NLIStatementPrompt):
    instruction = """Carefully evaluate if each statement is supported by the context.
Be strict in your verification - only mark as supported if directly stated."""

# Create metric instance and replace prompts
scorer = FactualCorrectness(llm=llm)
scorer.prompt = CustomClaimDecompositionPrompt()
scorer.nli_prompt = CustomNLIPrompt()

# Now the metric will use the custom prompts
result = await scorer.ascore(
    response="The Eiffel Tower is in Paris and was built in 1889.",
    reference="The Eiffel Tower is located in Paris. It was completed in 1889."
)
```

### 示例，定制 Faithfulness 示例 { #example-customizing-faithfulness-examples }

少样本示例会显著影响 LLM 输出，修改方法如下。

```python
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
    StatementFaithfulnessAnswer,
)

# Create custom prompt with domain-specific examples
class DomainSpecificNLIPrompt(NLIStatementPrompt):
    examples = [
        (
            NLIStatementInput(
                context="Machine learning is a field within artificial intelligence that enables systems to learn from data.",
                statements=[
                    "Machine learning is a subset of AI.",
                    "Machine learning uses statistical techniques.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Machine learning is a subset of AI.",
                        reason="The context states ML is 'a field within artificial intelligence', supporting this claim.",
                        verdict=1
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Machine learning uses statistical techniques.",
                        reason="The context doesn't mention statistical techniques.",
                        verdict=0
                    ),
                ]
            ),
        ),
    ]

# Update the metric with custom prompt
scorer = Faithfulness(llm=llm)
scorer.nli_statement_prompt = DomainSpecificNLIPrompt()

# Now evaluate with domain-specific prompts
result = await scorer.ascore(
    user_input="How do neural networks work?",
    response="Neural networks are inspired by biological neurons.",
    retrieved_contexts=["Artificial neural networks are computing systems loosely inspired by biological neural networks."]
)
```

## 将提示词适配到不同语言 { #adapting-prompts-to-different-languages }

通过 `adapt` 方法适配提示词的语言。

```python
from ragas.metrics.collections import Faithfulness

scorer = Faithfulness(llm=llm)

# Adapt the statement generator prompt to Spanish
adapted_prompt = await scorer.statement_generator_prompt.adapt(
    target_language="spanish",
    llm=llm,
    adapt_instruction=False  # Keep instruction in English, only translate examples
)

# Replace the prompt with the adapted version
scorer.statement_generator_prompt = adapted_prompt

# Now use the metric with Spanish examples
result = await scorer.ascore(
    user_input="¿Dónde nació Einstein?",
    response="Einstein nació en Alemania.",
    retrieved_contexts=["Albert Einstein nació en Alemania..."]
)
```

## 验证定制效果 { #verifying-your-customizations }

可以通过以下方式确认提示词修改是否生效。

```python
from ragas.metrics.collections.faithfulness.util import NLIStatementInput

# Create sample input to test the prompt
sample_input = NLIStatementInput(
    context="Paris is the capital and most populous city of France.",
    statements=["The capital of France is Paris.", "Paris is in Germany."]
)

# Generate and view the full prompt string
full_prompt = scorer.nli_statement_prompt.to_string(sample_input)
print("Full Prompt:")
print(full_prompt)
```

