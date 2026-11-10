# 将指标适配到目标语言 { #adapting-metrics-to-target-language }

评估非英语 LLM 应用时，可以将指标适配到目标语言。Ragas 使用 LLM 翻译提示词中的少样本示例。

## 配置 { #setup }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

metric = Faithfulness(llm=llm)
```

## 将提示词适配到目标语言 { #adapt-prompts-to-target-language }

Collections 指标将提示词作为直接属性暴露，使用 `adapt()` 方法翻译少样本示例。

```python
# Check original language
print(metric.statement_generator_prompt.language)
# english

# Adapt prompts to Hindi
metric.statement_generator_prompt = await metric.statement_generator_prompt.adapt(
    target_language="hindi", llm=llm
)
metric.nli_statement_prompt = await metric.nli_statement_prompt.adapt(
    target_language="hindi", llm=llm
)

# Verify adaptation
print(metric.statement_generator_prompt.language)
# hindi

# See translated example
print(metric.statement_generator_prompt.examples[0][0].question)
# अल्बर्ट आइंस्टीन कौन थे और वे किस चीज़ के लिए सबसे अधिक जाने जाते हैं?
```

!!! note
    默认只翻译少样本示例，指令仍保留英语。要同时翻译指令，将 `adapt_instruction=True`。

## 使用适配后的指标评估 { #evaluate-with-adapted-metric }

```python
result = await metric.ascore(
    user_input="भारत की राजधानी क्या है?",
    response="भारत की राजधानी नई दिल्ली है।",
    retrieved_contexts=["भारत की राजधानी नई दिल्ली है, जो देश का सबसे बड़ा शहर भी है।"],
)

print(f"Faithfulness: {result.value}")
# Faithfulness: 1.0
```

## 适配其他指标 { #adapting-other-metrics }

这一方式适用于所有带有提示词的 collections 指标。

```python
from ragas.metrics.collections import AnswerRelevancy
from ragas.embeddings.base import embedding_factory

embeddings = embedding_factory("openai", client=client)
relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

# Adapt the prompt
relevancy.prompt = await relevancy.prompt.adapt(
    target_language="spanish", llm=llm
)

# See translated example
print(relevancy.prompt.examples[0][0].response)
# Albert Einstein nació en Alemania.
```

## 适配 FactualCorrectness { #adapting-factualcorrectness }

FactualCorrectness 包含两个提示词，两者都需要适配。

```python
from ragas.metrics.collections import FactualCorrectness

metric = FactualCorrectness(llm=llm)

# Adapt both prompts to German
metric.prompt = await metric.prompt.adapt(
    target_language="german", llm=llm
)
metric.nli_prompt = await metric.nli_prompt.adapt(
    target_language="german", llm=llm
)

# Verify adaptation
print(metric.prompt.language)  # german
print(metric.nli_prompt.language)  # german

# Now use the adapted metric
result = await metric.ascore(
    response="Einstein wurde 1879 in Deutschland geboren.",
    reference="Albert Einstein wurde am 14. März 1879 in Ulm, Deutschland geboren."
)

print(f"Factual Correctness: {result.value}")
```

!!! tip
    与 Faithfulness 一样，FactualCorrectness 内部使用两个提示词。
    - `prompt`，用于拆分陈述的 ClaimDecompositionPrompt
    - `nli_prompt`，用于验证陈述的 NLIStatementPrompt

    评估非英语内容时，应同时适配两个提示词。
