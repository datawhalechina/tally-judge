## 事实正确性 { #factual-correctness }

`FactualCorrectness` 比较生成的 `response` 与 `reference`，衡量回答在事实层面与参考答案的一致程度。得分为 0～1，越高越好。该指标先使用 LLM 将回答和参考答案拆分为陈述，再通过自然语言推断判断事实重合情况，并使用精确率、召回率或 F1 得分进行量化。可以通过 `mode` 参数选择计算方式。

### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = FactualCorrectness(llm=llm)

# Evaluate
result = await scorer.ascore(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. It has a height of 1000ft."
)
print(f"Factual Correctness Score: {result.value}")
```

输出如下。

```
Factual Correctness Score: 0.67
```

`mode` 默认为 `f1`，可以设为 `precision` 或 `recall`。

```python
# Precision mode - measures what fraction of response claims are supported by reference
scorer = FactualCorrectness(llm=llm, mode="precision")
result = await scorer.ascore(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. It has a height of 1000ft."
)
print(f"Precision Score: {result.value}")
```

输出如下。

```
Precision Score: 1.0
```

还可以使用 `atomicity` 和 `coverage` 参数控制陈述拆分的粒度。

```python
# High granularity - more detailed claim decomposition
scorer = FactualCorrectness(
    llm=llm,
    mode="f1",
    atomicity="high",  # More atomic claims
    coverage="high"    # Comprehensive coverage
)
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris. It has a height of 1000ft."
    )
    ```

### 计算过程 { #how-its-calculated }

真正例（TP）、假正例（FP）和假负例（FN）的计算公式如下。

$$
\text{True Positive (TP)} = \text{Number of claims in response that are present in reference}
$$

$$
\text{False Positive (FP)} = \text{Number of claims in response that are not present in reference}
$$

$$
\text{False Negative (FN)} = \text{Number of claims in reference that are not present in response}
$$

精确率、召回率和 F1 得分的计算公式如下。

$$
\text{Precision} = {TP \over (TP + FP)}
$$

$$
\text{Recall} = {TP \over (TP + FN)}
$$

$$
\text{F1 Score} = {2 \times \text{Precision} \times \text{Recall} \over (\text{Precision} + \text{Recall})}
$$

### 控制陈述数量 { #controlling-the-number-of-claims }

回答和参考答案中的每个句子都可以拆分为一条或多条陈述。单个句子拆分出的陈述数量取决于应用需要的原子性（`atomicity`）和覆盖度（`coverage`）。


#### 示例 { #example_1 }

```python
scorer = FactualCorrectness(mode="precision",atomicity="low")
```
输出如下。
```
1.0
```


#### 理解原子性与覆盖度 { #understanding-atomicity-and-coverage }

陈述拆分受两个重要参数影响。

1. **原子性**
2. **覆盖度**

这两个参数控制生成陈述的粒度与完整性。

#### 原子性 { #atomicity }

**原子性**表示将句子拆分为最小、有独立含义的组成部分的程度。可以根据需求，选择细粒度陈述或更完整的综合陈述。

- **高原子性**。将句子拆分为基本的、不可再分的陈述，每条较短的陈述表达一项独立信息。

  **例如，**
  - 原句如下。
    - “阿尔伯特·爱因斯坦是一位德国理论物理学家，他提出了相对论，并为量子力学作出了贡献。”
  - 拆分后的陈述如下。
    - “阿尔伯特·爱因斯坦是一位德国理论物理学家。”
    - “阿尔伯特·爱因斯坦提出了相对论。”
    - “阿尔伯特·爱因斯坦为量子力学作出了贡献。”

- **低原子性**。尽可能保持句子完整，生成较少的陈述，每条可能包含多项信息。

  **例如，**
  - 原句如下。
    - “阿尔伯特·爱因斯坦是一位德国理论物理学家，他提出了相对论，并为量子力学作出了贡献。”
  - 拆分后的陈述如下。
    - “阿尔伯特·爱因斯坦是一位德国理论物理学家，他提出了相对论，并为量子力学作出了贡献。”

#### 覆盖度 { #coverage }

**覆盖度**表示拆分后的陈述对原句信息的完整覆盖程度。你可以选择保留全部细节，或只概括主要内容。

- **高覆盖度**。拆分后的陈述涵盖原句的全部信息，保留每个细节。

  **例如，**
  - 原句如下。
    - “玛丽·居里是一位波兰出生、后入籍法国的物理学家和化学家，她在放射性研究领域开展了开创性工作。”
  - 拆分后的陈述如下。
    - “玛丽·居里是一位波兰物理学家。”
    - “玛丽·居里是一位入籍法国的物理学家。”
    - “玛丽·居里是一位化学家。”
    - “玛丽·居里在放射性研究领域开展了开创性工作。”

- **低覆盖度**。仅保留主要信息，省略部分细节，给出较概括的内容。

  **例如，**
  - 原句如下。
    - “玛丽·居里是一位波兰出生、后入籍法国的物理学家和化学家，她在放射性研究领域开展了开创性工作。”
  - 拆分后的陈述如下。
    - “玛丽·居里是一位物理学家。”
    - “玛丽·居里研究过放射性。”

#### 组合原子性与覆盖度 { #combining-atomicity-and-coverage }

同时调整原子性和覆盖度，可以让陈述的细节与完整程度符合具体场景的需要。

- **高原子性与高覆盖度**。生成详细、全面的陈述，涵盖原句的各方面信息。

  **例如，**
  - 原句如下。
    - “查尔斯·巴贝奇是一位英国数学家、哲学家、发明家和机械工程师。”
  - 拆分后的陈述如下。
    - “查尔斯·巴贝奇是一位英国数学家。”
    - “查尔斯·巴贝奇是一位哲学家。”
    - “查尔斯·巴贝奇是一位发明家。”
    - “查尔斯·巴贝奇是一位机械工程师。”

- **低原子性与低覆盖度**。生成较少、细节较少的陈述，概括主要信息。

  **例如，**
  - 原句如下。
    - “查尔斯·巴贝奇是一位英国数学家、哲学家、发明家和机械工程师。”
  - 拆分后的陈述如下。
    - “查尔斯·巴贝奇是一位英国数学家。”
    - “查尔斯·巴贝奇是一位发明家。”

#### 实际应用 { #practical-application }

- 需要深入分析或信息提取时，使用**高原子性和高覆盖度**，获得详细、全面的拆分结果。
- 只需要关键信息时，例如生成摘要，可以使用**低原子性和低覆盖度**。

通过灵活控制陈述数量，可以让信息粒度适应应用需求。

## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness


sample = SingleTurnSample(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. I has a height of 1000ft."
)

scorer = FactualCorrectness(llm = evaluator_llm)
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.67
```

### 修改计算模式 { #changing-the-mode }

默认使用 `F1` 模式，可以通过 `mode` 参数改为 `precision` 或 `recall`。

```python
scorer = FactualCorrectness(llm = evaluator_llm, mode="precision")
```

输出如下。

```
1.0
```

### 控制原子性 { #controlling-atomicity }

```python
scorer = FactualCorrectness(mode="precision", atomicity="low")
```

输出如下。

```
1.0
```
