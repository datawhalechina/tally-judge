# 传统自然语言处理指标 { #traditional-nlp-metrics }

## 非 LLM 字符串相似度 { #non-llm-string-similarity }

`NonLLMStringSimilarity` 使用莱文斯坦距离、汉明距离和 Jaro 距离等传统字符串距离度量，衡量 `response` 与 `reference` 的相似程度，无须依赖大语言模型（LLM）。分数为 0～1，1 表示回答与参考文本完全匹配。

### 示例 { #example }

```python
from ragas.metrics.collections import NonLLMStringSimilarity, DistanceMeasure

# Create metric (no LLM/embeddings needed)
scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    response="The Eiffel Tower is located in India."
)
print(f"NonLLM String Similarity Score: {result.value}")
```

输出如下。

```
NonLLM String Similarity Score: 0.8918918918918919
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India."
    )
    ```

### 配置 { #configuration }

可以从 `DistanceMeasure` 中选择距离度量。以下示例使用汉明距离。

```python
scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.HAMMING)
```

可用的距离度量包括以下几种。
- `DistanceMeasure.LEVENSHTEIN`（默认）
- `DistanceMeasure.HAMMING`
- `DistanceMeasure.JARO`
- `DistanceMeasure.JARO_WINKLER`

### 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import NonLLMStringSimilarity

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="The Eiffel Tower is located in Paris."
)

scorer = NonLLMStringSimilarity()
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.8918918918918919
```

#### 使用其他距离度量的示例 { #example-with-different-distance-measure }

```python
from ragas.metrics._string import NonLLMStringSimilarity, DistanceMeasure

scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.HAMMING)
```


## BLEU 得分 { #bleu-score }

`BleuScore` 通过比较 `response` 与 `reference`，评估回答质量。它根据 n-gram 精确率和短句惩罚计算相似度。BLEU 最初用于机器翻译评估，也适用于其他自然语言处理任务。此指标无须 LLM，得分为 0～1，1 表示完全匹配。

### 示例 { #example_1 }

```python
from ragas.metrics.collections import BleuScore

# Create metric
scorer = BleuScore()

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    response="The Eiffel Tower is located in India."
)
print(f"BLEU Score: {result.value}")
```

输出如下。

```
BLEU Score: 0.7071067811865478
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India."
    )
    ```

### 配置 { #configuration_1 }

可以通过 `kwargs` 向底层 `sacrebleu.corpus_bleu` 函数传入附加参数。

```python
scorer = BleuScore(kwargs={"smooth_method": "exp"})
```

### 旧版指标 API { #legacy-metrics-api_1 }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample_1 }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="The Eiffel Tower is located in Paris."
)

scorer = BleuScore()
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.7071067811865478
```


## ROUGE 得分 { #rouge-score }

`RougeScore` 是一组评估自然语言生成质量的指标，使用 n-gram 的召回率、精确率和 F1 得分衡量 `response` 与 `reference` 的重合程度。此指标无须 LLM，得分为 0～1，1 表示完全匹配。

### 示例 { #example_2 }

```python
from ragas.metrics.collections import RougeScore

# Create metric (no LLM/embeddings needed)
scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    response="The Eiffel Tower is located in India."
)
print(f"ROUGE Score: {result.value}")
```

输出如下。

```
ROUGE Score: 0.8571428571428571
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India."
    )
    ```

### 配置 { #configuration_2 }

将 `rouge_type` 设为 `rouge1` 或 `rougeL`，可分别按一元词组或最长公共子序列计算得分。

```python
scorer = RougeScore(rouge_type="rouge1")
```

将 `mode` 设为 `precision`、`recall` 或 `fmeasure`，可分别计算精确率、召回率或 F1 得分。

```python
scorer = RougeScore(mode="recall")
```

### 旧版指标 API { #legacy-metrics-api_2 }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample_2 }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import RougeScore

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="The Eiffel Tower is located in Paris."
)

scorer = RougeScore()
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.8571428571428571
```

## 精确匹配 { #exact-match }

`ExactMatch` 检查回答与参考文本是否完全相同，适合要求输出逐字匹配的场景，例如工具调用参数。完全匹配时返回 1，否则返回 0。

### 示例 { #example_3 }

```python
from ragas.metrics.collections import ExactMatch

# Create metric (no LLM/embeddings needed)
scorer = ExactMatch()

# Evaluate
result = await scorer.ascore(
    reference="Paris",
    response="India"
)
print(f"Exact Match Score: {result.value}")
```

输出如下。

```
Exact Match Score: 0.0
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference="Paris",
        response="India"
    )
    ```

### 旧版指标 API { #legacy-metrics-api_3 }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample_3 }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ExactMatch

sample = SingleTurnSample(
    response="India",
    reference="Paris"
)

scorer = ExactMatch()
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.0
```

## 字符串包含性 { #string-presence }

`StringPresence` 检查回答是否包含参考文本，适合要求生成回答包含特定关键词或短语的场景。包含时返回 1，否则返回 0。

### 示例 { #example_4 }

```python
from ragas.metrics.collections import StringPresence

# Create metric (no LLM/embeddings needed)
scorer = StringPresence()

# Evaluate
result = await scorer.ascore(
    reference="Eiffel Tower",
    response="The Eiffel Tower is located in India."
)
print(f"String Presence Score: {result.value}")
```

输出如下。

```
String Presence Score: 1.0
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference="Eiffel Tower",
        response="The Eiffel Tower is located in India."
    )
    ```

### 旧版指标 API { #legacy-metrics-api_4 }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample_4 }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import StringPresence

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="Eiffel Tower"
)
scorer = StringPresence()
await scorer.single_turn_ascore(sample)
```

输出如下。

```
1.0
```

## CHRF 得分 { #chrf-score }

`CHRFScore` 使用**字符 n-gram F 分数**衡量 `response` 与 `reference` 的相似程度。BLEU 侧重精确率，CHRF 同时考虑**精确率与召回率**，更适合以下情况。

- 词形变化丰富的语言
- 存在改写或措辞较灵活的回答

CHRF 得分为 0～1，1 表示生成回答与参考文本完全匹配。此指标无须 LLM，完全通过确定性的比较计算。

### 示例 { #example_5 }

```python
from ragas.metrics.collections import CHRFScore

# Create metric (no LLM/embeddings needed)
scorer = CHRFScore()

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    response="The Eiffel Tower is located in India."
)
print(f"CHRF Score: {result.value}")
```

输出如下。

```
CHRF Score: 0.8048
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India."
    )
    ```

### 配置 { #configuration_3 }

可以通过 `kwargs` 向底层 `sacrebleu.corpus_chrf` 函数传入附加参数。

```python
# Customize character and word order
scorer = CHRFScore(kwargs={"char_order": 4, "word_order": 2})

# Customize beta (recall weight)
scorer = CHRFScore(kwargs={"beta": 3})
```

### 旧版指标 API { #legacy-metrics-api_5 }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

#### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample_5 }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ChrfScore

sample = SingleTurnSample(
    response="The Eiffel Tower is located in India.",
    reference="The Eiffel Tower is located in Paris."
)

scorer = ChrfScore()
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.8048
```
