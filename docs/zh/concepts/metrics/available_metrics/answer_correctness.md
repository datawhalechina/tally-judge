## 答案正确性 { #answer-correctness }

答案正确性衡量生成答案与标准答案的一致程度。它基于 `ground truth` 和 `answer` 计算，得分为 0～1。分数越高，表示生成答案越接近标准答案，正确性越好。

答案正确性综合考虑生成答案与标准答案的语义相似度和事实相似度，通过加权得到最终分数。必要时，也可以使用阈值将结果转换为二元值。

!!! note "嵌入模型要求"
    AnswerCorrectness 需要嵌入模型来计算语义相似度。使用 `evaluate()` 时，如果未显式提供嵌入模型，Ragas 会自动匹配 LLM 服务商对应的嵌入服务。例如，使用 Gemini 作为 LLM 时，会自动使用 Google 嵌入，无须 OpenAI API 密钥。也可以显式传入嵌入模型，以完全控制配置。


!!! example
    **标准答案**：爱因斯坦于 1879 年出生在德国。

    **高正确性答案**：1879 年，爱因斯坦出生在德国。

    **低正确性答案**：爱因斯坦于 1879 年出生在西班牙。


### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerCorrectness

# Setup LLM and embeddings
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

# Create metric
scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    reference="The first superbowl was held on January 15, 1967"
)
print(f"Answer Correctness Score: {result.value}")
```

输出如下。

```
Answer Correctness Score: 0.95
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        reference="The first superbowl was held on January 15, 1967"
    )
    ```

### 计算方法 { #calculation }

下面计算低正确性答案的得分。最终结果由事实正确性与答案相对标准答案的语义相似度加权得到。

事实正确性量化生成答案与标准答案之间的事实重合程度，涉及以下概念。
- TP（真正例），标准答案与生成答案中共有的事实或陈述。
- FP（假正例），生成答案中出现、但标准答案中没有的事实或陈述。
- FN（假负例），标准答案中出现、但生成答案中没有的事实或陈述。

在第二个示例中，
- TP 为 `[爱因斯坦出生于 1879 年]`
- FP 为 `[爱因斯坦出生在西班牙]`
- FN 为 `[爱因斯坦出生在德国]`

根据这三组陈述的数量，使用 F1 公式量化正确性。


$$
\text{F1 Score} = {|\text{TP} \over {(|\text{TP}| + 0.5 \times (|\text{FP}| + |\text{FN}|))}}
$$

随后计算生成答案与标准答案的语义相似度，详情参见[语义相似度](semantic_similarity.md)。


得到语义相似度后，将其与上述事实相似度加权平均，得到最终分数。可以通过 `weights` 参数调整权重。

## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用数据集的示例 { #example-with-dataset }

```python
from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[answer_correctness])
score.to_pandas()
```
