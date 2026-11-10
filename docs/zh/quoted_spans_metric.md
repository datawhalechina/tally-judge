## `QuotedSpansAlignment`

**用途**。衡量模型回答中的直接引文，
有多大比例能在检索资料中找到逐字一致的文本。得分范围为
[0, 1]，1.0 表示所有引文都有依据，0.0 表示
全部引文都未出现在资料中。

**意义**。用户通常更信任直接引用。模型引用
证据中没有的事实会削弱可靠性。此指标可以
发现引用偏移，即答案中的引述短语
缺少来源支持。

## 新版 Collections API（推荐） { #modern-collections-api-recommended }

```python
from ragas.metrics.collections import QuotedSpansAlignment

metric = QuotedSpansAlignment()

result = await metric.ascore(
    response='The study found that "machine learning improves accuracy".',
    retrieved_contexts=["Machine learning improves accuracy by 15%."]
)
print(f"Score: {result.value}")  # 1.0
print(f"Reason: {result.reason}")  # "Matched 1/1 quoted spans"
```

**参数**

- `name`，指标名称，默认为 `"quoted_spans_alignment"`
- `casefold`，匹配前是否统一小写，默认 True
- `min_span_words`，引文最少单词数，默认 3

**输入**

- `response: str`，包含引文的模型回答
- `retrieved_contexts: List[str]`，用于核对的来源段落列表

**输出**为包含以下内容的 `MetricResult`。

- `value`，[0, 1] 范围内的得分
- `reason`，匹配数量与引文总数的说明

**说明**

- 实现通过合并连续空白、统一小写规范化文本。
- 默认忽略少于三个单词的引文，可通过 `min_span_words` 调整。
- 回答不含引文时，得分为 1.0，表示没有需要核验的内容。

---

## 旧版 API（已弃用） { #legacy-api-deprecated }

> **警告**。旧版 `quoted_spans_alignment` 函数已弃用。
> 请使用 `ragas.metrics.collections` 中的 `QuotedSpansAlignment`。

**输入结构**

- `answers: List[str]`，模型回答列表，长度为 N
- `sources: List[List[str]]`，长度为 N 的列表，每项为该回答的来源段落列表

**输出**为一个字典，包含以下字段。

```python
{
  "citation_alignment_quoted_spans": float,  # score in [0,1]
  "matched": float,                          # number of spans found in sources
  "total": float                            # total number of spans considered
}
```

**说明**

- 所有回答中都没有引文时，旧版得分定义为 0.0，
  且 `total = 0`。

