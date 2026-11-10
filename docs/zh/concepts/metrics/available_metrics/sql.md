# SQL


## 基于执行结果的指标 { #execution-based-metrics }
这类指标在数据库中执行 SQL 查询，再将得到的 `response` 与预期结果比较。

### DataCompy 得分 { #datacompy-score }

`DataCompyScore` 使用 Python 库 DataCompy 比较两个 pandas DataFrame，并给出详细的差异报告。评估时，将 SQL 执行结果与预期数据 `reference` 比较。为进行比较，`response` 与 `reference` 都需要采用示例中的逗号分隔值格式。

可以通过 `mode` 参数选择按行或按列比较 DataFrame。

`mode` 为 `row` 时按行比较，为 `column` 时按列比较。

$$
\text{Precision } = {|\text{Number of matching rows in response and reference}| \over |\text{Total number of rows in response}|}
$$

$$
\text{Recall } = {|\text{Number of matching rows in response and reference}| \over |\text{Total number of rows in reference}|}
$$

默认按行（`row`）比较，采用精确率与召回率的调和平均值，即 F1 得分。

```python
from ragas.metrics.collections import DataCompyScore

data1 = """acct_id,dollar_amt,name,float_fld,date_fld
10000001234,123.45,George Maharis,14530.1555,2017-01-01
10000001235,0.45,Michael Bluth,1,2017-01-01
10000001236,1345,George Bluth,,2017-01-01
10000001237,123456,Bob Loblaw,345.12,2017-01-01
10000001238,1.05,Lucille Bluth,,2017-01-01
10000001238,1.05,Loose Seal Bluth,,2017-01-01
"""

data2 = """acct_id,dollar_amt,name,float_fld
10000001234,123.4,George Michael Bluth,14530.155
10000001235,0.45,Michael Bluth,
10000001236,1345,George Bluth,1
10000001237,123456,Robert Loblaw,345.12
10000001238,1.05,Loose Seal Bluth,111
"""

metric = DataCompyScore()
result = await metric.ascore(response=data1, reference=data2)
print(f"F1 Score: {result.value}")
print(f"Details: {result.reason}")
```

要按列比较，将 `mode` 参数设为 `column`。

```python
metric = DataCompyScore(mode="columns", metric="recall")
result = await metric.ascore(response=data1, reference=data2)
```

---

### DataCompyScore（旧版） { #datacompyscore-legacy }

!!! warning "已弃用"
    `ragas.metrics` 中的 `DataCompyScore` 已弃用，将在未来版本移除。请使用上文 `ragas.metrics.collections` 中的同名指标。

旧版 `DataCompyScore` 使用 `SingleTurnSample` 数据结构。

```python
from ragas.metrics import DataCompyScore
from ragas.dataset_schema import SingleTurnSample

data1 = """acct_id,dollar_amt,name,float_fld,date_fld
10000001234,123.45,George Maharis,14530.1555,2017-01-01
10000001235,0.45,Michael Bluth,1,2017-01-01
10000001236,1345,George Bluth,,2017-01-01
10000001237,123456,Bob Loblaw,345.12,2017-01-01
10000001238,1.05,Lucille Bluth,,2017-01-01
10000001238,1.05,Loose Seal Bluth,,2017-01-01
"""

data2 = """acct_id,dollar_amt,name,float_fld
10000001234,123.4,George Michael Bluth,14530.155
10000001235,0.45,Michael Bluth,
10000001236,1345,George Bluth,1
10000001237,123456,Robert Loblaw,345.12
10000001238,1.05,Loose Seal Bluth,111
"""
sample = SingleTurnSample(response=data1, reference=data2)
scorer = DataCompyScore()
await scorer.single_turn_ascore(sample)
```
要按列比较，将 `mode` 参数设为 `column`。


```python
scorer = DataCompyScore(mode="column", metric="recall")
```

## 无须执行查询的指标 { #non-execution-based-metrics }

在数据库中执行 SQL 可能耗时较长，有时也不可行。这种情况下，可以直接比较 SQL 查询，评估其质量。

### SQL 语义等价性 { #sql-semantic-equivalence }

`SQLSemanticEquivalence` 判断生成的 SQL 查询与参考查询在语义上是否等价。它使用 LLM，结合给定的数据库结构分析两个查询，判断它们是否会产生相同结果。

这是一个二元指标。
- **1.0**，两个 SQL 查询语义等价
- **0.0**，两个 SQL 查询不等价

指标会结合数据库结构判断等价性，并考虑不影响语义的语法差异，例如 `active = 1` 与 `active = true`。

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import SQLSemanticEquivalence

# Initialize the LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create the metric
metric = SQLSemanticEquivalence(llm=llm)

# Evaluate SQL equivalence
result = await metric.ascore(
    response="""
        SELECT p.product_name, SUM(oi.quantity) AS total_quantity
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_name;
    """,
    reference="""
        SELECT products.product_name, SUM(order_items.quantity) AS total_quantity
        FROM order_items
        INNER JOIN products ON order_items.product_id = products.product_id
        GROUP BY products.product_name;
    """,
    reference_contexts=[
        """
        Table order_items:
        - order_item_id: INT
        - order_id: INT
        - product_id: INT
        - quantity: INT
        """,
        """
        Table products:
        - product_id: INT
        - product_name: VARCHAR
        - price: DECIMAL
        """
    ]
)

print(f"Equivalent: {result.value == 1.0}")
print(f"Explanation: {result.reason}")
```

结果包含对两个查询的解释，以及判断是否等价的理由。

---

### LLMSQLEquivalence（旧版） { #llmsqlequivalence-legacy }

!!! warning "已弃用"
    `LLMSQLEquivalence` 已弃用，将在未来版本移除。请使用上文 `ragas.metrics.collections` 中的 `SQLSemanticEquivalence`。

`LLMSQLEquivalence` 是评估 SQL 语义等价性的旧版指标，使用 `SingleTurnSample` 数据结构，并需要单独设置 LLM。

```python
from ragas.metrics import LLMSQLEquivalence
from ragas.dataset_schema import SingleTurnSample

sample = SingleTurnSample(
    response="""
        SELECT p.product_name, SUM(oi.quantity) AS total_quantity
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_name;
    """,
    reference="""
        SELECT p.product_name, COUNT(oi.quantity) AS total_quantity
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_name;
    """,
    reference_contexts=[
        """
        Table order_items:
        - order_item_id: INT
        - order_id: INT
        - product_id: INT
        - quantity: INT
        """,
        """
        Table products:
        - product_id: INT
        - product_name: VARCHAR
        - price: DECIMAL
        """
    ]
)

scorer = LLMSQLEquivalence()
scorer.llm = openai_model
await scorer.single_turn_ascore(sample)
```
