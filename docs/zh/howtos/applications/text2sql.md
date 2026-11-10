# 评估文本转 SQL 智能体 { #how-to-evaluate-a-text-to-sql-agent }

本指南介绍如何使用 Ragas 系统化评估并改进文本转 SQL 系统。

你将完成以下工作。

- 配置用于评估的文本转 SQL 基线系统
- 创建评估指标
- 为 SQL 智能体构建可复用的评估流水线
- 根据错误分析实施改进

## 配置环境 { #setup-your-environment }

我们提供可安装运行的简单模块，便于你专注理解评估过程。

```bash
uv pip install "ragas-examples[text2sql]"
```

## 快速测试智能体 { #quick-agent-test }

测试智能体如何将自然语言转换为 SQL。

```python
import os
import asyncio
from openai import AsyncOpenAI
from ragas_examples.text2sql.text2sql_agent import Text2SQLAgent

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create agent
openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
agent = Text2SQLAgent(client=openai_client, model_name="gpt-5-mini")

# Test with a sample query
test_query = "How much open credit does customer Andrew Bennett?"
result = asyncio.run(agent.query(test_query))

print(f"Natural Query: {result['query']}")
print(f"Generated SQL: {result['sql']}")
```

??? note "输出"
    ```python
    Natural Query: How much open credit does customer Andrew Bennett?
    Generated SQL: select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Andrew Bennett" )
    ```

这样就能根据自然语言生成 SQL。下面建立系统化评估流程。

### 下载 BookSQL { #download-booksql }

运行智能体或数据库工具前，先从 Hugging Face 下载需要申请访问的 BookSQL 数据集。

```bash
huggingface-cli login
uv run python -m ragas_examples.text2sql.data_utils --download-data
```

如果出现身份验证错误，请先访问 [BookSQL 数据集页面](https://huggingface.co/datasets/Exploration-Lab/BookSQL)，接受使用条款。

!!! note "完整代码"
    智能体与评估流水线的完整代码见[此目录](https://github.com/vibrantlabsai/ragas/tree/main/examples/ragas_examples/text2sql)。

## 准备数据集 { #prepare-your-dataset }

我们从 BookSQL 中准备了 99 条均衡样本，简单、中等和困难查询各 33 条。可以直接开始评估，也可以按下一节创建自己的数据集。

**下载并检查示例数据集。**

```bash
# Download the sample CSV from GitHub
curl -o booksql_sample.csv https://raw.githubusercontent.com/vibrantlabsai/ragas/main/examples/ragas_examples/text2sql/datasets/booksql_sample.csv
# View the first few rows to understand the structure
head -5 booksql_sample.csv
```

| 查询 | SQL | 难度 | 数据划分 |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-------|
| Richard Aguirre 的未结余额是多少？ | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Richard Aguirre" )                                                                                               | medium | train |
| Sarah Oconnor 的未结余额是多少？ | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Sarah Oconnor" )                                                                                                 | medium | train |
| Jeffrey Moore 的平均发票金额是多少？ | select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Jeffrey Moore" and transaction_type = 'invoice')                                                                              | hard   | train |
| 客户 Andrew Bennett 的未结信用余额是多少？ | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Andrew Bennett" )                                                                                                | easy   | train |

??? info "📋 可选，示例数据集的准备过程"

    下载并检查数据集。

    本指南使用 [BookSQL 数据集](https://huggingface.co/datasets/Exploration-Lab/BookSQL)。已有数据集时可以跳过。

    **下载数据集。**

    ```bash
    export HF_TOKEN=your-huggingface-token
    uv run python -m ragas_examples.text2sql.data_utils --download-data
    ```

    **注意**。BookSQL 需要申请访问。如果出现身份验证错误，请访问[数据集页面](https://huggingface.co/datasets/Exploration-Lab/BookSQL)，接受条款，并运行 `huggingface-cli login`。

    **检查数据集结构。**

    ```bash
    # Check the database schema
    sqlite3 BookSQL-files/BookSQL/accounting.sqlite ".schema" | head -20
    ```

    **预期结构如下。**

    ```sql
    CREATE TABLE master_txn_table(
                        id INTEGER ,
                        businessID INTEGER NOT NULL ,
                        Transaction_ID INTEGER NOT NULL,
                        Transaction_DATE DATE NOT NULL,
                        Transaction_TYPE TEXT NOT NULL,
                        Amount DOUBLE NOT NULL,
                        CreatedDATE DATE NOT NULL,
                        CreatedUSER TEXT NOT NULL,
                        Account TEXT NOT NULL,
                        AR_paid TEXT,
                        AP_paid TEXT,
                        Due_DATE DATE,
                        Open_balance DOUBLE,
                        Customers TEXT,
                        Vendor TEXT,
                        Product_Service TEXT,
                        Quantity INTEGER,
                        Rate DOUBLE,
                        Credit DOUBLE,
    ```

    数据集包含以下内容。

    - **数据库**，包含发票、客户等会计数据的 SQLite 文件
    - **问题**，英语自然语言查询
    - **SQL**，对应的 SQL 查询
    - **难度**，分为简单、中等和困难

    创建均衡的评估子集。

    ```bash
    uv run python -m ragas_examples.text2sql.data_utils --create-sample --samples 33 --validate --require-data
    ```

    生成的均衡 CSV 包含已验证、能够返回实际数据的查询。

    **预期输出如下。**

    ```
    📖 Loading data from BookSQL-files/BookSQL/train.json...
    📊 Loaded 70828 total records
    🚂 Found 70828 train records
    🔍 Removed 35189 duplicate records (same Query + SQL)
    📊 35639 unique records remaining
    📈 Difficulty distribution (after deduplication):
       • medium: 20576 records
       • hard: 11901 records
       • easy: 3162 records
    ✅ Added 33 validated 'easy' records
    ✅ Added 33 validated 'medium' records
    ✅ Added 33 validated 'hard' records
    💾 Saved 99 records to datasets/booksql_sample.csv
    📋 Final distribution:
       • medium: 33 records
       • hard: 33 records
       • easy: 33 records
    ```

    这样得到 `datasets/booksql_sample.csv`，包含跨不同难度的 99 条均衡样本。


BookSQL 采用 CC BY-NC-SA 许可证，仅限非商业用途。详情及引用格式如下。

??? example "📋 许可与引用信息"

    !!! warning "许可与使用"
        BookSQL 数据集采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。本示例按非商业研究用途使用，禁止商业使用。

    - **数据集**：[Hugging Face 上的 Exploration-Lab/BookSQL](https://huggingface.co/datasets/Exploration-Lab/BookSQL) · [GitHub 仓库](https://github.com/Exploration-Lab/BookSQL)
    - **论文**：ACL Anthology 的 [BookSQL，会计领域的大规模文本转 SQL 数据集](https://aclanthology.org/2024.naacl-long.28/)

    如果在研究中使用 BookSQL，请引用以下论文。

    ```bibtex
    @inproceedings{kumar-etal-2024-booksql,
        title = {BookSQL: A Large Scale Text-to-SQL Dataset for Accounting Domain},
        author = {Kumar, Rahul and Raja, Amar and Harsola, Shrutendra and Subrahmaniam, Vignesh and Modi, Ashutosh},
        booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
        month = {June},
        year = {2024},
        address = {Mexico City, Mexico},
        publisher = {Association for Computational Linguistics},
    }
    ```

创建自己的评估数据集的方法参阅[核心概念中的数据集](../../concepts/datasets.md)。

## 配置文本转 SQL 系统 { #set-up-your-text-to-sql-system }

### 创建提示词 { #create-your-prompt }

**提取数据库结构。**

```bash
uv run python -m ragas_examples.text2sql.db_utils --schema
```

??? example "📋 预期数据库结构"

    ```
    === Database Schema ===
                 name  type                                     sql
    chart_of_accounts table CREATE TABLE chart_of_accounts(
                             id INTEGER ,
                             businessID INTEGER NOT NULL,
                             Account_name TEXT NOT NULL,
                             Account_type TEXT NOT NULL,
                             PRIMARY KEY(id,businessID,Account_name)
                             )
            customers table CREATE TABLE customers(
                             id INTEGER ,
                             businessID INTEGER NOT NULL,
                             customer_name TEXT NOT NULL,
                             customer_full_name TEXT ,
                             ... (continues for all columns)
                             PRIMARY KEY(id,businessID,Customer_name)
                             )
    ... (continues for all 7 tables with complete DDL)
    ```

**编写提示词。**

提示词采用以下模板结构。

```text
You are a SQL query generator for a business accounting database. Convert natural language queries to SQL queries.

DATABASE CONTEXT:
This is an accounting database (accounting.sqlite) containing business transaction and entity data.

TABLES AND THEIR PURPOSE:

- master_txn_table: Main transaction records for all business transactions
- chart_of_accounts: Account names and their types for all businesses
- products_service: Products/services and their types used by businesses
- customers: Customer records with billing/shipping details
- vendors: Vendor records with billing address details
- payment_method: Payment methods used by businesses
- employees: Employee details including name, ID, hire date

DATABASE SCHEMA (DDL):

[Complete DDL statements for all tables]

INSTRUCTIONS:
Convert the user's natural language query into a valid SQL SELECT query. Return only the SQL query, no explanations or formatting.
```

## 定义评估指标 { #define-evaluation-metrics }

文本转 SQL 系统需要衡量结果准确性的指标。这里以执行准确率为主要指标，检查生成 SQL 是否返回正确数据。

**执行准确率指标**使用 [datacompy](https://github.com/capitalone/datacompy) 比较预期 SQL 与生成 SQL 的实际执行结果，验证两者是否返回相同数据。

结果分为以下两类。

- `"correct"`，查询成功且结果符合预期
- `"incorrect"`，查询失败，或成功执行但返回错误结果


### 设置指标函数 { #setting-up-metric-functions }

使用 [Ragas 离散指标](../../concepts/metrics/overview/index.md)创建评估指标。

```python
# File: examples/ragas_examples/text2sql/evals.py
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult
from ragas_examples.text2sql.db_utils import execute_sql

@discrete_metric(name="execution_accuracy", allowed_values=["correct", "incorrect"])
def execution_accuracy(expected_sql: str, predicted_success: bool, predicted_result):
    """Compare execution results of predicted vs expected SQL using datacompy."""
    try:
        # Execute expected SQL
        expected_success, expected_result = execute_sql(expected_sql)
        if not expected_success:
            return MetricResult(
                value="incorrect",
                reason=f"Expected SQL failed to execute: {expected_result}"
            )

        # If predicted SQL fails, it's incorrect
        if not predicted_success:
            return MetricResult(
                value="incorrect",
                reason=f"Predicted SQL failed to execute: {predicted_result}"
            )

        # Both queries succeeded - compare DataFrames using datacompy
        if isinstance(expected_result, pd.DataFrame) and isinstance(predicted_result, pd.DataFrame):
            # Handle empty DataFrames
            if expected_result.empty and predicted_result.empty:
                return MetricResult(value="correct", reason="Both queries returned empty results")

            if expected_result.empty != predicted_result.empty:
                return MetricResult(
                    value="incorrect",
                    reason=f"Expected returned {len(expected_result)} rows, predicted returned {len(predicted_result)} rows"
                )

            # Use datacompy to compare DataFrames with index-based comparison
            comparison = datacompy.Compare(
                expected_result.reset_index(drop=True),
                predicted_result.reset_index(drop=True),
                on_index=True,  # Compare row-by-row by index position
                abs_tol=1e-10,  # Very small tolerance for floating point comparison
                rel_tol=1e-10,
                df1_name='expected',
                df2_name='predicted'
            )

            if comparison.matches():
                return MetricResult(
                    value="correct",
                    reason=f"DataFrames match exactly ({len(expected_result)} rows, {len(expected_result.columns)} columns)"
                )
            else:
                return MetricResult(
                    value="incorrect",
                    reason="DataFrames do not match - different data returned"
                )

    except Exception as e:
        return MetricResult(
            value="incorrect",
            reason=f"Execution accuracy evaluation failed: {str(e)}"
        )
```

### 实验函数 { #the-experiment-function }

[实验函数](../../concepts/experimentation.md)调度完整评估流水线，运行文本转 SQL 智能体，并为每条查询计算指标。

```python
# File: examples/ragas_examples/text2sql/evals.py
from typing import Optional
from openai import AsyncOpenAI
from ragas import experiment
from ragas_examples.text2sql.text2sql_agent import Text2SQLAgent
from ragas_examples.text2sql.db_utils import execute_sql

@experiment()
async def text2sql_experiment(
    row,
    model: str,
    prompt_file: Optional[str],
):
    """Experiment function for text-to-SQL evaluation."""
    # Create text-to-SQL agent
    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    agent = Text2SQLAgent(
        client=openai_client,
        model_name=model,
        prompt_file=prompt_file
    )

    # Generate SQL from natural language query
    result = await agent.query(row["Query"])

    # Execute predicted SQL
    try:
        predicted_success, predicted_result = execute_sql(result["sql"])
    except Exception as e:
        predicted_success, predicted_result = False, f"SQL execution failed: {str(e)}"

    # Score the response using execution accuracy
    accuracy_score = await execution_accuracy.ascore(
        expected_sql=row["SQL"],
        predicted_success=predicted_success,
        predicted_result=predicted_result,
    )

    return {
        "query": row["Query"],
        "expected_sql": row["SQL"],
        "predicted_sql": result["sql"],
        "level": row["Levels"],
        "execution_accuracy": accuracy_score.value,
        "accuracy_reason": accuracy_score.reason,
    }
```

### 数据集加载器 { #dataset-loader }

将评估数据加载为 [Ragas Dataset](../../concepts/datasets.md) 对象，用于实验执行。

```python
# File: examples/ragas_examples/text2sql/evals.py
import pandas as pd
from pathlib import Path
from typing import Optional
from ragas import Dataset

def load_dataset(limit: Optional[int] = None):
    """Load the text-to-SQL dataset from CSV file."""
    dataset_path = Path(__file__).parent / "datasets" / "booksql_sample.csv"

    # Read CSV
    df = pd.read_csv(dataset_path)

    # Limit dataset size if requested
    if limit is not None and limit > 0:
        df = df.head(limit)

    # Create Ragas Dataset
    dataset = Dataset(name="text2sql_booksql", backend="local/csv", root_dir=".")

    for _, row in df.iterrows():
        dataset.append({
            "Query": row["Query"],
            "SQL": row["SQL"],
            "Levels": row["Levels"],
            "split": row["split"],
        })

    return dataset
```

加载器提供 `limit` 参数，便于开发时先用少量样本快速发现基础错误，再扩大到完整评估。

## 运行基线评估 { #run-baseline-evaluation }

### 执行评估流水线并收集结果 { #execute-evaluation-pipeline-and-collect-results }

```python
import asyncio
from ragas_examples.text2sql.evals import text2sql_experiment, load_dataset

async def run_evaluation():
    """Run text-to-SQL evaluation with direct code approach."""
    # Load dataset
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")

    # Run the experiment
    results = await text2sql_experiment.arun(
        dataset,
        name="gpt-5-mini-prompt-v1",
        model="gpt-5-mini",
        prompt_file=None,
    )

    # Report results
    print(f"✅ gpt-5-mini-prompt-v1: {len(results)} cases evaluated")

    # Calculate and display accuracy
    accuracy_rate = sum(1 for r in results if r["execution_accuracy"] == "correct") / max(1, len(results))
    print(f"gpt-5-mini-prompt-v1 Execution Accuracy: {accuracy_rate:.2%}")

# Run the evaluation
await run_evaluation()
```

??? example "📋 提示词 v1 输出"

    ```text
    Loading dataset...
    Dataset loaded with 99 samples
    Running text-to-SQL evaluation with model: gpt-5-mini
    Using prompt file: prompt.txt
    Running experiment: 100%|██████████████████████| 99/99 [01:06<00:00,  1.49it/s]
    ✅ gpt-5-mini-prompt-v1: 99 cases evaluated
    gpt-5-mini-prompt-v1 Execution Accuracy: 2.02%
    ```

**配置选项**

- `model`，所用 OpenAI 模型，默认为 `"gpt-5-mini"`
- `prompt_file`，自定义提示词文件，默认 None，使用内置提示词
- `limit`，样本数量，默认 None，评估全部样本，也可指定上限
- `name`，自定义实验名称，便于跟踪结果

### 初步表现分析 { #initial-performance-analysis }

评估生成详细 CSV，每行包含以下内容。

- `query`，自然语言输入
- `expected_sql`，标准 SQL
- `predicted_sql`，生成的 SQL
- `level`，难度，easy/medium/hard
- `execution_accuracy`，结果比较判定，correct/incorrect
- `accuracy_reason`，详细比较结果

## 分析错误与失败规律 { #analyze-errors-and-failure-patterns }

运行后可以分析失败规律，理解智能体的错误。以下为基线结果中的部分样本。

| 查询 | 预期 SQL | 生成 SQL | 难度 | 执行准确性 |
|-------|--------------|---------------|-------|-------------------|
| Richard Aguirre 的未结余额是多少？ | `select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Richard Aguirre" )` | `SELECT customer_name, customer_full_name, Balance FROM customers WHERE customer_name = 'Richard Aguirre' OR customer_full_name = 'Richard Aguirre';` | medium | incorrect |
| 客户 Natalie Myers 的未结信用余额是多少？ | `select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Natalie Myers" )` | `SELECT Balance FROM customers WHERE customer_name = 'Natalie Myers' OR customer_full_name = 'Natalie Myers';` | easy | incorrect |
| Ashley Thompson 的平均发票金额是多少？ | `select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Ashley Thompson" and transaction_type = 'invoice')` | `SELECT AVG(Amount) FROM master_txn_table WHERE Customers = 'Ashley Thompson' AND Transaction_TYPE = 'Invoice';` | hard | incorrect |
| 显示与 Carol Smith 的交易数量。 | `select count(distinct transaction_id) from master_txn_table where customers = 'Carol Smith' or vendor = 'Carol Smith'` | `SELECT COUNT(*) AS num_transactions FROM master_txn_table WHERE Customers = 'Carol Smith';` | medium | incorrect |

**这些样本的主要问题如下。**

**执行准确率为 0%**，这里展示的查询均未返回预期结果。

**常见失败模式**

  - 使用错误表，例如 `customers` 与 `master_txn_table` 混淆
  - 缺少去重，例如将 `count(distinct transaction_id)` 写成 `COUNT(*)`
  - 筛选不完整，例如遗漏 `OR vendor = 'Name'`
  - 列名错误，例如将 `open_balance` 写成 `Balance`

这表明智能体虽能生成有效 SQL，但对业务逻辑与数据库关系的理解仍需改进。

### 错误分析 { #error-analysis }

人工逐行检查 CSV 并标注错误类型，可开展系统化分析。也可以用以下提示词让 AI 辅助分类。

??? example "📋 错误分类提示词"

    ```text
    You are analyzing why a Text2SQL prediction failed. Given the following information, identify the error codes and provide a brief analysis.

    Available error codes:
    - AGGR_DISTINCT_MISSING: Used COUNT/SUM without DISTINCT or deduplication
    - WRONG_FILTER_COLUMN: Filtered on the wrong column
    - WRONG_SOURCE_TABLE_OR_COLUMN: Selected metric from the wrong table/column
    - EXTRA_TRANSFORMATION_OR_CONDITION: Added ABS(), extra filters that change results
    - OUTPUT_COLUMN_ALIAS_MISMATCH: Output column names don't match
    - NULL_OR_EMPTY_RESULT: Result is None/empty due to wrong filters or source
    - GENERIC_VALUE_MISMATCH: Aggregation computed but numeric value differs for unclear reasons
    - OTHER: Fallback

    Query: [YOUR_QUERY]
    Expected SQL: [EXPECTED_SQL]
    Predicted SQL: [PREDICTED_SQL]
    Execution Accuracy: [ACCURACY_RESULT]
    Accuracy Reason: [ACCURACY_REASON]

    Respond with:
    - error_codes: array of applicable error codes (1 or more)
    - error_analysis: brief 1-3 sentence explanation of what went wrong
    ```

    将此提示词交给所选 LLM，分析结果 CSV 中的单条失败样本。

我们还提供了使用 GPT-5 自动分类的脚本，运行 `uv run python -m ragas_examples.text2sql.analyze_errors --input experiments/your_results.csv` 即可。

### 复核流程 { #review-process }

1. 查看汇总中**高频错误码**
2. 每种主要错误类型**人工检查 5～10 条样本**
3. **验证 AI 分类**是否符合你的理解
4. **确认预期 SQL 本身是否正确**，且符合数据库结构
5. **寻找 AI 遗漏的规律**
6. 根据影响与频率**确定修复优先级**

经过人工验证后，再将分析用于改进提示词、少样本示例或智能体架构。

### 错误分析示例结果 { #sample-error-analysis-results }

以下标注行展示第一轮改进后仍存在的问题。

| 查询 | 预期 SQL | 生成 SQL | 错误分析 | 错误码 |
|-------|--------------|---------------|----------------|-------------|
| Fuel 相关业务收到了多少收入？ | `select sum(credit) from master_txn_table as T1 join chart_of_accounts as T2 on T1.account = T2.account_name where account_type in ('Income', 'Other Income') and instr(account,"Fuel")` | `SELECT SUM(Amount) FROM (SELECT DISTINCT Transaction_ID, Amount FROM master_txn_table WHERE Customers = 'Fuel');` | 错误地对 Amount 求和，未连接 chart_of_accounts。还使用 Customers='Fuel'，遗漏账户名称包含 Fuel 及 Income/Other Income 类型的筛选。 | `["WRONG_SOURCE_TABLE_OR_COLUMN", "WRONG_FILTER_COLUMN", "NULL_OR_EMPTY_RESULT"]` |
| Roberta Shaw 最近一张发票的编号是什么？ | `select transaction_id from master_txn_table where transaction_type = 'invoice' and customers = "Roberta Shaw" order by transaction_date desc limit 1` | `SELECT Transaction_ID, Transaction_DATE, Amount, Open_balance, CreatedDATE, CreatedUSER, Account, Customers, Product_Service FROM master_txn_table WHERE Transaction_TYPE = 'invoice' AND Customers = 'Roberta Shaw' ORDER BY Transaction_DATE DESC LIMIT 1;` | 预期只返回 Transaction_ID，但生成查询选择了九列。虽然返回同一行，列结构仍不匹配。 | `["OUTPUT_COLUMN_ALIAS_MISMATCH"]` |
| Jeremy Strong 的平均发票金额是多少？ | `select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Jeremy Strong" and transaction_type = 'invoice')` | `SELECT AVG(Amount) FROM (SELECT DISTINCT Transaction_ID, Amount FROM master_txn_table WHERE Transaction_TYPE = 'invoice' AND Vendor = 'Jeremy Strong') AS t;` | 错误地使用 Vendor 筛选，正确字段应为 customers，可能匹配不到记录，使 AVG(amount) 返回 NULL。 | `["WRONG_FILTER_COLUMN", "NULL_OR_EMPTY_RESULT"]` |

**结果中的主要发现**

- **错误模式**
  - **缺少 OR 条件**，查询“与某人”的交易应同时检查 `customers` 和 `vendor`
  - **选错列**，财务查询中将 `credit` 错写为 `Amount`
  - **输出结构不匹配**，选择过多列或列名不正确
  - **缺少连接**，未连接 `chart_of_accounts` 以按账户类型筛选

下一轮重点完善筛选逻辑和财务查询处理。

根据通用规则调整提示词，避免针对单行数据打补丁。优先添加以数据库结构为依据的约束，减少过拟合。

重复以下闭环。

- 运行 → 标注 → 复核 → 确定通用约束 → 更新 `prompt_vX.txt` → 重跑 → 比较 → 继续迭代。
- 约束应简洁，且以数据库结构为依据，确保改进能够泛化。
- 为提示词管理版本（`prompt_v2.txt`、`prompt_v3.txt`、`prompt_v4.txt`），每版记录简要变更。
- 执行准确率连续两轮不再提高，或达到业务要求时停止。

## 改进系统 { #improve-your-system }

### 创建并使用新版提示词 { #create-and-use-a-new-prompt-version }

保留基线提示词，另建新版本进行迭代。

创建 `prompt_v2.txt`，加入简洁、可复用且符合数据库结构的通用约束。以下是从 `prompt_v1.txt` 生成 v2 时补充的一部分内容。

```text
- Use exact table and column names from the schema; do not invent fields
- Prefer transactional facts from `master_txn_table`; use entity tables for static attributes
- Map parties correctly in filters:
  - Customer-focused → filter on `Customers`
  - Vendor-focused → filter on `Vendor`
- Disambiguate events via `Transaction_TYPE` (e.g., invoices → `Transaction_TYPE = 'invoice'`)
- Avoid double-counting by deduplicating on `Transaction_ID` for counts and aggregates:
  - Counts: `count(distinct Transaction_ID)`
  - Aggregates: compute over a deduplicated subquery on `(Transaction_ID, metric_column)`
- For open credit/balance due per customer, aggregate `Open_balance` from `master_txn_table` filtered by `Customers` with deduplication
- Do not add extra transforms or filters (e.g., `abs()`, `< 0`) unless explicitly asked
- Keep a single `SELECT`; avoid aliases for final column names
```

将改进后的提示词保存为 `prompt_v2.txt`。

### 使用新提示词重新评估 { #re-run-evaluation-with-the-new-prompt }

```python
import asyncio
from ragas_examples.text2sql.evals import text2sql_experiment, load_dataset

async def run_v2_evaluation():
    """Run evaluation with prompt v2."""
    # Load dataset
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")

    # Run experiment
    results = await text2sql_experiment.arun(
        dataset,
        name="gpt-5-mini-prompt-v2",
        model="gpt-5-mini",
        prompt_file="prompt_v2.txt",
    )

    # Report results
    print(f"✅ gpt-5-mini-prompt-v2: {len(results)} cases evaluated")

    # Calculate accuracy
    accuracy_rate = sum(1 for r in results if r["execution_accuracy"] == "correct") / max(1, len(results))
    print(f"gpt-5-mini-prompt-v2 Execution Accuracy: {accuracy_rate:.2%}")

await run_v2_evaluation()
```

??? example "📋 提示词 v2 输出"

    ```text
    Loading dataset...
    Dataset loaded with 99 samples
    Running text-to-SQL evaluation with model: gpt-5-mini
    Using prompt file: prompt_v2.txt
    Running experiment: 100%|██████████████████████| 99/99 [01:00<00:00,  1.63it/s]
    ✅ gpt-5-mini-prompt-v2: 99 cases evaluated
    gpt-5-mini-prompt-v2 Execution Accuracy: 60.61%
    ```

使用 `prompt_v2` 后，执行准确率从 2.02% 提升到 60.61%。

检查 `experiments/` 中的新 CSV，继续迭代。

### 继续迭代，创建提示词 v3 { #continue-iterating-create-prompt-v3 }

v2 带来明显改进，但约 60% 的准确率仍有提升空间。深入分析发现以下重复出现的问题。

1. **误解财务概念**。模型常默认汇总 `Amount`，应根据任务使用收入列 `Credit` 或支出列 `Debit`，还经常遗漏与 `chart_of_accounts` 的连接，未按 Income 等账户类型筛选。
2. **添加多余变换**。无须的 `DISTINCT` 或 `Transaction_TYPE = 'invoice'` 等额外筛选会改变结果。
3. **列选择错误**。查询“显示所有交易”时，常使用 `SELECT *`，而预期为 `SELECT DISTINCT Transaction_ID`，导致结构不匹配。聚合结果列名也可能错误，例如 `max(transaction_date)` 与 `transaction_date` 不一致。
4. **筛选不完整**。遗漏 OR 条件，或完全选错筛选列，例如未同时检查 Customers 和 Vendor。

根据分析，创建 `prompt_v3.txt`，加入更具体、符合数据库结构的规则。

`prompt_v3.txt` 的主要补充如下。

```text
### CORE QUERY GENERATION GUIDELINES

1.  **Use Correct Schema**: Use exact table and column names...
2.  **Simplicity First**: Keep the query as simple as possible...
...

### ADVANCED QUERY PATTERNS

5.  **Financial Queries (Revenue, Sales, Expenses)**:
    -   **Metric Selection**:
        -   For revenue, income, sales, or money **received**: aggregate the `Credit` column.
        -   For expenses, bills, or money **spent**: aggregate the `Debit` column.
        -   Use the `Amount` column only when...
    -   **Categorical Financial Queries**: For questions involving financial categories... you **MUST** `JOIN` `master_txn_table` with `chart_of_accounts`...

6.  **Filtering Logic**:
    -   **Ambiguous Parties**: For questions about transactions "with" or "involving" a person or company, you **MUST** check both `Customers` and `Vendor` columns. E.g., `WHERE Customers = 'Name' OR Vendor = 'Name'`.
    -   **Avoid Extra Filters**: Do not add implicit filters...

7.  **Column Selection and Naming**:
    -   **Avoid `SELECT *`**: When asked to "show all transactions", return only `DISTINCT Transaction_ID`...
    -   **"Most Recent" / "Last" Queries**: To get the 'most recent' or 'last' record, use `ORDER BY Transaction_DATE DESC LIMIT 1`. This preserves the original column names... Avoid using `MAX()`...

```

新规则保持通用性，同时直接针对观察到的失败模式。

**使用 `prompt_v3.txt` 重新评估。**

```python
import asyncio
from ragas_examples.text2sql.evals import text2sql_experiment, load_dataset

async def run_v3_evaluation():
    """Run evaluation with prompt v3."""
    # Load dataset
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")

    # Run experiment
    results = await text2sql_experiment.arun(
        dataset,
        name="gpt-5-mini-prompt-v3",
        model="gpt-5-mini",
        prompt_file="prompt_v3.txt",
    )

    # Report results
    print(f"✅ gpt-5-mini-prompt-v3: {len(results)} cases evaluated")

    # Calculate accuracy
    accuracy_rate = sum(1 for r in results if r["execution_accuracy"] == "correct") / max(1, len(results))
    print(f"gpt-5-mini-prompt-v3 Execution Accuracy: {accuracy_rate:.2%}")

await run_v3_evaluation()
```

使用 `prompt_v3` 后，执行准确率从 60.61% 提升到 70.71%。

### 持续迭代的原则 { #key-principles-for-continued-iteration }

v3 约 70% 的准确率体现了系统化迭代的效果，还可以继续这一过程。

**持续迭代时注意以下原则。**

- 每轮针对最新结果中 **3～5 类高频错误**
- 新规则保持**通用性，并以数据库结构为依据**
- 准确率连续两三轮不再提高时，**结束本轮迭代**
- 提示词优化达到瓶颈时，可以尝试更强模型，或将 SQL 错误反馈给 LLM 自行修复，构建更完整的智能体流程。

## 比较结果 { #compare-results }

运行全部版本后，比较最终结果。

| 提示词 | 执行准确率 | 结果 CSV |
|---|---|---|
| v1（`prompt.txt`） | 2.02% | `experiments/...-prompt-v1.csv` |
| v2（`prompt_v2.txt`） | 60.61% | `experiments/...-prompt-v2.csv` |
| v3（`prompt_v3.txt`） | 70.71% | `experiments/...-prompt-v3.csv` |

**进展分析**
- **v1 → v2**。通过基本去重与业务逻辑规则，从 2.02% 提升到 60.61%，约提高 58.6 个百分点。
- **v2 → v3**。完善财务查询、筛选和列选择规则，从 60.61% 提升到 70.71%，再提高约 10.1 个百分点。
- 改进针对错误分析中识别出的财务概念、多余变换与筛选不完整等模式。

## 总结 { #conclusion }

本指南建立了文本转 SQL 系统的系统化评估流程。

**主要做法如下。**

- 使用执行准确率比较真实查询结果
- 按“评估 → 分析错误 → 改进 → 重复”的流程迭代

Ragas 自动处理调度与结果汇总，为持续衡量和改进系统提供可靠基础。
