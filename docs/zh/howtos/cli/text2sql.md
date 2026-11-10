# 文本转 SQL 评估快速入门 { #text-to-sql-evaluation-quickstart }

`text2sql` 模板通过比较 SQL 执行结果，评估文本转 SQL 系统。

## 创建项目 { #create-the-project }

```sh
ragas quickstart text2sql
cd text2sql
```

## 安装依赖 { #install-dependencies }

```sh
uv sync
```

## 设置 API 密钥 { #set-your-api-key }

```sh
export OPENAI_API_KEY="your-openai-key"
```

## 运行评估 { #run-the-evaluation }

```sh
uv run python evals.py
```

## 项目结构 { #project-structure }

```
text2sql/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── text2sql_agent.py      # Text-to-SQL agent
├── db_utils.py            # Database utilities
├── evals.py               # Evaluation workflow
├── prompt.txt             # Base prompt template
├── prompt_v2.txt          # Improved prompt v2
├── prompt_v3.txt          # Improved prompt v3
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/
    │   └── booksql_sample.csv  # Sample book database queries
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## 评估内容 { #what-it-evaluates }

模板评估从自然语言生成 SQL 的能力。

- **智能体**，将自然语言转换为 SQL 查询
- **数据库**，包含作者、书名和类型的示例图书数据库
- **测试用例**，自然语言问题及预期 SQL 查询
- **指标**，通过 datacompy 比较查询结果，衡量执行准确率

## 理解代码 { #understanding-the-code }

### 智能体（`text2sql_agent.py`） { #the-agent-text2sql_agentpy }

将自然语言转换为 SQL。

```python
from text2sql_agent import Text2SQLAgent

agent = Text2SQLAgent(client=openai_client)
sql = await agent.generate_sql("Find all books by Jane Austen")
```

### 评估（`evals.py`） { #the-evaluation-evalspy }

比较执行结果。

```python
@discrete_metric(name="execution_accuracy", allowed_values=["correct", "incorrect"])
def execution_accuracy(expected_sql: str, predicted_success: bool, predicted_result):
    # Executes both SQLs and compares results using datacompy
    # Returns "correct" if results match, "incorrect" otherwise
```

## 测试数据 { #test-data }

模板的 `evals/datasets/booksql_sample.csv` 包含图书数据库的示例问题和预期 SQL 查询。

## 自定义 { #customization }

### 使用自己的数据库 { #use-your-own-database }

更新 `db_utils.py`，连接自己的数据库。

```python
def get_db_connection():
    return sqlite3.connect("your_database.db")
```

### 尝试不同提示词 { #try-different-prompts }

模板在 `prompt.txt`、`prompt_v2.txt` 和 `prompt_v3.txt` 中提供三个提示词版本，可以逐一测试，比较效果。

## 下一步 { #next-steps }

- [智能体评估](agent_evals.md)，评估 AI 智能体
- [工作流评估](workflow_eval.md)，评估复杂工作流
