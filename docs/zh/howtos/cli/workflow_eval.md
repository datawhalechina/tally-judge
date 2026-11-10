# 工作流评估快速入门 { #workflow-evaluation-quickstart }

`workflow_eval` 模板通过邮件分类与路由，评估复杂 LLM 工作流。

## 创建项目 { #create-the-project }

```sh
ragas quickstart workflow_eval
cd workflow_eval
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
workflow_eval/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── workflow.py            # Workflow implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## 评估内容 { #what-it-evaluates }

模板评估客服邮件分类工作流。

- **工作流**，多步处理邮件，依次分类、提取信息和生成回复
- **类别**，缺陷报告、功能请求、账单问题
- **测试用例**，客户邮件及其预期类别和应提取字段
- **指标**，检查分类准确性的自定义离散指标

## 理解代码 { #understanding-the-code }

### 工作流（`workflow.py`） { #the-workflow-workflowpy }

实现客服邮件工作流。

```python
from workflow import default_workflow_client

workflow = default_workflow_client()
result = workflow.process_email("I found a bug in version 2.1.4...")
# Returns: category, extracted fields, response
```

### 评估（`evals.py`） { #the-evaluation-evalspy }

根据通过标准测试工作流准确性。

```python
def load_dataset():
    dataset_dict = [
        {
            "email": "Hi, I'm getting error code XYZ-123 when using version 2.1.4...",
            "pass_criteria": "category Bug Report; product_version 2.1.4; error_code XYZ-123",
        },
        # More test cases...
    ]
```

指标检查工作流能否正确完成以下操作。
- 判断邮件类别
- 提取版本、错误码、发票号等相关字段
- 生成恰当回复

## 测试用例 { #test-cases }

模板包含多种场景。

- **缺陷报告**，含版本号与错误码
- **功能请求**，含紧急程度与产品领域
- **账单问题**，含发票号与金额

## 自定义 { #customization }

### 添加自己的工作流 { #add-your-own-workflow }

用自己的工作流替换示例。

```python
from your_workflow import YourWorkflow

workflow = YourWorkflow()

@experiment()
async def run_experiment(row):
    result = await workflow.process(row["input"])
    # Evaluate result...
```

## 下一步 { #next-steps }

- [智能体评估](agent_evals.md)，评估 AI 智能体
- [LlamaIndex 智能体评估](llamaIndex_agent_evals.md)，评估 LlamaIndex 工作流
