# LlamaIndex 智能体评估快速入门 { #llamaindex-agent-evaluation-quickstart }

`llamaIndex_agent_evals` 模板通过工具调用准确率指标，评估 LlamaIndex 工作流智能体。

## 创建项目 { #create-the-project }

```sh
ragas quickstart llamaIndex_agent_evals
cd llamaIndex_agent_evals
```

## 安装依赖 { #install-dependencies }

```sh
uv sync
```

## 设置 API 密钥 { #set-your-api-keys }

```sh
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"  # For evaluator LLM
```

## 运行评估 { #run-the-evaluation }

```sh
uv run python evals.py
```

## 项目结构 { #project-structure }

```
llamaIndex_agent_evals/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── llamaindex_agent.py    # LlamaIndex agent with tools
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/
    │   └── contexts/      # Test context files (JSON)
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## 评估内容 { #what-it-evaluates }

模板评估 LlamaIndex 智能体的工具调用准确性。

- **智能体**，带有添加、删除和列出列表项等工具的 LlamaIndex `FunctionAgent`
- **测试用例**，重复添加、含糊的删除请求等复杂场景
- **指标**，工具调用准确性与回答正确性

## 理解代码 { #understanding-the-code }

### 智能体（`llamaindex_agent.py`） { #the-agent-llamaindex_agentpy }

带有简单工具的 LlamaIndex 智能体。

```python
from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(
    name="list_manager",
    tools=[add_item, remove_item, list_items],
    llm=llm
)
```

### 评估（`evals.py`） { #the-evaluation-evalspy }

使用 F1 得分测试工具调用准确性。

```python
@numeric_metric(name="tool_call_accuracy")
def tool_call_accuracy_metric(predicted_calls: List[Dict], ground_truth_calls: List[Dict]):
    # Compares predicted vs ground truth tool calls
    # Returns F1 score between 0.0 and 1.0
```

## 测试数据 { #test-data }

模板的 `evals/datasets/contexts/` 包含 JSON 测试上下文。

- `ambiguous_removal_request.json`，测试含糊请求的处理
- `duplicate_addition.json`，测试重复操作的处理
- `repeated_removal.json`，测试重复删除

## 下一步 { #next-steps }

- [智能体评估](agent_evals.md)，评估通用 AI 智能体
- [工作流评估](workflow_eval.md)，评估复杂工作流
