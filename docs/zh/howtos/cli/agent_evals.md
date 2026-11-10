# 智能体评估快速入门 { #agent-evaluation-quickstart }

`agent_evals` 模板通过正确性指标，评估求解数学问题的 AI 智能体。

## 创建项目 { #create-the-project }

```sh
ragas quickstart agent_evals
cd agent_evals
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
agent_evals/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── agent.py               # Math solving agent implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## 评估内容 { #what-it-evaluates }

模板评估 AI 智能体求解数学表达式的能力。

- **智能体**，通过工具逐步求解数学问题
- **测试用例**，例如 `(2 + 3) * (6 - 2)`、`100 / 5 + 3 * 2`
- **指标**，二元正确性，正确为 1.0，错误为 0.0

## 理解代码 { #understanding-the-code }

### 智能体（`agent.py`） { #the-agent-agentpy }

实现带计算器工具的数学智能体。

```python
from agent import get_default_agent

math_agent = get_default_agent()
result = math_agent.solve("15 - 3 / 4")
```

### 评估（`evals.py`） { #the-evaluation-evalspy }

使用多种数学问题测试智能体。

```python
@numeric_metric(name="correctness", allowed_values=(0.0, 1.0))
def correctness_metric(prediction: float, actual: float):
    result = 1.0 if abs(prediction - actual) < 1e-5 else 0.0
    return MetricResult(value=result, reason=f"Prediction: {prediction}, Actual: {actual}")
```

## 下一步 { #next-steps }

- [LlamaIndex 智能体评估](llamaIndex_agent_evals.md)，评估 LlamaIndex 智能体
- [自定义指标](../../concepts/metrics/overview/index.md)，编写自己的指标
