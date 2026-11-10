# 提示词评估快速入门 { #prompt-evaluation-quickstart }

`prompt_evals` 模板通过情感分析任务，评估并比较不同提示词变体。

## 创建项目 { #create-the-project }

```sh
ragas quickstart prompt_evals
cd prompt_evals
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
prompt_evals/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── prompt.py              # Prompt implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## 评估内容 { #what-it-evaluates }

模板评估提示词在情感分类任务中的效果。

- **任务**，正面/负面情感分析
- **测试用例**，带有预期情感标签的电影评论
- **指标**，二元准确性，判定通过或失败

## 理解代码 { #understanding-the-code }

### 提示词（`prompt.py`） { #the-prompt-promptpy }

实现情感分析提示词。

```python
from prompt import run_prompt

sentiment = run_prompt("I loved the movie! It was fantastic.")
# Returns: "positive" or "negative"
```

### 评估（`evals.py`） { #the-evaluation-evalspy }

测试提示词准确率。

```python
@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def my_metric(prediction: str, actual: str):
    return (
        MetricResult(value="pass", reason="")
        if prediction == actual
        else MetricResult(value="fail", reason="")
    )
```

## 测试数据 { #test-data }

数据集包含电影评论。

```python
dataset_dict = [
    {"text": "I loved the movie! It was fantastic.", "label": "positive"},
    {"text": "The movie was terrible and boring.", "label": "negative"},
    # More examples...
]
```

## 自定义 { #customization }

### 测试不同提示词 { #test-different-prompts }

修改 `prompt.py`，测试不同变体。

```python
# Version 1: Simple
prompt = f"Is this positive or negative: {text}"

# Version 2: With examples
prompt = f"""Classify sentiment:
Examples:
- "Great movie" -> positive
- "Boring film" -> negative

Text: {text}
Sentiment:"""

# Compare results across versions
```

### 添加更多指标 { #add-more-metrics }

评估其他维度。

```python
from ragas.metrics import NumericalMetric

confidence = NumericalMetric(
    name="confidence",
    prompt="Rate confidence 1-5 in this classification: {prediction}",
    allowed_values=(1, 5),
)
```

## 下一步 { #next-steps }

- [裁判对齐](judge_alignment.md)，衡量 LLM 裁判与人工的一致性
- [LLM 基准测试](benchmark_llm.md)，比较不同模型
