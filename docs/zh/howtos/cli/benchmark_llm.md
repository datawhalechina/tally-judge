# LLM 基准测试快速入门 { #llm-benchmarking-quickstart }

`benchmark_llm` 模板在折扣计算任务上测试并比较不同 LLM。

## 创建项目 { #create-the-project }

```sh
ragas quickstart benchmark_llm
cd benchmark_llm
```

## 安装依赖 { #install-dependencies }

```sh
uv sync
```

## 设置 API 密钥 { #set-your-api-keys }

```sh
export OPENAI_API_KEY="your-openai-key"
# Or other provider keys as needed
```

## 运行评估 { #run-the-evaluation }

```sh
uv run python evals.py
```

测试某个指定模型。

```sh
uv run python evals.py --model gpt-4o
uv run python evals.py --model gpt-3.5-turbo
```

## 项目结构 { #project-structure }

```
benchmark_llm/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── prompt.py              # Prompt implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/
    │   └── discount_benchmark.csv  # Customer profiles and expected discounts
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## 评估内容 { #what-it-evaluates }

模板衡量 LLM 在结构化输出任务中的表现。

- **任务**，根据客户资料计算折扣百分比
- **模型**，比较 GPT-4、GPT-3.5、Claude、Gemini 等
- **输出格式**，包含折扣百分比的 JSON
- **指标**，折扣准确性，判断正确或错误

## 理解代码 { #understanding-the-code }

### 提示词（`prompt.py`） { #the-prompt-promptpy }

根据客户资料计算折扣。

```python
from prompt import run_prompt

profile = "Premium customer, 5 years tenure, $50k annual spend"
result = await run_prompt(profile, model="gpt-4o")
# Returns: {"discount_percentage": 15}
```

### 评估（`evals.py`） { #the-evaluation-evalspy }

测试模型准确率。

```python
@discrete_metric(name="discount_accuracy", allowed_values=["correct", "incorrect"])
def discount_accuracy(prediction: str, expected_discount):
    parsed_json = json.loads(prediction)
    predicted_discount = parsed_json.get("discount_percentage")

    if predicted_discount == int(expected_discount):
        return MetricResult(value="correct", ...)
    else:
        return MetricResult(value="incorrect", ...)
```

## 测试数据 { #test-data }

模板的 `evals/datasets/discount_benchmark.csv` 包含以下内容。

- 客户资料，例如客户年限、消费额和等级
- 预期折扣百分比
- 折扣计算的业务规则

## 测试多个模型 { #benchmarking-multiple-models }

使用不同模型运行同一评估。

```sh
# GPT-4
uv run python evals.py --model gpt-4o

# GPT-3.5
uv run python evals.py --model gpt-3.5-turbo

# Claude
uv run python evals.py --model claude-3-5-sonnet-20241022

# Compare results
```

## 自定义 { #customization }

### 添加自己的任务 { #add-your-own-task }

修改提示词，测试其他能力。

```python
# Code generation
prompt = "Generate Python code to {task}"

# Summarization
prompt = "Summarize this text in 50 words: {text}"

# Classification
prompt = "Classify this email as spam/not-spam: {email}"
```

### 比较成本与延迟 { #compare-cost-and-latency }

跟踪更多指标。

```python
import time

start = time.time()
response = await run_prompt(profile, model=model_name)
latency = time.time() - start

# Log cost and latency alongside accuracy
```

## 分析结果 { #analyzing-results }

比较模型表现。

```python
import pandas as pd

gpt4_results = pd.read_csv("evals/experiments/gpt4_benchmark.csv")
gpt35_results = pd.read_csv("evals/experiments/gpt35_benchmark.csv")

print(f"GPT-4 Accuracy: {(gpt4_results['discount_accuracy'] == 'correct').mean():.1%}")
print(f"GPT-3.5 Accuracy: {(gpt35_results['discount_accuracy'] == 'correct').mean():.1%}")
```

## 下一步 { #next-steps }

- [裁判对齐](judge_alignment.md)，衡量裁判与人工的一致性
- [提示词评估](prompt_evals.md)，比较不同提示词
