# 运行第一个实验 { #run-your-first-experiment }

本教程使用 `@experiment` 装饰器和本地 CSV 后端，带你运行第一个 Ragas 实验。

## 前置条件 { #prerequisites }

- Python 3.9+
- 已安装 Ragas，参见[安装说明](install.md)

## Hello World 👋

![](../../_static/imgs/experiments_quickstart/hello_world.gif)

### 1. 安装（如果尚未安装） { #1-install-if-you-havent-already }

```bash
pip install ragas
```

### 2. 创建 `hello_world.py` { #2-create-hello_worldpy }

将以下代码复制到新文件，保存为 `hello_world.py`。

```python
import numpy as np
from ragas import Dataset, experiment
from ragas.metrics import MetricResult, discrete_metric


# Define a custom metric for accuracy
@discrete_metric(name="accuracy_score", allowed_values=["pass", "fail"])
def accuracy_score(response: str, expected: str):
    result = "pass" if expected.lower().strip() == response.lower().strip() else "fail"
    return MetricResult(value=result, reason=f"Match: {result == 'pass'}")


# Mock application endpoint that simulates an AI application response
def mock_app_endpoint(**kwargs) -> str:
    return np.random.choice(["Paris", "4", "Blue Whale", "Einstein", "Python"])


# Create an experiment that uses the mock application endpoint and the accuracy metric
@experiment()
async def run_experiment(row):
    response = mock_app_endpoint(query=row.get("query"))
    accuracy = accuracy_score.score(response=response, expected=row.get("expected_output"))
    return {**row, "response": response, "accuracy": accuracy.value}


if __name__ == "__main__":
    import asyncio

    # Create dataset inline
    dataset = Dataset(name="test_dataset", backend="local/csv", root_dir=".")
    test_data = [
        {"query": "What is the capital of France?", "expected_output": "Paris"},
        {"query": "What is 2 + 2?", "expected_output": "4"},
        {"query": "What is the largest animal?", "expected_output": "Blue Whale"},
        {"query": "Who developed the theory of relativity?", "expected_output": "Einstein"},
        {"query": "What programming language is named after a snake?", "expected_output": "Python"},
    ]

    for sample in test_data:
        dataset.append(sample)
    dataset.save()

    # Run experiment
    _ = asyncio.run(run_experiment.arun(dataset, name="first_experiment"))
```

### 3. 检查生成的文件 { #3-inspect-the-generated-files }

```bash
tree .
```

你应当能看到以下内容。

```
├── datasets
│   └── test_dataset.csv
└── experiments
    └── first_experiment.csv
```

### 4. 查看第一个实验的结果 { #4-view-the-results-of-your-first-experiment }

```bash
open experiments/first_experiment.csv
```

输出预览如下。

![](../../_static/imgs/experiments_quickstart/output_first_experiment.png)

## 下一步 { #next-steps }

- 阅读[实验的核心概念](../concepts/experimentation.md)，了解实验背后的原理
- 在[评估指标](../concepts/metrics/index.md)中探索可用指标

