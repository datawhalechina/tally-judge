# 提示词评估 { #prompt-evaluation }

本教程将编写一条简单的评估流水线，评估 AI 系统中的提示词。示例是一个电影评论情感分类器。完成后，你将掌握如何通过评估驱动开发来评估和迭代单个提示词。

```mermaid
flowchart LR
    A["'这部电影太棒了！<br/>表演和剧情都很好。'"] --> B["分类提示词"]
    B --> C["正面"]
```


首先测试一个简单的提示词，将电影评论分为正面或负面。

先安装 ragas 示例包，并设置 OpenAI API 密钥。

```bash
pip install ragas[examples]
export OPENAI_API_KEY = "your_openai_api_key"
```

然后测试提示词。

```bash
python -m ragas_examples.prompt_evals.prompt
```

这会测试输入 `"The movie was fantastic and I loved every moment of it!"`，预期输出为 `"positive"`。

> **💡 快速开始**。要直接查看完整评估过程，可以跳到[端到端运行命令](#running-the-example-end-to-end)，一次完成所有步骤并自动生成 CSV 结果。

接下来，为提示词准备几组示例输入和预期输出，并保存为 CSV 文件。

```python
import pandas as pd

samples = [{"text": "I loved the movie! It was fantastic.", "label": "positive"},
    {"text": "The movie was terrible and boring.", "label": "negative"},
    {"text": "It was an average film, nothing special.", "label": "positive"},
    {"text": "Absolutely amazing! Best movie of the year.", "label": "positive"}]
pd.DataFrame(samples).to_csv("datasets/test_dataset.csv", index=False)
```

我们需要衡量提示词在该任务上的表现。下面定义一个指标，比较提示词输出与预期输出，并返回通过或失败。

```python
from ragas.metrics import discrete_metric
from ragas.metrics.result import MetricResult

@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def my_metric(prediction: str, actual: str):
    """Calculate accuracy of the prediction."""
    return MetricResult(value="pass", reason="") if prediction == actual else MetricResult(value="fail", reason="")
```

然后编写实验循环，在测试数据集上运行提示词，用指标评估输出，并将结果保存为 CSV 文件。

```python
from ragas import experiment

@experiment()
async def run_experiment(row):

    response = run_prompt(row["text"])
    score = my_metric.score(
        prediction=response,
        actual=row["label"]
    )

    experiment_view = {
        **row,
        "response":response,
        "score":score.value,
    }
    return experiment_view
```

此后，每次修改提示词，都可以运行实验，观察改动对表现的影响。

### 传入附加参数 { #passing-additional-parameters }

可以向实验函数传入模型或配置等附加参数。

```python
@experiment()
async def run_experiment(row, model):
    response = run_prompt(row["text"], model=model)
    score = my_metric.score(
        prediction=response,
        actual=row["label"]
    )

    experiment_view = {
        **row,
        "response": response,
        "score": score.value,
    }
    return experiment_view

# Run with specific parameters
run_experiment.arun(dataset, "gpt-4")

# Or use keyword arguments
run_experiment.arun(dataset, model="gpt-4o")
```


## 端到端运行示例 { #running-the-example-end-to-end }

1. 设置 OpenAI API 密钥
```bash
export OPENAI_API_KEY = "your_openai_api_key"
```
2. 运行评估
```bash
python -m ragas_examples.prompt_evals.evals
```

这将完成以下操作。

- 创建包含电影评论示例的测试数据集
- 对每个样本运行情感分类提示词
- 使用准确率指标评估结果
- 将全部结果导出为 CSV 文件

你已经使用 Ragas 完成了第一次评估！打开 `experiments/experiment_name.csv` 文件即可查看结果。
