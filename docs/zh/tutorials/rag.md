# 评估简单的 RAG 系统 { #evaluate-a-simple-rag-system }

本教程将编写一条简单的评估流水线，用于评估 RAG（检索增强生成）系统。完成后，你将掌握如何通过评估驱动开发来评估和迭代 RAG 系统。

```mermaid
flowchart LR
    A["查询<br/>'什么是 Ragas 0.3？'"] --> B[检索系统]

    C[文档语料<br/> Ragas 0.3 文档📄] --> B

    B --> D[LLM + 提示词]
    A --> D

    D --> E[最终答案]
```

首先编写一个简单的 RAG 系统，从语料中检索相关文档，再使用 LLM 生成答案。

```bash
python -m ragas_examples.rag_eval.rag
```


接着，为 RAG 系统准备几组示例查询和预期输出，保存为 CSV 文件。

```python
import pandas as pd

samples = [
    {"query": "What is Ragas 0.3?", "grading_notes": "- Ragas 0.3 is a library for evaluating LLM applications."},
    {"query": "How to install Ragas?", "grading_notes": "- install from source  - install from pip using ragas[examples]"},
    {"query": "What are the main features of Ragas?", "grading_notes": "organised around - experiments - datasets - metrics."}
]
pd.DataFrame(samples).to_csv("datasets/test_dataset.csv", index=False)
```

为了衡量 RAG 系统的表现，我们将定义一个基于 LLM 的指标，比较系统输出与评分说明，并返回通过或失败。

```python
from ragas.metrics import DiscreteMetric
my_metric = DiscreteMetric(
    name="correctness",
    prompt = "Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)
```

然后编写实验循环，在测试数据集上运行 RAG 系统，用指标评估输出，并将结果保存为 CSV 文件。

```python
@experiment()
async def run_experiment(row):
    response = rag_client.query(row["query"])

    score = my_metric.score(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"]
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
    }
    return experiment_view
```

此后，每次修改 RAG 流水线，都可以运行实验，观察其表现如何变化。

## 端到端运行示例 { #running-the-example-end-to-end }

1. 设置 OpenAI API 密钥
```bash
export OPENAI_API_KEY="your_openai_api_key"
```
2. 运行评估
```bash
python -m ragas_examples.rag_eval.evals
```

你已经使用 Ragas 完成了第一次评估！打开 `experiments/experiment_name.csv` 文件即可查看结果。
