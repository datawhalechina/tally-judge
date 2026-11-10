# Zeno
## 使用 Zeno 可视化 Ragas 结果 { #visualizing-ragas-results-with-zeno }

[Zeno](https://zenoml.com) 评估平台可以可视化并探索 Ragas 评估结果。

> 可以[查看本教程的最终效果](https://hub.zenoml.com/project/b35c83b8-0b22-4b9c-aedb-80964011d7a7/ragas%20FICA%20eval)。

先安装 `zeno-client`。

```bash
pip install zeno-client
```

在 [hub.zenoml.com](https://hub.zenoml.com) 创建账户，并在[账户页](https://hub.zenoml.com/account)生成 API 密钥。

接着继续[入门指南](../../getstarted/evals.md)中的评估。


```python
import os

import pandas as pd
from datasets import load_dataset
from zeno_client import ZenoClient, ZenoMetric

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
```


```python
# Set API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["ZENO_API_KEY"] = "your-zeno-api-key"
```


```python
fiqa_eval = load_dataset("vibrantlabsai/fiqa", "ragas_eval")
result = evaluate(
    fiqa_eval["baseline"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

df = result.to_pandas()
df.head()
```

将包含数据与结果的 `df` 上传到 Zeno。

先创建项目，指定自定义 RAG 视图与需要分析的指标列。


```python
client = ZenoClient(os.environ["ZENO_API_KEY"])

project = client.create_project(
    name="Ragas FICA eval",
    description="Evaluation of RAG model using Ragas on the FICA dataset",
    view={
        "data": {
            "type": "vstack",
            "keys": {
                "question": {"type": "markdown"},
                "texts": {
                    "type": "list",
                    "elements": {"type": "markdown"},
                    "border": True,
                    "pad": True,
                },
            },
        },
        "label": {
            "type": "markdown",
        },
        "output": {
            "type": "vstack",
            "keys": {
                "answer": {"type": "markdown"},
                "ground_truth": {
                    "type": "list",
                    "elements": {"type": "markdown"},
                    "border": True,
                    "pad": True,
                },
            },
        },
        "size": "large",
    },
    metrics=[
        ZenoMetric(
            name="context_precision", type="mean", columns=["context_precision"]
        ),
        ZenoMetric(name="faithfulness", type="mean", columns=["faithfulness"]),
        ZenoMetric(name="answer_relevancy", type="mean", columns=["answer_relevancy"]),
        ZenoMetric(name="context_recall", type="mean", columns=["context_recall"]),
    ],
)
```

上传包含问题与标准答案的基础数据集。


```python
data_df = pd.DataFrame(
    {
        "data": df.apply(
            lambda x: {"question": x["question"], "texts": list(x["contexts"])}, axis=1
        ),
        "label": df["ground_truth"].apply(lambda x: "\n".join(x)),
    }
)
data_df["id"] = data_df.index

project.upload_dataset(
    data_df, id_column="id", data_column="data", label_column="label"
)
```

最后上传 RAG 输出与 Ragas 指标。

比较与迭代时，可以为多个模型重复这一过程。


```python
output_df = df[
    [
        "context_precision",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
    ]
].copy()

output_df["output"] = df.apply(
    lambda x: {"answer": x["answer"], "ground_truth": list(x["ground_truth"])}, axis=1
)
output_df["id"] = output_df.index

project.upload_system(
    output_df, name="Base System", id_column="id", output_column="output"
)
```

如有问题，可通过 [Discord](https://discord.gg/km62pDKAkE) 或 [hello@zenoml.com](mailto:hello@zenoml.com) 联系 Zeno 团队。
