# Helicone

本教程介绍结合 Helicone 与 Ragas，监控和评估 RAG 系统。

## 前置条件 { #prerequisites }

开始前，需要 Helicone 账户与 API 密钥。

1. 登录 [Helicone](https://www.helicone.ai)，或创建账户。
2. 在[开发者页面](https://helicone.ai/developer)生成 API 密钥。

**注意**。这里生成只写 API 密钥，身份验证详情参阅 [Helicone 文档](https://docs.helicone.ai/getting-started/helicone-api-keys)。

妥善保存密钥，后续集成时使用。

## 配置 { #setup }

先安装依赖并配置环境。


```python
!pip install datasets ragas openai
```


```python
import os

from datasets import Dataset

from ragas import evaluate
from ragas.integrations.helicone import helicone_config  # import helicone_config
from ragas.metrics import answer_relevancy, context_precision, faithfulness

# Set up Helicone
HELICONE_API_KEY = (
    "your_helicone_api_key_here"  # Replace with your actual Helicone API key
)
helicone_config.api_key = HELICONE_API_KEY
os.environ["OPENAI_API_KEY"] = (
    "your_openai_api_key_here"  # Replace with your actual OpenAI API key
)

# Verify Helicone API key is set
if HELICONE_API_KEY == "your_helicone_api_key_here":
    raise ValueError(
        "Please replace 'your_helicone_api_key_here' with your actual Helicone API key."
    )
```

## 准备数据 { #prepare-data }

准备评估 RAG 系统的示例数据。


```python
data_samples = {
    "question": ["When was the first Super Bowl?", "Who has won the most Super Bowls?"],
    "answer": [
        "The first Super Bowl was held on January 15, 1967.",
        "The New England Patriots have won the most Super Bowls, with six championships.",
    ],
    "contexts": [
        [
            "The First AFL–NFL World Championship Game, later known as Super Bowl I, was played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles, California."
        ],
        [
            "As of 2021, the New England Patriots have won the most Super Bowls with six championships, all under the leadership of quarterback Tom Brady and head coach Bill Belichick."
        ],
    ],
    "ground_truth": [
        "The first Super Bowl was held on January 15, 1967.",
        "The New England Patriots have won the most Super Bowls, with six championships as of 2021.",
    ],
}

dataset = Dataset.from_dict(data_samples)
print(dataset)
```

## 使用 Ragas 评估 { #evaluate-with-ragas }

运行评估，Helicone 将自动记录过程中的 API 调用。


```python
# Evaluate using Ragas
score = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

# Display results
print(score.to_pandas())
```

## 在 Helicone 中查看结果 { #viewing-results-in-helicone }

评估期间的 API 调用会自动记录到 Helicone，可在控制台分析系统表现与行为。

查看步骤如下。
1. 进入 [Helicone 控制台](https://www.helicone.ai/dashboard)。
2. 打开 Requests 页面。
3. 查看 Ragas 评估产生的调用。

日志可以帮助分析以下内容。
- 评估中的 API 调用数量
- 每次调用的延迟、token 用量等表现
- 评估期间发生的错误与问题

这一集成结合 Ragas 的评估能力与 Helicone 的监控分析能力。
