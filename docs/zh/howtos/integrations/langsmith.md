# LangSmith

[LangSmith](https://docs.smith.langchain.com/) 支持 LLM 应用的开发与部署，提供追踪、分析和优化工具，便于管理复杂交互。

本教程介绍如何用 LangSmith 记录 Ragas 评估追踪。本例 Ragas 使用 LangChain，配置 LangSmith 后即可自动记录。

## 配置 LangSmith { #setting-up-langsmith }

设置以下环境变量，详情参阅 [LangSmith 文档](https://docs.smith.langchain.com/#quick-start)。

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # Defaults to "default" if not set
```

## 获取数据集 { #getting-the-dataset }

创建评估数据集或评估单条样本时，字段术语应与 `SingleTurnSample` 或 `MultiTurnSample` 的结构一致。


```python
from ragas import EvaluationDataset


dataset = [
    {
        "user_input": "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
        "retrieved_contexts": [
            "Andrew Ng, CEO of Landing AI, is known for his pioneering work in deep learning and for democratizing AI education through Coursera."
        ],
        "response": "Andrew Ng is widely recognized for democratizing AI education through platforms like Coursera.",
        "reference": "Andrew Ng, CEO of Landing AI, is known for democratizing AI education through Coursera.",
    },
    {
        "user_input": "Who is Sam Altman?",
        "retrieved_contexts": [
            "Sam Altman, CEO of OpenAI, has advanced AI research and advocates for safe, beneficial AI technologies."
        ],
        "response": "Sam Altman is the CEO of OpenAI and advocates for safe, beneficial AI technologies.",
        "reference": "Sam Altman, CEO of OpenAI, has advanced AI research and advocates for safe AI.",
    },
    {
        "user_input": "Who is Demis Hassabis and how did he gain prominence?",
        "retrieved_contexts": [
            "Demis Hassabis, CEO of DeepMind, is known for developing systems like AlphaGo that master complex games."
        ],
        "response": "Demis Hassabis is the CEO of DeepMind, known for developing systems like AlphaGo.",
        "reference": "Demis Hassabis, CEO of DeepMind, is known for developing AlphaGo.",
    },
    {
        "user_input": "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
        "retrieved_contexts": [
            "Sundar Pichai, CEO of Google and Alphabet Inc., leads innovation across Google's product ecosystem."
        ],
        "response": "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem.",
        "reference": "Sundar Pichai, CEO of Google and Alphabet Inc., leads innovation across Google's product ecosystem.",
    },
    {
        "user_input": "How did Arvind Krishna transform IBM?",
        "retrieved_contexts": [
            "Arvind Krishna, CEO of IBM, transformed the company by focusing on cloud computing and AI solutions."
        ],
        "response": "Arvind Krishna transformed IBM by focusing on cloud computing and AI solutions.",
        "reference": "Arvind Krishna, CEO of IBM, transformed the company through cloud computing and AI.",
    },
]

evaluation_dataset = EvaluationDataset.from_list(dataset)
```

## 追踪 Ragas 指标 { #tracing-ragas-metrics }

运行数据集评估后，追踪会出现在 LangSmith 中指定项目或 default 项目下。


```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)

result
```

输出如下。
```
Evaluating:   0%|          | 0/15 [00:00<?, ?it/s]

{'context_recall': 1.0000, 'faithfulness': 0.9333, 'factual_correctness': 0.8520}
```

## LangSmith 控制台 { #langsmith-dashboard }
![LangSmith 控制台](../../../_static/langsmith_dashboard.png)
