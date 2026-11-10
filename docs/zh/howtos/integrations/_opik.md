# Comet Opik

本教程介绍结合 Opik 与 Ragas，监控和评估 RAG 流水线。

主要有两种方式。

1. 使用 Ragas 指标为追踪评分
2. 使用 Ragas 的 `evaluate` 函数评估数据集

<center><img src="https://raw.githubusercontent.com/comet-ml/opik/main/apps/opik-documentation/documentation/static/img/opik-project-dashboard.png" alt="Comet Opik project dashboard screenshot with list of traces and spans" width="600" style="border: 0.5px solid #ddd;"/></center>

## 配置 { #setup }

[Comet](https://www.comet.com/site?utm_medium=docs&utm_source=ragas&utm_campaign=opik) 提供托管 Opik 平台，[创建账户](https://www.comet.com/signup?from=llm&utm_medium=docs&utm_source=ragas&utm_campaign=opik)后即可获取 API 密钥。

> 也可以在本地运行 Opik，详情参阅[安装指南](https://www.comet.com/docs/opik/self-host/self_hosting_opik?utm_medium=docs&utm_source=ragas&utm_campaign=opik/)。


```python
import getpass
import os

os.environ["OPIK_API_KEY"] = getpass.getpass("Opik API Key: ")
os.environ["OPIK_WORKSPACE"] = input(
    "Comet workspace (often the same as your username): "
)
```

本地运行时，设置以下配置。


```python
# import os
# os.environ["OPIK_URL_OVERRIDE"] = "http://localhost:5173/api"
```

## 准备环境 { #preparing-our-environment }

安装所需库、配置 OpenAI API 密钥，并创建 Opik 数据集。


```python
%pip install opik --quiet

import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
```


## 集成 Opik 与 Ragas { #integrating-opik-with-ragas }

### 使用 Ragas 指标为追踪评分 { #using-ragas-metrics-to-score-traces }

Ragas 指标包括 `answer_relevancy`、`answer_similarity`、`answer_correctness`、`context_precision`、`context_recall`、`context_entity_recall` 和 `summarization_score` 等。完整列表见[文档](https://docs.ragas.io/en/latest/references/metrics.html#)。

可以实时计算指标，并记录到 Opik 的 trace 或 span。本例先创建简单 RAG，再用 `answer_relevancy` 评分。

#### 创建 Ragas 指标 { #create-the-ragas-metric }

直接使用指标时，需要使用 `RunConfig` 和 LLM 初始化。本例使用 LangChain，并启用 Opik 追踪器。

先初始化指标。


```python
# Import the metric
# Import some additional dependencies
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy

# Initialize the Ragas metric
llm = LangchainLLMWrapper(ChatOpenAI())
emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=emb)
```

指标异步评分，使用 `asyncio` 运行评分函数，测试示例问题。


```python
# Run this cell first if you are running this in a Jupyter notebook
import nest_asyncio

nest_asyncio.apply()
```


```python
import asyncio

from ragas.dataset_schema import SingleTurnSample
from ragas.integrations.opik import OpikTracer


# Define the scoring function
def compute_metric(metric, row):
    row = SingleTurnSample(**row)

    opik_tracer = OpikTracer()

    async def get_score(opik_tracer, metric, row):
        score = await metric.single_turn_ascore(row, callbacks=[OpikTracer()])
        return score

    # Run the async function using the current event loop
    loop = asyncio.get_event_loop()

    result = loop.run_until_complete(get_score(opik_tracer, metric, row))
    return result


# Score a simple example
row = {
    "user_input": "What is the capital of France?",
    "response": "Paris",
    "retrieved_contexts": ["Paris is the capital of France.", "Paris is in France."],
}

score = compute_metric(answer_relevancy_metric, row)
print("Answer Relevancy score:", score)
```

    Answer Relevancy score: 1.0


进入 Opik，可以看到 Default Project 下的新追踪。

#### 为追踪评分 { #score-traces }

使用 `update_current_trace`，将反馈分数写入当前追踪。

评分 span 会加入追踪，便于细粒度分析。但本例同步等待 Ragas 评分完成，生产场景需要考虑由此增加的延迟。


```python
from opik import track
from opik.opik_context import update_current_trace


@track
def retrieve_contexts(question):
    # Define the retrieval function, in this case we will hard code the contexts
    return ["Paris is the capital of France.", "Paris is in France."]


@track
def answer_question(question, contexts):
    # Define the answer function, in this case we will hard code the answer
    return "Paris"


@track(name="Compute Ragas metric score", capture_input=False)
def compute_rag_score(answer_relevancy_metric, question, answer, contexts):
    # Define the score function
    row = {"user_input": question, "response": answer, "retrieved_contexts": contexts}
    score = compute_metric(answer_relevancy_metric, row)
    return score


@track
def rag_pipeline(question):
    # Define the pipeline
    contexts = retrieve_contexts(question)
    answer = answer_question(question, contexts)

    score = compute_rag_score(answer_relevancy_metric, question, answer, contexts)
    update_current_trace(
        feedback_scores=[{"name": "answer_relevancy", "value": round(score, 4)}]
    )

    return answer


rag_pipeline("What is the capital of France?")
```




    'Paris'



from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

fiqa_eval = load_dataset("vibrantlabsai/fiqa", "ragas_eval")

# Reformat the dataset to match the schema expected by the Ragas evaluate function
dataset = fiqa_eval["baseline"].select(range(3))

dataset = dataset.map(
    lambda x: {
        "user_input": x["question"],
        "reference": x["ground_truth"],
        "retrieved_contexts": x["contexts"],
    }
)

opik_tracer_eval = OpikTracer(tags=["ragas_eval"], metadata={"evaluation_run": True})

result = evaluate(
    dataset,
    metrics=[context_precision, faithfulness, answer_relevancy],
    callbacks=[opik_tracer_eval],
)

print(result)


```python
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

fiqa_eval = load_dataset("vibrantlabsai/fiqa", "ragas_eval")

# Reformat the dataset to match the schema expected by the Ragas evaluate function
dataset = fiqa_eval["baseline"].select(range(3))

dataset = dataset.map(
    lambda x: {
        "user_input": x["question"],
        "reference": x["ground_truth"],
        "retrieved_contexts": x["contexts"],
    }
)

opik_tracer_eval = OpikTracer(tags=["ragas_eval"], metadata={"evaluation_run": True})

result = evaluate(
    dataset,
    metrics=[context_precision, faithfulness, answer_relevancy],
    callbacks=[opik_tracer_eval],
)

print(result)
```


    Evaluating:   0%|          | 0/6 [00:00<?, ?it/s]


    {'context_precision': 1.0000, 'faithfulness': 0.7375, 'answer_relevancy': 0.9889}

