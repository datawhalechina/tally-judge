# Langfuse

结合 Ragas 与 Langfuse，可以评估并监控检索增强生成（RAG）流水线。

## 什么是 Langfuse？ { #what-is-langfuse }

Langfuse（[GitHub](https://github.com/langfuse/langfuse)）是支持 LLM [追踪](https://langfuse.com/docs/tracing)、[提示词管理](https://langfuse.com/docs/prompts/get-started)和[评估](https://langfuse.com/docs/scores/overview)的开源平台。它可以为 trace 与 span 评分，分析 RAG 表现，并支持 [OpenAI](https://langfuse.com/docs/integrations/openai/python/get-started)、[LangChain](https://langfuse.com/docs/integrations/langchain/tracing) 等[多种集成](https://langfuse.com/docs/integrations/overview)。

## 结合 Langfuse 与 Ragas 的收益 { #key-benefits-of-using-langfuse-with-ragas }

- **追踪评分**，为 trace 与 span [添加分数](https://langfuse.com/docs/scores/overview)，了解 RAG 表现。
- **详细分析**，对追踪[分组与分析](https://langfuse.com/docs/analytics/overview)，发现低质量结果。
- **评分报告**，按具体场景与用户群体深入查看。

Ragas（[GitHub](https://github.com/vibrantlabsai/ragas)）可对 trace/span 进行[基于模型的评估](https://langfuse.com/docs/scores/model-based-evals)，尤其适合 RAG。无参考指标可以直接用于 Langfuse 收集的生产追踪，无须标准答案。

## 开始使用 { #getting-started }

本指南介绍结合 Ragas 与 Langfuse 的端到端 RAG 评估。

### 环境 { #the-environment }

[注册 Langfuse](https://cloud.langfuse.com)，获取 API 密钥。


```python
import os

# get keys for your project from https://cloud.langfuse.com
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."

# your openai key
# os.environ["OPENAI_API_KEY"] = "sk-..."
```


```python
%pip install datasets ragas llama_index python-dotenv --upgrade
```

### 数据 { #the-data }

本例使用已通过 RAG 查询并收集输出的数据集。下文也介绍如何从 Langfuse 获取生产数据。

数据集包含以下列。
- `question`，*list[str]*，用于评估的问题
- `answer`，*list[str]*，RAG 生成并提供给用户的答案
- `contexts`，*list[list[str]]*，生成答案时传给 LLM 的上下文
- `ground_truth`，list[list[str]]，问题的标准答案，在线评估中可能无法获取，可以省略


```python
from datasets import load_dataset

amnesty_qa = load_dataset("vibrantlabsai/amnesty_qa", "english_v2")["eval"]
amnesty_qa
```

    Found cached dataset amnesty_qa (/home/jjmachan/.cache/huggingface/datasets/vibrantlabs___amnesty_qa/english_v2/2.0.0/d0ed9800191a31943ee52a5c22ee4305e28a33f5edcd9a323802112cff07cc24)



      0%|          | 0/1 [00:00<?, ?it/s]





    Dataset({
        features: ['question', 'ground_truth', 'answer', 'contexts'],
        num_rows: 20
    })



### 指标 { #the-metrics }
本例使用以下 Ragas 指标。

- [`faithfulness`](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html)，衡量答案与给定上下文在事实层面的一致性。
- [`answer_relevancy`](https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html)，衡量答案是否切题。
- [`context precision`](https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html)，衡量相关片段是否排在检索结果前列，理想情况下相关内容应优先出现。该示例结合问题与上下文计算，得分为 0～1，越高越好。
- [`aspect_critique`](https://docs.ragas.io/en/latest/concepts/metrics/critique.html)，按无害性、正确性等预定义或自定义维度评估。

详情参阅[指标文档](https://docs.ragas.io/en/latest/concepts/metrics/index.html)。


```python
# import metrics
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.metrics.critique import SUPPORTED_ASPECTS, harmfulness

# metrics you chose
metrics = [faithfulness, answer_relevancy, context_precision, harmfulness]
```

使用所选 LLM 与嵌入模型初始化指标，本例使用 OpenAI。


```python
from ragas.run_config import RunConfig
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings


# util function to init Ragas Metrics
def init_ragas_metrics(metrics, llm, embedding):
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)
```


```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

llm = ChatOpenAI()
emb = OpenAIEmbeddings()

init_ragas_metrics(
    metrics,
    llm=LangchainLLMWrapper(llm),
    embedding=LangchainEmbeddingsWrapper(emb),
)
```

### 配置 { #the-setup }
基于模型的评估可按两种方式运行。

1. **逐条评分**。评估每个 trace，详细了解每次 RAG 调用，但成本较高。
2. **批量评分**。定期随机抽样追踪并评分，降低费用，但可能遗漏重要样本。

下面分别介绍。

### 为追踪评分 { #score-the-trace }

先加载一条追踪数据，演示 Ragas 评分。


```python
row = amnesty_qa[0]
print("question: ", row["question"])
print("answer: ", row["answer"])
```

    question:  What are the global implications of the USA Supreme Court ruling on abortion?
    answer:  The global implications of the USA Supreme Court ruling on abortion can be significant, as it sets a precedent for other countries and influences the global discourse on reproductive rights. Here are some potential implications:

    1. Influence on other countries: The Supreme Court's ruling can serve as a reference point for other countries grappling with their own abortion laws. It can provide legal arguments and reasoning that advocates for reproductive rights can use to challenge restrictive abortion laws in their respective jurisdictions.

    2. Strengthening of global reproductive rights movements: A favorable ruling by the Supreme Court can energize and empower reproductive rights movements worldwide. It can serve as a rallying point for activists and organizations advocating for women's rights, leading to increased mobilization and advocacy efforts globally.

    3. Counteracting anti-abortion movements: Conversely, a ruling that restricts abortion rights can embolden anti-abortion movements globally. It can provide legitimacy to their arguments and encourage similar restrictive measures in other countries, potentially leading to a rollback of existing reproductive rights.

    4. Impact on international aid and policies: The Supreme Court's ruling can influence international aid and policies related to reproductive health. It can shape the priorities and funding decisions of donor countries and organizations, potentially leading to increased support for reproductive rights initiatives or conversely, restrictions on funding for abortion-related services.

    5. Shaping international human rights standards: The ruling can contribute to the development of international human rights standards regarding reproductive rights. It can influence the interpretation and application of existing human rights treaties and conventions, potentially strengthening the recognition of reproductive rights as fundamental human rights globally.

    6. Global health implications: The Supreme Court's ruling can have implications for global health outcomes, particularly in countries with restrictive abortion laws. It can impact the availability and accessibility of safe and legal abortion services, potentially leading to an increase in unsafe abortions and related health complications.

    It is important to note that the specific implications will depend on the nature of the Supreme Court ruling and the subsequent actions taken by governments, activists, and organizations both within and outside the United States.


初始化 Langfuse 客户端 SDK，为应用添加埋点。


```python
from langfuse import Langfuse

langfuse = Langfuse()
```

定义辅助函数，使用选定指标为追踪评分。


```python
async def score_with_ragas(query, chunks, answer):
    scores = {}
    for m in metrics:
        print(f"calculating {m.name}")
        scores[m.name] = await m.ascore(
            row={"question": query, "contexts": chunks, "answer": answer}
        )
    return scores
```


```python
question, contexts, answer = row["question"], row["contexts"], row["answer"]
await score_with_ragas(question, contexts, answer)
```

    calculating faithfulness
    calculating answer_relevancy


    Using 'context_precision' without ground truth will be soon depreciated. Use 'context_utilization' instead


    calculating context_precision
    calculating harmfulness





    {'faithfulness': 0.0,
     'answer_relevancy': 0.9999999999999996,
     'context_precision': 0.9999999999,
     'harmfulness': 0}



下面的简化应用在每次请求时计算分数，依次执行以下步骤。

1. 接收用户问题
2. 从数据库或向量存储获取可用于回答的上下文
3. 将问题与上下文传给 LLM 生成答案

这些步骤作为 span 记录在同一条 Langfuse trace 中，详情参阅[追踪文档](https://langfuse.com/docs/tracing)。


```python
# the logic of the dummy application is
# given a question fetch the correspoinding contexts and answers from a dict

import hashlib


def hash_string(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()


q_to_c = {}  # map between question and context
q_to_a = {}  # map between question and answer
for row in amnesty_qa:
    q_hash = hash_string(row["question"])
    q_to_c[q_hash] = row["contexts"]
    q_to_a[q_hash] = row["answer"]
```


```python
# if your running this in a notebook - please run this cell
# to manage asyncio event loops
import nest_asyncio

nest_asyncio.apply()
```


```python
from langfuse.decorators import observe, langfuse_context
from asyncio import run


@observe()
def retriver(question: str):
    return q_to_c[question]


@observe()
def generator(question):
    return q_to_a[question]


@observe()
def rag_pipeline(question):
    q_hash = hash_string(question)
    contexts = retriver(q_hash)
    generated_answer = generator(q_hash)

    # score the runs
    score = run(score_with_ragas(question, contexts, answer=generated_answer))
    for s in score:
        langfuse_context.score_current_trace(name=s, value=score[s])
    return generated_answer
```


```python
question, contexts, answer = row["question"], row["contexts"], row["answer"]
generated_answer = rag_pipeline(amnesty_qa[0]["question"])
```

    calculating faithfulness
    calculating answer_relevancy


    Using 'context_precision' without ground truth will be soon depreciated. Use 'context_utilization' instead


    calculating context_precision
    calculating harmfulness


### 在 Langfuse 中分析得分 { #analyze-the-scores-in-langfuse }

在 Langfuse 界面[分析得分](https://langfuse.com/docs/analytics/overview)，查看每个问题或用户的结果。

→ 可以通过[交互演示](https://langfuse.com/docs/demo)体验控制台。

![带 Ragas 得分的追踪](https://langfuse.com/images/docs/ragas-trace-score.png)

评分是阻塞操作，应先把答案发送给用户，再等待评分完成。也可以在独立线程运行 `score_with_ragas()`，传入 `trace_id` 记录得分。

## 参考资料 { #resources }

- 参阅[基于模型的评估](https://langfuse.com/docs/scores/model-based-evals)，了解更多 Ragas 用法。
- [进一步了解 LLM 应用分析与改进](https://langfuse.com/faq/all/llm-analytics-101)。

## 反馈 { #feedback }

如有建议或需求，请提交 GitHub [issue](https://langfuse.com/issue)，或在 [Discord](https://discord.langfuse.com/) 与社区交流。




