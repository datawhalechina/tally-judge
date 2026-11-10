# 入门，结合 Vertex AI 使用 Ragas { #getting-started-ragas-with-vertex-ai }

本教程属于结合 Vertex AI 模型使用 Ragas 的三篇系列教程，介绍基础配置。其余两篇可以按任意顺序阅读。

- [对齐 LLM 指标](vertexai_alignment.md)，训练并对齐评估模型，使其更贴近人工判断。
- [模型比较](vertexai_model_comparision.md)，使用 Ragas 指标比较 Vertex AI 模型在 RAG 问答任务上的表现。

下面开始。

## 概览 { #overview }

本教程演示如何使用 Vertex AI Studio 的生成模型，开始通过 Ragas 评估生成式 AI 应用。

**Ragas** 提供一套评估工具和指标，帮助开发者系统化评估并优化 LLM 应用。

本教程涵盖以下内容。

1. 为 Ragas 评估准备数据
2. 了解 Ragas 提供的不同类型指标

更多场景与进阶功能，请查阅以下文档。

- [Ragas 核心概念](../../concepts/index.md)
- [Ragas 操作指南](../index.md)

## 开始使用 { #getting-started }

## 安装依赖 { #install-dependencies }


```python
!pip install --upgrade --user --quiet langchain-core langchain-google-vertexai langchain ragas rouge_score
```

### 重启运行时 { #restart-runtime }

要在当前 Jupyter 环境使用新安装的软件包，需要重启运行时。执行下面的单元格即可重启内核。

重启可能需要一分钟或更长时间，完成后继续下一步。


```python
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### 验证 notebook 环境（仅 Colab） { #authenticate-your-notebook-environment-colab-only }

如果在 Google Colab 运行，请执行以下单元格进行身份验证。


```python
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### 设置 Google Cloud 项目信息并初始化 Vertex AI SDK { #set-google-cloud-project-information-and-initialize-vertex-ai-sdk }


```python
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## 使用 Ragas 指标配置评估 { #set-up-eval-using-ragas-metrics }

**Ragas** 是开源 AI 应用评估库，支持基于计算的指标、基于模型的指标以及自定义指标。完整列表请参阅[可用指标文档](../../concepts/metrics/available_metrics/index.md)。

下面介绍不同类型指标的用法。

- **自定义指标**，定义最符合应用需求的评判标准。
- **基于模型的指标**，通过 LLM 调用，按具体标准评价输出，支持有参考和无参考场景。
- **基于计算的指标**，按数学公式计算，无须调用 LLM。


### 定义 evaluator_llm { #define-evaluator_llm }

使用基于模型的指标前，需要配置评估 LLM 与嵌入模型。


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


evaluator_llm = LangchainLLMWrapper(VertexAI(model_name="gemini-pro"))
evaluator_embeddings = LangchainEmbeddingsWrapper(VertexAIEmbeddings(model_name="text-embedding-004"))
```

### 创建基于模型的逐样本指标 { #build-your-own-model-based-point-wise-metrics-using-ragas }

自定义指标可以通过评分规则、细则或自定义逻辑，引导 LLM 按具体任务要求评估回答。详情参阅[通用指标](../../concepts/metrics/available_metrics/general_purpose.md)。


```python
from ragas import evaluate
from ragas.metrics import AspectCritic, RubricsScore

maliciousness_definition = (
    "Is the response intended to harm, deceive, or exploit users?"
)

aspect_critic = AspectCritic(
    name="maliciousness",
    definition=maliciousness_definition,
    llm=evaluator_llm,
)

# adapeted google's helpfulness_prompt_template
helpfulness_rubrics = {
    "score1_description": "Response is useless/irrelevant, contains inaccurate/deceptive/misleading information, and/or contains harmful/offensive content. The user would feel not at all satisfied with the content in the response.",
    "score2_description": "Response is minimally relevant to the instruction and may provide some vaguely useful information, but it lacks clarity and detail. It might contain minor inaccuracies. The user would feel only slightly satisfied with the content in the response.",
    "score3_description": "Response is relevant to the instruction and provides some useful content, but could be more relevant, well-defined, comprehensive, and/or detailed. The user would feel somewhat satisfied with the content in the response.",
    "score4_description": "Response is very relevant to the instruction, providing clearly defined information that addresses the instruction's core needs.  It may include additional insights that go slightly beyond the immediate instruction.  The user would feel quite satisfied with the content in the response.",
    "score5_description": "Response is useful and very comprehensive with well-defined key details to address the needs in the instruction and usually beyond what explicitly asked. The user would feel very satisfied with the content in the response.",
}

rubrics_score = RubricsScore(name="helpfulness", rubrics=helpfulness_rubrics, llm=evaluator_llm)
```

### Ragas 基于模型的指标 { #ragas-model-based-metrics }

这类指标调用预训练语言模型，对照具体标准分析生成文本，结合上下文作出更接近人工的细致判断。详情参阅[指标文档](../../concepts/metrics/available_metrics/index.md)。


```python
from ragas import evaluate
from ragas.metrics import ContextPrecision, Faithfulness

context_precision = ContextPrecision(llm=evaluator_llm)
faithfulness = Faithfulness(llm=evaluator_llm)
```

### Ragas 基于计算的指标 { #ragas-computation-based-metrics }

这类指标使用字符串匹配、n-gram 和统计方法，完全通过计算衡量文本相似性与质量。详情参阅[传统指标](../../concepts/metrics/available_metrics/traditional.md)。


```python
from ragas.metrics import RougeScore

rouge_score = RougeScore()
```

## 准备数据集 { #prepare-your-dataset }

使用 Ragas 评估前，需要将数据转换为 `EvaluationDataset`，详情参阅[评估数据集](../../concepts/components/eval_dataset.md)。

以下列样本为例。


```python
# questions or query from user
user_inputs = [
    "Which part of the brain does short-term memory seem to rely on?",
    "What provided the Roman senate with exuberance?",
    "What area did the Hasan-jalalians command?",
]

# retrieved data used in answer generation
retrieved_contexts = [
    ["Short-term memory is supported by transient patterns of neuronal communication, dependent on regions of the frontal lobe (especially dorsolateral prefrontal cortex) and the parietal lobe. Long-term memory, on the other hand, is maintained by more stable and permanent changes in neural connections widely spread throughout the brain. The hippocampus is essential (for learning new information) to the consolidation of information from short-term to long-term memory, although it does not seem to store information itself. Without the hippocampus, new memories are unable to be stored into long-term memory, as learned from patient Henry Molaison after removal of both his hippocampi, and there will be a very short attention span. Furthermore, it may be involved in changing neural connections for a period of three months or more after the initial learning."],
    ["In 62 BC, Pompey returned victorious from Asia. The Senate, elated by its successes against Catiline, refused to ratify the arrangements that Pompey had made. Pompey, in effect, became powerless. Thus, when Julius Caesar returned from a governorship in Spain in 61 BC, he found it easy to make an arrangement with Pompey. Caesar and Pompey, along with Crassus, established a private agreement, now known as the First Triumvirate. Under the agreement, Pompey's arrangements would be ratified. Caesar would be elected consul in 59 BC, and would then serve as governor of Gaul for five years. Crassus was promised a future consulship."],
    ["The Seljuk Empire soon started to collapse. In the early 12th century, Armenian princes of the Zakarid noble family drove out the Seljuk Turks and established a semi-independent Armenian principality in Northern and Eastern Armenia, known as Zakarid Armenia, which lasted under the patronage of the Georgian Kingdom. The noble family of Orbelians shared control with the Zakarids in various parts of the country, especially in Syunik and Vayots Dzor, while the Armenian family of Hasan-Jalalians controlled provinces of Artsakh and Utik as the Kingdom of Artsakh."],
]

# answers generated by the rag
responses = [
    "frontal lobe and the parietal lobe",
    "The Roman Senate was filled with exuberance due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Syunik and Vayots Dzor.",
]

# expected responses or ground truth
references = [
    "frontal lobe and the parietal lobe",
    "Due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Artsakh and Utik.",
]
```

将其转换为 Ragas 的 EvaluationDataset。


```python
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

n = len(user_inputs)
samples = []


for i in range(n):

    sample = SingleTurnSample(
        user_input=user_inputs[i],
        retrieved_contexts=retrieved_contexts[i],
        response=responses[i],
        reference=references[i],
    )
    samples.append(sample)


ragas_eval_dataset = EvaluationDataset(samples=samples)
ragas_eval_dataset.to_pandas()
```
输出如下。
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border=1>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Which part of the brain does short-term memory...</td>
      <td>[Short-term memory is supported by transient p...</td>
      <td>frontal lobe and the parietal lobe</td>
      <td>frontal lobe and the parietal lobe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What provided the Roman senate with exuberance?</td>
      <td>[In 62 BC, Pompey returned victorious from Asi...</td>
      <td>The Roman Senate was filled with exuberance du...</td>
      <td>Due to successes against Catiline.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What area did the Hasan-jalalians command?</td>
      <td>[The Seljuk Empire soon started to collapse. I...</td>
      <td>The Hasan-Jalalians commanded the area of Syun...</td>
      <td>The Hasan-Jalalians commanded the area of Arts...</td>
    </tr>
  </tbody>
</table>
</div>

## 运行评估 { #run-evaluation }

定义好数据集与指标后，将它们传给 Ragas 的 `evaluate` 函数。

```python
from ragas import evaluate

ragas_metrics = [aspect_critic, context_precision, faithfulness, rouge_score, rubrics_score]

result = evaluate(
    metrics=ragas_metrics,
    dataset=ragas_eval_dataset
)
result
```
```
Evaluating: 100%|██████████| 15/15 [00:00<?, ?it/s]
```

查看数据集中每行的详细得分。


```python
result.to_pandas()
```


输出如下。
<div>
<table border=1>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>maliciousness</th>
      <th>context_precision</th>
      <th>faithfulness</th>
      <th>rouge_score(mode=fmeasure)</th>
      <th>helpfulness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Which part of the brain does short-term memory...</td>
      <td>[Short-term memory is supported by transient p...</td>
      <td>frontal lobe and the parietal lobe</td>
      <td>frontal lobe and the parietal lobe</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What provided the Roman senate with exuberance?</td>
      <td>[In 62 BC, Pompey returned victorious from Asi...</td>
      <td>The Roman Senate was filled with exuberance du...</td>
      <td>Due to successes against Catiline.</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.588235</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What area did the Hasan-jalalians command?</td>
      <td>[The Seljuk Empire soon started to collapse. I...</td>
      <td>The Hasan-Jalalians commanded the area of Syun...</td>
      <td>The Hasan-Jalalians commanded the area of Arts...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.761905</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

继续阅读本系列其他教程。

- [对齐 LLM 指标](vertexai_alignment.md)，训练并对齐评估模型，使其更贴近人工判断。
- [模型比较](vertexai_model_comparision.md)，使用 Ragas 指标比较 Vertex AI 模型在 RAG 问答任务上的表现。
