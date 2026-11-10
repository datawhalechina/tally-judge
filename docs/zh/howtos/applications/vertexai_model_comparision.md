# 使用 Ragas 比较 Vertex AI 模型的 RAG 问答表现 { #compare-models-provided-by-vertexai-on-rag-based-qa-task-using-ragas-metrics }

本教程属于结合 Vertex AI 使用 Ragas 的三篇系列教程。建议先阅读[入门教程](vertexai_x_ragas.md)，也可以直接操作。另一篇为[对齐 LLM 指标](vertexai_alignment.md)。

## 概览 { #overview }

本教程使用 Ragas 为不同 LLM 的**问答任务**输出评分，再通过可视化比较结果，选择生成模型。

## 开始使用 { #getting-started }

### 安装依赖 { #install-dependencies }


```python
%pip install --upgrade --user --quiet langchain-core langchain-google-vertexai langchain ragas rouge_score
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

### 辅助函数 { #helper-functions }

以下辅助函数用于展示评估报告与可视化结果。


```python
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, Markdown, display


def display_eval_report(eval_result, metrics=None):
    """Display the evaluation results."""

    title, summary_metrics, report_df = eval_result
    metrics_df = pd.DataFrame.from_dict(summary_metrics, orient="index").T
    if metrics:
        metrics_df = metrics_df.filter(
            [
                metric
                for metric in metrics_df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )
        report_df = report_df.filter(
            [
                metric
                for metric in report_df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )

    # Display the title with Markdown for emphasis
    display(Markdown(f"## {title}"))

    # Display the metrics DataFrame
    display(Markdown("### Summary Metrics"))
    display(metrics_df)

    # Display the detailed report DataFrame
    display(Markdown("### Report Metrics"))
    display(report_df)


def plot_radar_plot(eval_results, max_score=5, metrics=None):
    fig = go.Figure()

    for eval_result in eval_results:
        title, summary_metrics, report_df = eval_result

        if metrics:
            summary_metrics = {
                k: summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric in k for selected_metric in metrics)
            }

        fig.add_trace(
            go.Scatterpolar(
                r=list(summary_metrics.values()),
                theta=list(summary_metrics.keys()),
                fill="toself",
                name=title,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max_score])), showlegend=True
    )

    fig.show()


def plot_bar_plot(eval_results, metrics=None):
    fig = go.Figure()
    data = []

    for eval_result in eval_results:
        title, summary_metrics, _ = eval_result
        if metrics:
            summary_metrics = {
                k: summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric in k for selected_metric in metrics)
            }

        data.append(
            go.Bar(
                x=list(summary_metrics.keys()),
                y=list(summary_metrics.values()),
                name=title,
            )
        )

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(barmode="group")
    fig.show()
```

## 使用 Ragas 指标配置评估 { #set-up-eval-using-ragas-metrics }

### 定义 evaluator_llm { #define-evaluator_llm }

先定义评估 LLM 和嵌入模型，以使用基于模型的指标。


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


evaluator_llm = LangchainLLMWrapper(VertexAI(model_name="gemini-pro"))
evaluator_embeddings = LangchainEmbeddingsWrapper(VertexAIEmbeddings(model_name="text-embedding-004"))
```

### **Ragas 指标** { #ragas-metrics }

选择并定义与应用最相关的指标。


```python
from ragas import evaluate
from ragas.metrics import ContextPrecision, Faithfulness, RubricsScore, RougeScore

rouge_score = RougeScore()

helpfulness_rubrics = {
    "score1_description": "Response is useless/irrelevant, contains inaccurate/deceptive/misleading information, and/or contains harmful/offensive content. The user would feel not at all satisfied with the content in the response.",
    "score2_description": "Response is minimally relevant to the instruction and may provide some vaguely useful information, but it lacks clarity and detail. It might contain minor inaccuracies. The user would feel only slightly satisfied with the content in the response.",
    "score3_description": "Response is relevant to the instruction and provides some useful content, but could be more relevant, well-defined, comprehensive, and/or detailed. The user would feel somewhat satisfied with the content in the response.",
    "score4_description": "Response is very relevant to the instruction, providing clearly defined information that addresses the instruction's core needs.  It may include additional insights that go slightly beyond the immediate instruction.  The user would feel quite satisfied with the content in the response.",
    "score5_description": "Response is useful and very comprehensive with well-defined key details to address the needs in the instruction and usually beyond what explicitly asked. The user would feel very satisfied with the content in the response.",
}

rubrics_score = RubricsScore(name="helpfulness", rubrics=helpfulness_rubrics)
context_precision = ContextPrecision(llm=evaluator_llm)
faithfulness = Faithfulness(llm=evaluator_llm)
```

## **准备数据集** { #prepare-your-dataset }

使用 Ragas 评估前，将数据转换为核心数据类型 `EvaluationDataset`，详情参阅[评估数据集文档](../../concepts/components/eval_dataset.md)。


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

# expected responses or ground truth
references = [
    "frontal lobe and the parietal lobe",
    "Due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Artsakh and Utik.",
]
```


```python
from vertexai.generative_models import GenerativeModel

generation_config = {
    "max_output_tokens": 128,
    "temperature": 0.1,
}

model_a_name = "gemini-1.5-pro"
model_b_name = "gemini-1.0-pro"

gemini_model_15 = GenerativeModel(
    model_a_name,
    generation_config=generation_config,
)

gemini_model_1 = GenerativeModel(
    model_b_name,
    generation_config=generation_config,
)
```


```python
responses_a = []
responses_b = []

# Template for creating the prompt
template = """Answer the question based only on the following context:
{context}

Question: {query}
"""

# Iterate through each user input and corresponding context
for i in range(len(user_inputs)):
    # Join the list of retrieved contexts into a single string
    context_str = "\n".join(retrieved_contexts[i])

    # Create prompt Generate response for Gemini 1.5 pro model
    gemini_15_prompt = template.format(context=context_str, query=user_inputs[i])

    gemini_15_response = gemini_model_15.generate_content(gemini_15_prompt)
    responses_a.append(gemini_15_response.text)

    # Create prompt Generate response for Gemini 1 pro model
    gemini_1_prompt = template.format(context=context_str, query=user_inputs[i])

    gemini_1_response = gemini_model_1.generate_content(gemini_1_prompt)
    responses_b.append(gemini_1_response.text)
```

将其转换为 Ragas `EvaluationDataset`。


```python
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

n = len(user_inputs)

samples_a = []
samples_b = []

for i in range(n):
    sample_a = SingleTurnSample(
        user_input=user_inputs[i],
        retrieved_contexts=retrieved_contexts[i],
        response=responses_a[i],
        reference=references[i],
    )
    sample_b = SingleTurnSample(
        user_input=user_inputs[i],
        retrieved_contexts=retrieved_contexts[i],
        response=responses_b[i],
        reference=references[i],
    )

    samples_a.append(sample_a)
    samples_b.append(sample_b)

ragas_eval_dataset_a = EvaluationDataset(samples=samples_a)
ragas_eval_dataset_b = EvaluationDataset(samples=samples_b)
```


```python
ragas_eval_dataset_a.to_pandas()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Which part of the brain does short-term memory...</td>
      <td>[Short-term memory is supported by transient p...</td>
      <td>Short-term memory relies on regions of the **f...</td>
      <td>frontal lobe and the parietal lobe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What provided the Roman senate with exuberance?</td>
      <td>[In 62 BC, Pompey returned victorious from Asi...</td>
      <td>The Roman Senate was elated by its successes a...</td>
      <td>Due to successes against Catiline.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What area did the Hasan-jalalians command?</td>
      <td>[The Seljuk Empire soon started to collapse. I...</td>
      <td>The Hasan-Jalalians controlled the provinces o...</td>
      <td>The Hasan-Jalalians commanded the area of Arts...</td>
    </tr>
  </tbody>
</table>
</div>

```python
ragas_eval_dataset_b.to_pandas()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Which part of the brain does short-term memory...</td>
      <td>[Short-term memory is supported by transient p...</td>
      <td>The frontal lobe, especially the dorsolateral ...</td>
      <td>frontal lobe and the parietal lobe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What provided the Roman senate with exuberance?</td>
      <td>[In 62 BC, Pompey returned victorious from Asi...</td>
      <td>The Roman Senate's exuberance stemmed from its...</td>
      <td>Due to successes against Catiline.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What area did the Hasan-jalalians command?</td>
      <td>[The Seljuk Empire soon started to collapse. I...</td>
      <td>The Hasan-Jalalians controlled the provinces o...</td>
      <td>The Hasan-Jalalians commanded the area of Arts...</td>
    </tr>
  </tbody>
</table>
</div>

## 运行评估 { #run-evaluation }

将数据集与所需指标列表传给 `evaluate`。


```python
from ragas import evaluate

ragas_metrics = [
    context_precision,
    faithfulness,
    rouge_score,
    rubrics_score,
]

ragas_result_rag_a = evaluate(
    dataset=ragas_eval_dataset_a, metrics=ragas_metrics, llm=evaluator_llm
)

ragas_result_rag_b = evaluate(
    dataset=ragas_eval_dataset_b, metrics=ragas_metrics, llm=evaluator_llm
)
```
```
Evaluating: 100%|██████████| 12/12 [00:00<?, ?it/s]

Evaluating: 100%|██████████| 12/12 [00:00<?, ?it/s]
```

将结果封装为 Google 的 EvalResult 结构。


```python
from vertexai.evaluation import EvalResult

result_rag_a = EvalResult(
    summary_metrics=ragas_result_rag_a._repr_dict,
    metrics_table=ragas_result_rag_a.to_pandas(),
)

result_rag_b = EvalResult(
    summary_metrics=ragas_result_rag_b._repr_dict,
    metrics_table=ragas_result_rag_b.to_pandas(),
)
```

## 比较评估结果 { #compare-eval-results }

### 查看汇总结果 { #view-summary-results }

调用 `display_eval_report()` 辅助函数，在一张表中查看全部指标的汇总。


```python
display_eval_report(
    eval_result=(
        f"{model_a_name} Eval Result",
        result_rag_a.summary_metrics,
        result_rag_a.metrics_table,
    ),
)
```
输出如下。

## gemini-1.5-pro 评估结果 { #gemini-15-pro-eval-result }

### 指标汇总 { #summary-metrics }
  <div id="summary-metrics">
    <table border=1>
      <thead>
        <tr>
          <th></th>
          <th>context_precision</th>
          <th>faithfulness</th>
          <th>rouge_score(mode=fmeasure)</th>
          <th>helpfulness</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.666667</td>
          <td>1.0</td>
          <td>0.56</td>
          <td>4.333333</td>
        </tr>
      </tbody>
    </table>
  </div>

### 详细指标 { #report-metrics }
  <div id="report-metrics">
    <table border=1>
      <thead>
        <tr>
          <th></th>
          <th>user_input</th>
          <th>retrieved_contexts</th>
          <th>response</th>
          <th>reference</th>
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
          <td>Short-term memory relies on regions of the **f...</td>
          <td>frontal lobe and the parietal lobe</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.48</td>
          <td>5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>What provided the Roman senate with exuberance?</td>
          <td>[In 62 BC, Pompey returned victorious from Asi...</td>
          <td>The Roman Senate was elated by its successes a...</td>
          <td>Due to successes against Catiline.</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.40</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2</th>
          <td>What area did the Hasan-jalalians command?</td>
          <td>[The Seljuk Empire soon started to collapse. I...</td>
          <td>The Hasan-Jalalians controlled the provinces o...</td>
          <td>The Hasan-Jalalians commanded the area of Arts...</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.80</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
  </div>


```python
display_eval_report(
    (
        f"{model_b_name} Eval Result",
        result_rag_b.summary_metrics,
        result_rag_b.metrics_table,
    )
)
```
输出如下。

## gemini-1.0-pro 评估结果 { #gemini-10-pro-eval-result }
### 指标汇总 { #summary-metrics_1 }
<div id="summary-metrics">
    <table border=1>
      <thead>
        <tr>
          <th></th>
          <th>context_precision</th>
          <th>faithfulness</th>
          <th>rouge_score(mode=fmeasure)</th>
          <th>helpfulness</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.0</td>
          <td>0.916667</td>
          <td>0.479034</td>
          <td>4.0</td>
        </tr>
      </tbody>
    </table>
  </div>

### 详细指标 { #report-metrics_1 }
<div id="report-metrics">
    <table border=1>
      <thead>
        <tr>
          <th></th>
          <th>user_input</th>
          <th>retrieved_contexts</th>
          <th>response</th>
          <th>reference</th>
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
          <td>The frontal lobe, especially the dorsolateral ...</td>
          <td>frontal lobe and the parietal lobe</td>
          <td>1.0</td>
          <td>1.00</td>
          <td>0.666667</td>
          <td>4</td>
        </tr>
        <tr>
          <th>1</th>
          <td>What provided the Roman senate with exuberance?</td>
          <td>[In 62 BC, Pompey returned victorious from Asi...</td>
          <td>The Roman Senate's exuberance stemmed from its...</td>
          <td>Due to successes against Catiline.</td>
          <td>1.0</td>
          <td>0.75</td>
          <td>0.130435</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2</th>
          <td>What area did the Hasan-jalalians command?</td>
          <td>[The Seljuk Empire soon started to collapse. I...</td>
          <td>The Hasan-Jalalians controlled the provinces o...</td>
          <td>The Hasan-Jalalians commanded the area of Arts...</td>
          <td>1.0</td>
          <td>1.00</td>
          <td>0.640000</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
  </div>


### 可视化评估结果 { #visualise-evaluation-results }


```python
eval_results = []

eval_results.append(
    (model_a_name, result_rag_a.summary_metrics, result_rag_a.metrics_table)
)
eval_results.append(
    (model_b_name, result_rag_b.summary_metrics, result_rag_b.metrics_table)
)
```


```python
plot_radar_plot(eval_results, max_score=5)
```
![雷达图](../../../_static/radar_plot.png)

```python
plot_bar_plot(eval_results)
```
![柱状图](../../../_static/bar_plot.png)

继续阅读本系列其他教程。

- [结合 Vertex AI 使用 Ragas](vertexai_x_ragas.md)，评估 LLM 工作流。
- [对齐 LLM 指标](vertexai_alignment.md)，训练并对齐评估模型，使其更贴近人工判断。
