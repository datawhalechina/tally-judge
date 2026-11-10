# 让 LLM 评估与人工判断对齐 { #aligning-llm-evaluators-with-human-judgment }

本教程属于结合 Vertex AI 使用 Ragas 的三篇系列教程。建议先阅读[入门教程](vertexai_x_ragas.md)，也可以直接跟随本篇操作。另一篇为[模型比较](vertexai_model_comparision.md)。

## 概览 { #overview }

本教程介绍如何使用 Ragas 训练并对齐自定义 LLM 指标。LLM 评估器有时会因风格、上下文或细节理解差异，作出偏离人工预期的判断。下面通过优化指标，使其更准确地反映人工判断。

你将完成以下步骤。

1. 使用 Ragas 定义基于模型的指标。
2. 从 HHH 数据集的 helpful 子集创建 EvaluationDataset。
3. 运行初始评估，建立指标表现的基线。
4. 审阅并标注 15～20 个评估样本。
5. 使用标注数据训练指标。
6. 重新评估，观察与人工判断的一致性是否提高。

## 开始使用 { #getting-started }

### 安装依赖 { #install-dependencies }


```python
%pip install --upgrade --user --quiet langchain-core langchain-google-vertexai langchain ragas
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

## 设置评估指标 { #set-up-eval-metrics }

LLM 指标具有较大潜力，但其判断有时与人工不一致。通过反馈闭环，可以缩小这一差距。

### 定义 evaluator_llm { #define-evaluator_llm }

导入所需封装，定义评估 LLM 与嵌入模型。


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


evaluator_llm = LangchainLLMWrapper(VertexAI(model_name="gemini-2.0-flash-001"))
evaluator_embeddings = LangchainEmbeddingsWrapper(VertexAIEmbeddings(model_name="text-embedding-004"))
```

### Ragas 指标 { #ragas-metrics }

Ragas 的多种模型指标可以通过调优对齐人工判断。本例使用用户定义的二元指标 **Aspect Critic**，详情参阅[相关文档](../../concepts/metrics/available_metrics/general_purpose.md#aspect-critic)。


```python
from ragas.metrics import AspectCritic

helpfulness_critic = AspectCritic(
    name="helpfulness",
    definition="Evaluate how helpful the assistant's response is to the user's query.",
    llm=evaluator_llm
)
```

运行以下代码，预览对齐前实际发送给 LLM 的提示词。


```python
print(helpfulness_critic.get_prompts()["single_turn_aspect_critic_prompt"].instruction)
```
输出如下。
```
Evaluate the Input based on the criterial defined. Use only 'Yes' (1) and 'No' (0) as verdict.
Criteria Definition: Evaluate how helpful the assistant's response is to the user's query.
```

### 定义对齐得分 { #defining-alignment-score }

本例使用二元指标，因此通过 F1 得分衡量一致性。对于其他指标，可调整函数，采用适合的度量方法。


```python
from typing import List
from sklearn.metrics import f1_score

def alignment_score(human_score: List[float], llm_score: List[float]) -> float:
    """
    Computes the alignment between human-annotated binary scores and LLM-generated binary scores
    using the F1-score metric.

    Args:
        human_score (List[int]): Binary labels from human evaluation (0 or 1).
        llm_score (List[int]): Binary labels from LLM predictions (0 or 1).

    Returns:
        float: The F1-score measuring alignment.
    """
    return f1_score(human_score, llm_score)
```

## 准备数据集 { #prepare-your-dataset }

`process_hhh_dataset` 为训练和对齐评估器处理 [HHH 数据集](https://paperswithcode.com/dataset/hhh?utm_source=chatgpt.com)。它交替赋予 0 和 1 标签，1 表示有帮助，0 表示无帮助，以标识更受偏好的回答。


```python
import numpy as np
from datasets import load_dataset
from ragas import EvaluationDataset


def process_hhh_dataset(split: str = "helpful", total_count: int = 50):
	dataset = load_dataset("HuggingFaceH4/hhh_alignment",split, split=f"test[:{total_count}]")
	data = []
	expert_scores = []

	for idx, entry in enumerate(dataset):
		# Extract input and target details
		user_input = entry['input']
		choices = entry['targets']['choices']
		labels = entry['targets']['labels']

		# Choose target based on whether the index is even or odd
		if idx % 2 == 0:
			target_label = 1
			score = 1
		else:
			target_label = 0
			score = 0

		label_index = labels.index(target_label)

		response = choices[label_index]

		data.append({
			'user_input': user_input,
			'response': response,
		})
		expert_scores.append(score)

	return EvaluationDataset.from_list(data), expert_scores

eval_dataset, expert_scores = process_hhh_dataset()
```

## 运行评估 { #run-evaluation }

定义好数据集与有帮助程度指标后，运行评估。


```python
from ragas import evaluate

results = evaluate(eval_dataset, metrics=[helpfulness_critic])
```
```
Evaluating: 100%|██████████| 50/50 [00:00<?, ?it/s]
```

首次运行展示当前评估器与人工之间的不一致程度，后续训练将尝试改善。

将指标结果与专家评分比较。


```python
human_score = expert_scores
llm_score = results.to_pandas()["helpfulness"].values

initial_score = alignment_score(human_score, llm_score)
initial_score
```
输出如下。
```
0.8076923076923077
```


## 审阅与标注 { #review-and-annotate }

获得评估结果后，开始审阅并标注。正如[让 LLM 裁判与人工评估对齐](https://blog.ragas.io/aligning-llm-as-judge-with-human-evaluators)所述，详细反馈有助于缩小差距。至少标注 15～20 条，覆盖指标可能判断不一致的不同场景。

这里提供了[示例标注文件](../../../_static/annotated_data.json)，可下载使用。

## 训练与对齐 { #training-and-alignment }

使用标注示例训练指标。训练采用无梯度提示词优化方法，根据反馈同时调整指令与少样本示例。


```python
from ragas.config import InstructionConfig, DemonstrationConfig

demo_config = DemonstrationConfig(embedding=evaluator_embeddings)
inst_config = InstructionConfig(llm=evaluator_llm)

helpfulness_critic.train(
    path="annotated_data.json",
    instruction_config=inst_config,
    demonstration_config=demo_config,
)
```
```
Overall Progress: 100%|██████████| 170/170 [00:00<?, ?it/s]

Few-shot examples [single_turn_aspect_critic_prompt]: 100%|██████████| 16/16 [00:00<?, ?it/s]
```

训练后，查看优化后的指令。


```python
print(helpfulness_critic.get_prompts()["single_turn_aspect_critic_prompt"].instruction)
```
输出如下。
```
You are provided with a user input and an assistant/model response. Your task is to evaluate the quality of the response based on how well it addresses the user input, considering all requests and constraints. Assign a score/verdict of 1 if the response is helpful, appropriate, and effective, and 0 if it is not. A good response should be accurate, complete, relevant, and provide a tangible improvement or solution, without omitting key information. Provide a brief explanation for your score/verdict.
```

## 重新评估 { #re-evaluate }

使用经过人工反馈对齐的指标重新评估数据集，量化改进程度与可靠性提升。


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


evaluator_llm = LangchainLLMWrapper(VertexAI(model_name="gemini-pro"))
evaluator_embeddings = LangchainEmbeddingsWrapper(VertexAIEmbeddings(model_name="text-embedding-004"))
```


```python
from ragas import evaluate

results2 = evaluate(eval_dataset, metrics=[helpfulness_critic])
```
```
Evaluating: 100%|██████████| 50/50 [00:00<?, ?it/s]
```

将更新后的结果与专家评分比较。


```python
human_score = expert_scores
llm_score = results2.to_pandas()["helpfulness"].values

new_score = alignment_score(human_score, llm_score)
new_score
```
输出如下。
```
0.8444444444444444
```

继续阅读本系列其他教程。

- [结合 Vertex AI 使用 Ragas](vertexai_x_ragas.md)，评估 LLM 工作流。
- [模型比较](vertexai_model_comparision.md)，使用 Ragas 指标比较 Vertex AI 模型在 RAG 问答任务上的表现。
