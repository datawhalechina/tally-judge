# 从 v0.1 迁移到 v0.2 { #migration-from-v01-to-v02 }

从 v0.2 开始，Ragas 逐步从 RAG 流水线评估库扩展为通用 LLM 应用评估库。这带来了一些基础性、不兼容的改动。本指南帮助你完成迁移。

## 内容概览 { #outline }

1. 评估数据集
2. 指标
3. 测试集生成
4. 提示词对象

## 评估数据集 { #evaluation-dataset }

数据类型从 HuggingFace 的 [`Dataset`](https://huggingface.co/docs/datasets/v3.0.1/en/package_reference/main_classes#datasets.Dataset) 改为 Ragas 的 [`EvaluationDataset`][ragas.dataset_schema.EvaluationDataset]。详情参阅[评估数据集](../../concepts/components/eval_dataset.md)和[评估样本](../../concepts/components/eval_sample.md)。

可以按以下方式转换。

```python
from ragas import EvaluationDataset, SingleTurnSample

hf_dataset = ... # your huggingface evaluation dataset
eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

# save eval dataset
eval_dataset.to_csv("path/to/save/dataset.csv")

# load eva dataset
eval_dataset = EvaluationDataset.from_csv("path/to/save/dataset.csv")
```

## 评估指标 { #metrics }

原有默认指标继续支持，同时新增多种指标，完整列表见[文档](../../concepts/metrics/available_metrics/index.md)。

指标使用方式有两项变化。

首先，建议用所选评估 LLM 显式初始化指标，再传给 [`evaluate()`][ragas.evaluation.evaluate]，使每个指标所用模型更加明确。

```python
from ragas.metrics import faithfullness # old way, not recommended but still supported till v0.3
from ragas.metrics import Faithfulness

# preffered way
faithfulness_metric = Faithfulness(llm=your_evaluator_llm)
```
其次，[`metrics.ascore`][ragas.metrics.base.Metric.ascore] 将弃用，改用 [`single_turn_ascore`][ragas.metrics.base.SingleTurnMetric.single_turn_ascore]，迁移方法如下。

```python
# create a Single Turn Sample
from ragas import SingleTurnSample

sample = SingleTurnSample(
    user_input="user query",
    response="response from your pipeline",
    retrieved_contexts=["retrieved", "contexts", "from your pipeline" ]
)

# Init the metric
from ragas.metrics import Faithfulness
faithfulness_metric = Faithfulness(llm=your_evaluator_llm)
await faithfulness_metric.single_turn_ascore(sample)
```
输出如下。
```
1
```

## 测试集生成 { #testset-generation }

[测试集生成](../../concepts/test_data_generation/rag.md)已经重构，以降低成本。端到端用法见[快速入门](../../getstarted/rag_testset_generation.md)。

**主要变化**

- 使用新的知识图谱替代 `Docstore`
- 新增 `Transforms`，将文档转换为信息丰富的知识图谱
- 通过 `Synthesizer` 对象提供更多定制能力
- 新工作流成本更低，便于保存中间状态

如果迁移遇到问题，欢迎在社区讨论，我们会提供帮助。🙂

## 提示词对象 { #prompt-object }

所有提示词重写为基于 [`BasePrompt`][ragas.prompt.base.BasePrompt] 的 [`PydanticPrompt`][ragas.prompt.pydantic_prompt.PydanticPrompt]。使用旧版 `Prompt` 时，需要按文档升级。

- [创建提示词的操作指南](../customizations/metrics/modifying-prompts-metrics.md)
- [相关 GitHub PR](https://github.com/vibrantlabsai/ragas/pull/1462)

!!! note "需要更多帮助？"

    如有问题，可以在[此 issue](https://github.com/vibrantlabsai/ragas/issues/1486)提问，或通过 [cal.com](https://cal.com/shahul-ragas/30min)联系我们。

