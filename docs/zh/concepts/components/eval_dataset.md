# 评估数据集 { #evaluation-dataset }

评估数据集是一组类型一致的[数据样本](eval_sample.md)，用于衡量 AI 应用的表现与能力。在 Ragas 中，`EvaluationDataset` 类以结构化方式组织和管理评估样本。

- [概览](#overview)
- [使用 SingleTurnSample 创建评估数据集](#creating-an-evaluation-dataset-from-singleturnsamples)
- [从 Hugging Face Datasets 加载评估数据集](#loading-an-evaluation-dataset-from-hugging-face-datasets)

## 概览 { #overview }

### 评估数据集的结构 { #structure-of-an-evaluation-dataset }

评估数据集具有以下结构要求。

- **样本**。由 [SingleTurnSample](eval_sample.md#singleturnsample) 或 [MultiTurnSample](eval_sample.md#multiturnsample) 实例组成，每个样本代表一次独立交互或一个场景。
- **一致性**。数据集中所有样本应当属于同一类型，即全部为单轮样本或全部为多轮样本，以保证评估一致性。


### 构建有效评估数据集的建议 { #guidelines-for-curating-an-effective-evaluation-dataset }

- **明确目标**。确定需要评估的应用能力和测试场景，收集能够反映这些目标的样本。

- **收集具有代表性的数据**。覆盖多种场景、用户输入和预期回答，全面评估 AI 应用。可以从不同来源收集数据，也可以[生成合成数据](../../howtos/customizations/index.md#testset-generation)。

- **兼顾质量与规模**。样本量应足以支持有意义的分析，同时便于管理。确保数据质量可靠，能够准确反映需要评估的真实场景。


## 使用 SingleTurnSample 创建评估数据集 { #creating-an-evaluation-dataset-from-singleturnsamples }

本例通过多个 `SingleTurnSample` 实例创建 `EvaluationDataset`，依次介绍样本创建、数据集组装及基本操作。


**第一步，导入所需类**

首先导入 `SingleTurnSample` 与 `EvaluationDataset` 类。
```python
from ragas import SingleTurnSample, EvaluationDataset
```

**第二步，创建样本**

创建多个 `SingleTurnSample` 实例，分别表示各条评估样本。

```python
# Sample 1
sample1 = SingleTurnSample(
    user_input="What is the capital of Germany?",
    retrieved_contexts=["Berlin is the capital and largest city of Germany."],
    response="The capital of Germany is Berlin.",
    reference="Berlin",
)

# Sample 2
sample2 = SingleTurnSample(
    user_input="Who wrote 'Pride and Prejudice'?",
    retrieved_contexts=["'Pride and Prejudice' is a novel by Jane Austen."],
    response="'Pride and Prejudice' was written by Jane Austen.",
    reference="Jane Austen",
)

# Sample 3
sample3 = SingleTurnSample(
    user_input="What's the chemical formula for water?",
    retrieved_contexts=["Water has the chemical formula H2O."],
    response="The chemical formula for water is H2O.",
    reference="H2O",
)
```

**第三步，创建 EvaluationDataset**

将 `SingleTurnSample` 实例列表传给 `EvaluationDataset`。

```python
dataset = EvaluationDataset(samples=[sample1, sample2, sample3])
```

## 从 Hugging Face Datasets 加载评估数据集 { #loading-an-evaluation-dataset-from-hugging-face-datasets }

实际使用时，你可能需要从已有数据源加载评估数据，例如 Hugging Face Datasets 库。以下示例演示如何加载 Hugging Face 数据集，并转换为 `EvaluationDataset` 实例。

确保数据集包含评估所需的字段，例如用户输入、检索上下文、回答和参考答案。

```python
from datasets import load_dataset
dataset = load_dataset("vibrantlabsai/amnesty_qa","english_v3")
```

将数据集加载为 Ragas 的 `EvaluationDataset` 对象。

```python
from ragas import EvaluationDataset

eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])
```
