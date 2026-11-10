# 数据集与实验结果 { #datasets-and-experiment-results }

评估 AI 系统时，通常会使用两类数据。

1. **评估数据集**，存放在 `datasets` 目录中
2. **评估结果**，存放在 `experiments` 目录中

## 评估数据集 { #evaluation-datasets }

评估数据集包含以下内容。

1. 输入，系统需要处理的一组输入数据
2. 预期输出（可选），系统针对给定输入应当生成的结果或回答
3. 元数据（可选），与数据集一同存储的补充信息

例如，检索增强生成（RAG）系统的数据集可以包含查询（系统输入）、评分说明（用于评价系统输出），以及查询复杂度等元数据。

元数据便于对数据集进行分组和细分，从不同维度分析结果。例如，可以分别查看系统对复杂查询与简单查询的表现，或比较其处理不同语言的能力。

## 实验结果 { #experiment-results }

实验结果包括以下内容。

1. 数据集中的全部属性
2. 被评估系统的回答
3. 各项指标的结果
4. 可选的元数据，例如指向某次输入对应系统追踪记录的 URI

例如，RAG 系统的实验结果可以包含查询、评分说明、回答、准确率得分、系统追踪链接等。

## 在 Ragas 中使用数据集 { #working-with-datasets-in-ragas }

Ragas 提供 `Dataset` 类来管理评估数据集，用法如下。

### 创建数据集 { #creating-a-dataset }

```python
from ragas import Dataset

# Create a new dataset
dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir="./data")

# Add a sample to the dataset
dataset.append({
    "id": "sample_1",
    "query": "What is the capital of France?",
    "expected_answer": "Paris",
    "metadata": {"complexity": "simple", "language": "en"}
})
```

### 加载已有数据集 { #loading-an-existing-dataset }

```python
# Load an existing dataset
dataset = Dataset.load(
    name="my_evaluation",
    backend="local/csv",
    root_dir="./data"
)
```

### 数据集结构 { #dataset-structure }

Ragas 的数据集结构灵活，可以包含评估所需的任意字段。常见字段包括以下几类。

- `id`，每个样本的唯一标识符
- `query` 或 `input`，AI 系统的输入
- `expected_output` 或 `ground_truth`，预期回答（如有）
- `metadata`，样本的补充信息

### 创建数据集的最佳实践 { #best-practices-for-dataset-creation }

1. **样本具有代表性**。确保数据集能够反映 AI 系统将要遇到的真实场景。

2. **分布均衡**。覆盖不同难度、主题和边界情况。

3. **注重质量**。精心整理的高质量样本比大量低质量样本更有价值。

4. **提供充分的元数据**。记录相关元数据，便于从不同维度分析表现。

5. **进行版本管理**。跟踪数据集随时间的变化，确保实验可复现。

## 数据集存储与管理 { #dataset-storage-and-management }

### 本地存储 { #local-storage }

本地开发或使用小型数据集时，可以采用 CSV 文件。

```python
dataset = Dataset(name="my_eval", backend="local/csv", root_dir="./datasets")
```

### 云端存储 { #cloud-storage }

处理大型数据集或开展团队协作时，可以考虑使用云端后端。

```python
# Google Drive (experimental)
dataset = Dataset(name="my_eval", backend="gdrive", root_dir="folder_id")

# Other backends can be added as needed
```

### 数据集版本管理 { #dataset-versioning }

记录数据集版本，确保实验可复现。

```python
# Include version in dataset name
dataset = Dataset(name="my_eval_v1.2", backend="local/csv", root_dir="./datasets")
```

## 集成到评估工作流 { #integration-with-evaluation-workflows }

数据集可以直接用于 Ragas 评估工作流。

```python
from ragas import experiment, Dataset

# Load your dataset
dataset = Dataset.load(name="my_evaluation", backend="local/csv", root_dir="./data")

# Define your experiment
@experiment()
async def my_experiment(row):
    # Process the input through your AI system
    response = await my_ai_system(row["query"])

    # Return results for metric evaluation
    return {
        **row,  # Include original data
        "response": response,
        "experiment_name": "baseline_v1"
    }

# Run evaluation on the dataset
results = await my_experiment.arun(dataset)
```

通过这种集成方式，可以分别管理测试数据（datasets）与评估结果（experiments），便于跟踪进展并比较不同方案。
