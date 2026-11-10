# 优化器 API 参考 { #optimizers-api-reference }

Ragas 提供自动优化指标提示词的优化器。本页介绍可用的优化器类及其配置。

## 概览 { #overview }

优化器使用带有标准评分的标注数据集，通过以下方式改进提示词，提高指标准确性。

- **指令优化**，寻找更合适的提示词表述
- **示例优化**，选择有效的少样本示例
- **搜索策略**，高效探索提示词空间

## 核心类 { #core-classes }

::: ragas.optimizers
    options:
        members:
            - Optimizer
            - GeneticOptimizer
            - DSPyOptimizer

## GeneticOptimizer

用于提示词指令的简单进化优化器。

### 参数 { #parameters }

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `max_steps` | `int` | 50 | 最大进化步数 |
| `population_size` | `int` | 10 | 每代种群大小 |
| `mutation_rate` | `float` | 0.2 | 变异概率 |

### 用法 { #usage }

```python
from ragas.optimizers import GeneticOptimizer
from ragas.config import InstructionConfig

optimizer = GeneticOptimizer(
    max_steps=50,
    population_size=10,
)

config = InstructionConfig(llm=llm, optimizer=optimizer)
metric.optimize_prompts(dataset, config)
```

### 工作原理 { #how-it-works }

1. 生成一组提示词变体
2. 在标注数据集上评估每个变体
3. 选择表现最好的个体
4. 通过交叉和变异产生下一代
5. 重复执行，最多迭代 max_steps 次

**优点**。简单，适用于数据有限的场景。
**局限**。收敛较慢，只优化指令。

## DSPyOptimizer

采用 DSPy 的 [MIPROv2](https://dspy.ai/api/optimizers/MIPROv2/) 算法的进阶优化器。

### 参数 { #parameters_1 }

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `num_candidates` | `int` | 10 | 尝试的提示词变体数量 |
| `max_bootstrapped_demos` | `int` | 5 | 自动生成示例的最大数量 |
| `max_labeled_demos` | `int` | 5 | 人工标注示例的最大数量 |
| `init_temperature` | `float` | 1.0 | 探索温度（0.0～2.0） |

### 用法 { #usage_1 }

```python
from ragas.optimizers import DSPyOptimizer
from ragas.config import InstructionConfig

optimizer = DSPyOptimizer(
    num_candidates=10,
    max_bootstrapped_demos=5,
    max_labeled_demos=5,
)

config = InstructionConfig(llm=llm, optimizer=optimizer)
metric.optimize_prompts(dataset, config)
```

### 工作原理 { #how-it-works_1 }

1. 生成候选提示词指令
2. 从数据中自举生成少样本示例
3. 选择最有效的人工标注示例
4. 在数据集上评估各种组合
5. 返回表现最好的配置

进一步了解 DSPy 的概念。
- [签名](https://dspy.ai/learn/programming/signatures/)，DSPy 定义输入/输出规范的方式
- [优化器](https://dspy.ai/learn/optimization/optimizers/)，改进提示词与语言模型权重的算法
- [模块](https://dspy.ai/learn/programming/modules/)，LLM 程序的构建单元

**优点**。结合指令与示例，通常取得更好的结果。
**局限**。需要安装 DSPy，LLM 调用更多。

### 安装 { #installation }

[DSPy](https://dspy.ai/) 是可选依赖。

```bash
# Using uv (recommended)
uv add "ragas[dspy]"

# Using pip
pip install "ragas[dspy]"
```

### 成本估算 { #cost-estimation }

每次优化的 LLM 调用次数约为，

```
Total calls ≈ num_candidates × 30 + max_bootstrapped_demos × 7
```

例如，

- 默认配置（10、5、5），约 335 次
- 节省预算的配置（5、2、3），约 164 次
- 更充分探索的配置（20、10、10），约 670 次

## 优化器基类 { #optimizer-base-class }

::: ragas.optimizers.base.Optimizer
    options:
        show_source: false
        members:
            - optimize

## 配置 { #configuration }

两种优化器均通过 `InstructionConfig` 配置。

```python
from ragas.config import InstructionConfig

config = InstructionConfig(
    llm=llm,                      # LLM for optimization
    optimizer=optimizer_instance, # Optimizer to use
)

# Use with metric
metric.optimize_prompts(dataset, config)
```

## 数据集格式 { #dataset-format }

优化器需要包含标准评分的标注数据集。

```python
from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation
)

# Create annotated sample
prompt_annotation = PromptAnnotation(
    prompt_input={"user_input": "...", "response": "..."},
    prompt_output={"score": 0.9},
    edited_output=None,  # Optional: corrected output
)

sample = SampleAnnotation(
    metric_input={"user_input": "...", "response": "..."},
    metric_output=0.9,  # Ground truth score
    prompts={"metric_prompt": prompt_annotation},
    is_accepted=True,  # Include in optimization
)

# Create dataset
dataset = SingleMetricAnnotation(
    name="metric_name",
    samples=[sample, ...]  # 20-50+ samples recommended
)
```

## 损失函数 { #loss-functions }

优化器通过损失函数衡量提示词质量。

```python
from ragas.losses import MSELoss, HuberLoss

# Mean Squared Error (default)
loss = MSELoss()

# Huber Loss (robust to outliers)
loss = HuberLoss(delta=1.0)

# Use with config
config = InstructionConfig(llm=llm, optimizer=optimizer, loss=loss)
```

## 比较 { #comparison }

| 特性 | GeneticOptimizer | DSPyOptimizer |
|---------|------------------|---------------|
| 安装 | 内置 | 需要 `ragas[dspy]` |
| 优化目标 | 仅指令 | 指令与示例 |
| 最小数据集规模 | 10 个以上样本 | 20 个以上样本 |
| 典型 LLM 调用次数 | 100～500 | 200～700 |
| 准确率提升 | +5～8% | +8～12% |
| 适用场景 | 快速优化 | 生产环境指标 |

## 另请参阅 { #see-also }

- [DSPy 优化器指南](../howtos/customizations/optimizers/index.md)，详细用法
- [指标定制](../concepts/metrics/overview/index.md)，创建指标
- [提示词 API 参考](prompt.md)，理解提示词

## 更多资料 { #additional-resources }

**DSPy 文档**
- [DSPy 官方文档](https://dspy.ai/)，完整使用指南
- [MIPROv2 API 参考](https://dspy.ai/api/optimizers/MIPROv2/)，MIPROv2 详细文档
- [DSPy 优化器概览](https://dspy.ai/learn/optimization/optimizers/)，各类优化器指南
- [DSPy GitHub 仓库](https://github.com/stanfordnlp/dspy)，源码与示例

**研究论文**
- [多阶段语言模型程序的指令与示例优化](https://arxiv.org/abs/2406.11695)，MIPROv2 论文
