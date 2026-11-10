# 使用 DSPy 优化器进行高级提示词优化 { #dspy-optimizer-for-advanced-prompt-optimization }

DSPyOptimizer 通过 DSPy 的 MIPROv2 算法优化 Ragas 指标提示词，将指令优化与示例优化结合起来，通常能取得优于简单进化方法的效果。

## 概览 { #overview }

**DSPyOptimizer** 使用 MIPROv2（根据结果排序的多提示词指令提议算法），从以下方面优化指标提示词。

- **指令优化**，生成并测试多个提示词变体
- **示例优化**，自动选择有效的少样本示例
- **联合搜索**，同时探索指令与示例空间

当你拥有高质量标注数据时，这种方法通常比简单的 GeneticOptimizer 更有效。

## 安装 { #installation }

DSPy 是可选依赖，安装方法如下。

```bash
# Using uv (recommended)
uv add "ragas[dspy]"

# Using pip
pip install "ragas[dspy]"
```

## 基本用法 { #basic-usage }

### 前置条件 { #prerequisites }

需要准备以下内容。

1. **标注数据集**，包含指标的标准评分
2. **带提示词的指标**，使用 PydanticPrompt 的指标，Ragas 中多数指标属于此类
3. **LLM**，用于优化的语言模型，考虑成本时可选择 gpt-4o-mini

### 快速开始 { #quick-start }

```python
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from ragas.optimizers import DSPyOptimizer
from ragas.config import InstructionConfig

# Setup LLM for optimization
client = OpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Initialize metric
metric = Faithfulness(llm=llm)

# Create annotated dataset (see below for format)
dataset = create_annotated_dataset()

# Configure DSPy optimizer
config = InstructionConfig(
    llm=llm,
    optimizer=DSPyOptimizer(
        num_candidates=10,          # Try 10 prompt variations
        max_bootstrapped_demos=5,   # Generate up to 5 examples
        max_labeled_demos=5,        # Use up to 5 human annotations
    )
)

# Optimize the metric's prompts
metric.optimize_prompts(dataset, config)

# Save optimized prompts for reuse
metric.save_prompts("optimized_faithfulness.json")
```

### 标注数据集格式 { #annotated-dataset-format }

DSPy 优化器需要标准标注。

```python
from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation
)

# Create prompt annotations
prompt_annotation = PromptAnnotation(
    prompt_input={"user_input": "...", "response": "..."},
    prompt_output={"score": 0.9},  # Actual metric output
    edited_output=None,  # Or corrected output if needed
)

# Create sample with annotations
sample = SampleAnnotation(
    metric_input={"user_input": "...", "response": "..."},
    metric_output=0.9,  # Ground truth score
    prompts={"faithfulness_prompt": prompt_annotation},
    is_accepted=True,  # Whether to use in optimization
)

# Create dataset
dataset = SingleMetricAnnotation(
    name="faithfulness",
    samples=[sample, ...]  # Need 20-50+ samples for best results
)
```

## 进阶配置 { #advanced-configuration }

### 优化参数 { #optimization-parameters }

可以通过以下配置控制 MIPROv2。

```python
optimizer = DSPyOptimizer(
    num_candidates=20,           # More candidates = better prompts, higher cost
    max_bootstrapped_demos=10,   # Auto-generated few-shot examples
    max_labeled_demos=10,        # Human-annotated examples to use
    init_temperature=1.0,        # Exploration temperature (0.0-2.0)
)
```

**参数说明**

| 参数 | 默认值 | 描述 | 成本影响 |
|-----------|---------|-------------|-------------|
| `num_candidates` | 10 | 尝试的提示词变体数 | 高，线性增长 |
| `max_bootstrapped_demos` | 5 | 自动生成的示例数 | 中，增加 LLM 调用 |
| `max_labeled_demos` | 5 | 使用的人工标注示例数 | 低，使用现有数据 |
| `init_temperature` | 1.0 | 探索随机性 | 无，只影响算法 |

### 成本优化 { #cost-optimization }

MIPROv2 优化可能成本较高，可通过以下方式降低费用。

```python
# Budget-conscious configuration
budget_optimizer = DSPyOptimizer(
    num_candidates=5,            # Fewer candidates
    max_bootstrapped_demos=2,    # Fewer generated examples
    max_labeled_demos=3,         # More reliance on annotations
    init_temperature=0.5,        # Less exploration
)

# Use cheaper LLM for optimization
cheap_llm = llm_factory("gpt-4o-mini", client=client)
config = InstructionConfig(llm=cheap_llm, optimizer=budget_optimizer)
```

**成本估算**

- 每个候选约需 10～50 次 LLM 调用
- 每个自举示例约需 5～10 次调用
- 总次数约为 `num_candidates * 30 + max_bootstrapped_demos * 7`

## 与 GeneticOptimizer 比较 { #comparing-with-geneticoptimizer }

### 何时使用 DSPyOptimizer { #when-to-use-dspyoptimizer }

✅ **适用于以下情况。**

- 拥有 50 个以上高质量标注示例
- 需要尽可能高的指标准确性
- 可以承担优化所需的 100～500 次 LLM 调用
- 正在优化生产环境中的关键指标

### 何时使用 GeneticOptimizer { #when-to-use-geneticoptimizer }

✅ **适用于以下情况。**

- 标注数据有限，少于 20 个示例
- 需要更快、成本更低的优化
- 正处于原型设计阶段
- 仅优化简单指令即可满足需求

### 并列比较 { #side-by-side-comparison }

```python
from ragas.optimizers import GeneticOptimizer, DSPyOptimizer

# Genetic optimizer - simpler, faster, cheaper
genetic_config = InstructionConfig(
    llm=llm,
    optimizer=GeneticOptimizer(
        max_steps=50,          # Evolution steps
        population_size=10,    # Population per generation
    )
)

# DSPy optimizer - advanced, better results, more expensive
dspy_config = InstructionConfig(
    llm=llm,
    optimizer=DSPyOptimizer(
        num_candidates=10,
        max_bootstrapped_demos=5,
        max_labeled_demos=5,
    )
)

# Compare results
metric_genetic = Faithfulness(llm=llm)
metric_genetic.optimize_prompts(dataset, genetic_config)

metric_dspy = Faithfulness(llm=llm)
metric_dspy.optimize_prompts(dataset, dspy_config)

# Evaluate on holdout set
test_scores_genetic = metric_genetic.batch_score(test_set)
test_scores_dspy = metric_dspy.batch_score(test_set)
```

**典型结果**

| 指标 | GeneticOptimizer | DSPyOptimizer | 提升 |
|--------|------------------|---------------|-------------|
| 忠实度 | 0.82 | 0.89 | +8.5% |
| 回答相关性 | 0.75 | 0.84 | +12% |
| 上下文精确率 | 0.78 | 0.86 | +10% |

## 优化多个指标 { #working-with-multiple-metrics }

采用同样的方法优化多项指标。

```python
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision
)

metrics = {
    "faithfulness": Faithfulness(llm=llm),
    "answer_relevancy": AnswerRelevancy(llm=llm),
    "context_precision": ContextPrecision(llm=llm),
}

# Optimize each metric
for name, metric in metrics.items():
    print(f"Optimizing {name}...")

    # Load metric-specific dataset
    dataset = load_annotated_dataset(name)

    # Optimize
    metric.optimize_prompts(dataset, dspy_config)

    # Save
    metric.save_prompts(f"optimized_{name}.json")
```

## 故障排查 { #troubleshooting }

### 导入错误 { #import-error }

如果出现 `ImportError: DSPy optimizer requires dspy-ai`，

```bash
# Install the DSPy extra
uv add "ragas[dspy]"
# or
pip install "ragas[dspy]"
```

### 优化耗时过长 { #optimization-takes-too-long }

可以减少 LLM 调用次数。

```python
fast_optimizer = DSPyOptimizer(
    num_candidates=3,      # Minimum viable
    max_bootstrapped_demos=1,
    max_labeled_demos=3,
)
```

### 结果不佳 { #poor-results }

常见原因包括以下几种。

1. **数据不足**，需要 20 个以上高质量标注
2. **标注质量不高**，确保标准评分准确
3. **LLM 不合适**，使用 gpt-4o 或更强的模型优化
4. **配置不合适**，先尝试默认参数

### 内存问题 { #memory-issues }

MIPROv2 处理大数据集时可能占用较多内存。

```python
# Process in smaller batches
from ragas.dataset_schema import SingleMetricAnnotation

def optimize_in_batches(dataset, batch_size=20):
    # Split dataset
    batches = [
        dataset.select(range(i, min(i + batch_size, len(dataset.samples))))
        for i in range(0, len(dataset.samples), batch_size)
    ]

    # Optimize on first batch for speed
    best_batch = batches[0]
    metric.optimize_prompts(best_batch, dspy_config)
```

## 最佳实践 { #best-practices }

### 数据质量 { #data-quality }

1. **示例多样**，覆盖常见场景与边界情况
2. **标注准确**，复核标准评分
3. **数量充足**，生产指标使用 50 个以上示例

### 优化策略 { #optimization-strategy }

1. **从小规模开始**，先测试 3～5 个候选
2. **逐步迭代**，按需增加参数
3. **验证效果**，始终在留出集上测试
4. **缓存结果**，保存优化后的提示词，避免重复运行

### 生产部署 { #production-deployment }

```python
# 1. Optimize offline
metric = Faithfulness(llm=optimization_llm)
metric.optimize_prompts(training_dataset, dspy_config)
metric.save_prompts("production_faithfulness.json")

# 2. Load in production
production_metric = Faithfulness(llm=production_llm)
production_metric.load_prompts("production_faithfulness.json")

# 3. Use for evaluation
results = production_metric.batch_score(production_samples)
```

## 另请参阅 { #see-also }

- [优化器 API 参考](../../../references/optimizers.md)，完整 API 文档
- [指标定制](../../../concepts/metrics/overview/index.md)，创建自定义指标
- [DSPy 文档](https://dspy-docs.vercel.app/)，进一步了解 DSPy
