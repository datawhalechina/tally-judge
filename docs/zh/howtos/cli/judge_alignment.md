# 裁判对齐快速入门 { #judge-alignment-quickstart }

`judge_alignment` 模板衡量 LLM 裁判与人工评判标准的一致程度。

## 创建项目 { #create-the-project }

```sh
ragas quickstart judge_alignment
cd judge_alignment
```

## 安装依赖 { #install-dependencies }

```sh
uv sync
```

## 设置 API 密钥 { #set-your-api-key }

```sh
export OPENAI_API_KEY="your-openai-key"
```

## 运行评估 { #run-the-evaluation }

```sh
uv run python evals.py
```

## 项目结构 { #project-structure }

```
judge_alignment/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## 评估内容 { #what-it-evaluates }

模板评估 LLM 裁判与人工判断的对齐程度。

- **场景**，使用 LLM 裁判评价已有回答
- **人工标签**，作为标准的通过/失败标注
- **LLM 裁判**，根据评分标准评价同一组回答
- **对齐指标**，人工与 LLM 判断的一致程度

## 理解代码 { #understanding-the-code }

### 裁判指标（`evals.py`） { #judge-metrics-evalspy }

比较两种裁判实现。

```python
# Baseline judge (simple prompt)
accuracy_metric = DiscreteMetric(
    name="accuracy",
    prompt="Check if response contains points from grading notes...",
    allowed_values=["pass", "fail"],
)

# Improved judge (enhanced with abbreviation guide)
accuracy_metric_v2 = DiscreteMetric(
    name="accuracy",
    prompt="""Evaluate if response covers ALL key concepts...

    ABBREVIATION GUIDE:
    • Financial: val=valuation, post-$=post-money, rev=revenue...
    • Business: mkt=market, reg=regulation...
    """,
    allowed_values=["pass", "fail"],
)
```

### 评估 { #the-evaluation }

测试与人工判断的一致性。

```python
@discrete_metric(name="alignment", allowed_values=["aligned", "misaligned"])
def alignment_metric(llm_judgment: str, human_judgment: str):
    # Compares LLM judge output with human label
    return "aligned" if llm_judgment == human_judgment else "misaligned"
```

## 测试数据 { #test-data }

数据集包含以下内容。
- 已评估的回答
- 人工通过/失败标签
- 包含预期要点的评分说明
- 多种缩略语与业务术语

## 使用场景 { #use-cases }

### 比较裁判版本 { #compare-judge-versions }

分别使用两种裁判运行实验。

```python
# Test baseline judge
results_v1 = await run_with_judge(accuracy_metric)

# Test improved judge
results_v2 = await run_with_judge(accuracy_metric_v2)

# Compare alignment rates
```

### 提升裁判质量 { #improve-judge-quality }

迭代裁判提示词，提高一致性。

1. 找出判断不一致的规律
2. 明确评分标准，更新裁判提示词
3. 重新评估一致性
4. 重复执行，直到达到要求

## 下一步 { #next-steps }

- [提示词评估](prompt_evals.md)，比较不同提示词
- [LLM 基准测试](benchmark_llm.md)，比较不同模型
