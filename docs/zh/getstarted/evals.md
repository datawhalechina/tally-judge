# 评估简单的 LLM 应用 { #evaluate-a-simple-llm-application }

本指南演示如何使用 `ragas` 测试和评估 LLM 应用。你只需具备 AI 应用构建与评估的基本知识。安装 `ragas` 的方法请参阅[安装说明](install.md)。

!!! tip "获取可运行示例"
    要快速体验这些概念，可以使用快速入门命令创建项目。

    === "uvx（推荐）"
        ```sh
        uvx ragas quickstart rag_eval
        cd rag_eval
        uv sync
        ```

    === "先安装 Ragas"
        ```sh
        pip install ragas
        ragas quickstart rag_eval
        cd rag_eval
        uv sync
        ```

    这会生成一个带有示例代码的完整项目。跟随本指南，即可理解这些代码的作用。

## 项目结构 { #project-structure }

生成的项目如下。

```sh
rag_eval/
├── README.md             # Project documentation and setup instructions
├── pyproject.toml        # Project configuration for uv and pip
├── evals.py              # Your evaluation workflow
├── rag.py                # Your RAG/LLM application
├── __init__.py           # Makes this a Python package
└── evals/                # Evaluation artifacts
    ├── datasets/         # Test data files (optional)
    ├── experiments/      # Results from running evaluations (CSV files saved here)
    └── logs/             # Evaluation execution logs
```

**重点关注以下文件。**

- **`evals.py`**，包含数据集加载和评估逻辑，定义评估工作流
- **`rag.py`**，包含 RAG/LLM 应用代码，例如查询引擎、检索逻辑等

## 理解代码 { #understanding-the-code }

在生成项目的 `evals.py` 文件中，可以看到主要工作流程。

1. **加载数据集**，使用 `SingleTurnSample` 定义测试用例
2. **查询 RAG 系统**，获取应用的回答
3. **评估回答**，将回答与标准答案比较
4. **展示结果**，在控制台输出评估摘要
5. **保存结果**，自动写入 `evals/experiments/` 目录下的 CSV 文件

模板提供了可自行定制的模块化函数。

```python
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset

def load_dataset():
    """Load test dataset for evaluation."""
    data_samples = [
        SingleTurnSample(
            user_input="What is Ragas?",
            response="",  # Will be filled by querying RAG
            reference="Ragas is an evaluation framework for LLM applications",
            retrieved_contexts=[],
        ),
        # Add more test cases...
    ]
    return EvaluationDataset(samples=data_samples)
```

你可以加入更多[指标](../concepts/metrics/available_metrics/index.md)和更复杂的评估逻辑。进一步了解 [Ragas 中的评估](../concepts/experimentation.md)。

### 选择 LLM 服务商 { #choosing-your-llm-provider }

快速入门项目默认在 `_init_clients()` 函数中初始化 OpenAI LLM。通过 `llm_factory`，可以轻松切换服务商。

=== "OpenAI"
    设置 OpenAI API 密钥。

    ```sh
    export OPENAI_API_KEY="your-openai-key"
    ```

    在 `evals.py` 的 `_init_clients()` 函数中使用以下代码。

    ```python
    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI()
    llm = llm_factory("gpt-4o", client=client)
    ```

    快速入门项目已经完成了这项配置。

=== "Anthropic Claude"
    设置 Anthropic API 密钥。

    ```sh
    export ANTHROPIC_API_KEY="your-anthropic-key"
    ```

    在 `evals.py` 的 `_init_clients()` 函数中使用以下代码。

    ```python
    import os
    from anthropic import Anthropic
    from ragas.llms import llm_factory

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    llm = llm_factory("claude-3-5-sonnet-20241022", provider="anthropic", client=client)
    ```

=== "Google Gemini"
    设置 Google 凭据。

    ```sh
    export GOOGLE_API_KEY="your-google-api-key"
    ```

    在 `evals.py` 的 `_init_clients()` 函数中使用以下代码。

    ```python
    import os
    import google.generativeai as genai
    from ragas.llms import llm_factory

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    client = genai.GenerativeModel("gemini-2.0-flash")
    llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
    ```

=== "本地模型（Ollama）"
    在本地安装并运行 Ollama，然后修改 `evals.py` 中的 `_init_clients()` 函数。

    ```python
    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI(
        api_key="ollama",  # Ollama doesn't require a real key
        base_url="http://localhost:11434/v1"
    )
    llm = llm_factory("mistral", provider="openai", client=client)
    ```

=== "自定义及其他服务商"
    对于提供 OpenAI 兼容 API 的 LLM，可以使用以下方式。

    ```python
    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI(
        api_key="your-api-key",
        base_url="https://your-api-endpoint"
    )
    llm = llm_factory("model-name", provider="openai", client=client)
    ```

    更多详情，请了解 [LLM 集成](../concepts/metrics/index.md)。

### 使用内置指标 { #using-pre-built-metrics }

`ragas` 为常见评估任务提供了内置指标。例如，[Aspect Critique](../concepts/metrics/available_metrics/aspect_critic.md) 使用 `DiscreteMetric` 评估输出的任意指定维度。

```python
import asyncio
from openai import AsyncOpenAI
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory

# Setup your evaluator LLM
client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o", client=client)

# Create a custom aspect evaluator
metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="""Evaluate if the summary is accurate and captures key information.

Response: {response}

Answer with only 'accurate' or 'inaccurate'."""
)

# Score your application's output
async def main():
    score = await metric.ascore(
        llm=evaluator_llm,
        response="The summary of the text is..."
    )
    print(f"Score: {score.value}")  # 'accurate' or 'inaccurate'
    print(f"Reason: {score.reason}")


if __name__ == "__main__":
    asyncio.run(main())
```

这些内置指标可以帮助你省去从头编写评估逻辑的工作。查看[所有可用指标](../concepts/metrics/available_metrics/index.md)。

!!! info
    `ragas` 还提供许多其他指标，包括需要 `reference` 的指标和不需要参考答案的指标。如果现有指标无法满足需求，也可以创建自己的指标。详情请阅读[评估指标](../concepts/metrics/index.md)。

### 在数据集上评估 { #evaluating-on-a-dataset }

快速入门项目中的 `load_dataset()` 函数会创建包含多个样本的测试数据。

```python
from ragas import Dataset

# Create a dataset with multiple test samples
dataset = Dataset(
    name="test_dataset",
    backend="local/csv",  # Can also use JSONL, Google Drive, or in-memory
    root_dir=".",
)

# Add samples to the dataset
data_samples = [
    {
        "user_input": "What is ragas?",
        "response": "Ragas is an evaluation framework...",
        "expected": "Ragas provides objective metrics..."
    },
    {
        "user_input": "How do metrics work?",
        "response": "Metrics score your application...",
        "expected": "Metrics evaluate performance..."
    },
]

for sample in data_samples:
    dataset.append(sample)

# Save to disk
dataset.save()
```

这样可以一次评估多个测试用例。进一步了解[数据集与实验](../concepts/components/eval_dataset.md)。

生成项目的 `evals/datasets/` 目录包含示例数据。你可以编辑这些文件，添加更多测试用例。

### 希望通过评估改进 AI 应用？ { #want-help-in-improving-your-ai-application-using-evals }

过去两年，我们通过评估帮助许多 AI 应用提升了质量。

我们正在将这些经验融入产品，让评估闭环取代凭感觉判断，使你能够专注于构建出色的 AI 应用。

如果你希望通过评估改进 AI 应用并扩大其应用规模，欢迎联系我们。


🔗 [预约交流](https://bit.ly/3EBYq4J)，或发送邮件至 [founders@vibrantlabs.com](mailto:founders@vibrantlabs.com)。


![](../../_static/ragas_app.gif)




## 下一步 { #up-next }

- [评估简单的 RAG 应用](rag_eval.md)
