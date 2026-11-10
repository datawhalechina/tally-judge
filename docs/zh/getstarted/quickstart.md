# 快速开始，迅速运行你的评估 { #quick-start-get-evaluations-running-in-a-flash }

只需几条命令，就能创建完整的评估项目，几分钟上手 Ragas。

## 第一步，创建项目 { #step-1-create-your-project }

选择以下任一方式。

=== "uvx（推荐）"
    无须预先安装。`uvx` 会自动下载并运行 ragas。

    ```sh
    uvx ragas quickstart rag_eval
    cd rag_eval
    ```

=== "先安装 Ragas"
    先安装 ragas，再创建项目。

    ```sh
    pip install ragas
    ragas quickstart rag_eval
    cd rag_eval
    ```

## 第二步，安装依赖 { #step-2-install-dependencies }

安装项目依赖。

```sh
uv sync
```

也可以使用 `pip`。

```sh
pip install -e .
```

## 第三步，设置 API 密钥 { #step-3-set-your-api-key }

快速入门示例默认使用 OpenAI，设置 API 密钥后即可运行。只需稍作修改，也能使用其他服务商。

=== "OpenAI（默认）"
    ```sh
    export OPENAI_API_KEY="your-openai-key"
    ```

    快速入门项目已配置为使用 OpenAI，可以直接运行。

=== "Anthropic Claude"
    设置 Anthropic API 密钥。

    ```sh
    export ANTHROPIC_API_KEY="your-anthropic-key"
    ```

    然后修改 `evals.py` 中的 LLM 初始化代码。

    ```python
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

    然后修改 `evals.py` 中的 LLM 初始化代码。

    **方式一，使用 Google 官方库（推荐）**

    ```python
    import google.generativeai as genai
    from ragas.llms import llm_factory

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    client = genai.GenerativeModel("gemini-2.0-flash")
    llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
    # Adapter is auto-detected as "litellm" for google provider
    ```

    更多 Gemini 选项和详细配置，请参阅 [Google Gemini 集成指南](../howtos/integrations/gemini.md)。

=== "本地模型（Ollama）"
    在本地安装并运行 Ollama，然后修改 `evals.py` 中的 LLM 初始化代码。

    ```python
    from openai import OpenAI
    from ragas.llms import llm_factory

    # Create an OpenAI-compatible client for Ollama
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

## 项目结构 { #project-structure }

生成的项目包含以下内容。

```sh
rag_eval/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── rag.py                 # Your RAG application
├── evals.py               # Evaluation workflow
├── __init__.py            # Makes this a Python package
└── evals/
    ├── datasets/          # Test data files
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## 第四步，运行评估 { #step-4-run-your-evaluation }

运行评估脚本。

```sh
uv run python evals.py
```

如果使用 `pip` 安装，也可以这样运行。

```sh
python evals.py
```

评估过程会依次完成以下操作。
- 从 `evals.py` 的 `load_dataset()` 函数加载测试数据
- 使用测试问题查询 RAG 应用
- 评估回答
- 在控制台展示结果
- 将结果保存为 CSV 文件，存放在 `evals/experiments/` 目录中

![](../../_static/imgs/results/rag_eval_result.png)

完整的评估流程已经运行起来了！🎉

---

## 定制评估 { #customize-your-evaluation }

### 添加测试用例 { #add-more-test-cases }

编辑 `evals.py` 中的 `load_dataset()` 函数，添加更多测试问题。

```python
from ragas import Dataset

def load_dataset():
    """Load test dataset for evaluation."""
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )

    data_samples = [
        {
            "question": "What is Ragas?",
            "grading_notes": "Ragas is an evaluation framework for LLM applications",
        },
        {
            "question": "How do metrics work?",
            "grading_notes": "Metrics evaluate the quality and performance of LLM responses",
        },
        # Add more test cases here
    ]

    for sample in data_samples:
        dataset.append(sample)

    dataset.save()
    return dataset
```

### 自定义评估指标 { #customize-evaluation-metrics }

模板提供了一个 `DiscreteMetric`，用于实现自定义评估逻辑。你可以通过以下方式调整评估。

1. **修改指标提示词**，调整评判标准
2. **调整允许值**，修改有效的输出类别
3. **添加更多指标**，从不同维度开展评估

下面是修改指标的示例。

```python
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory

my_metric = DiscreteMetric(
    name="custom_evaluation",
    prompt="Evaluate this response: {response} based on: {context}. Return 'excellent', 'good', or 'poor'.",
    allowed_values=["excellent", "good", "poor"],
)
```

## 下一步 { #whats-next }

- **学习基本概念**。阅读[评估简单的 LLM 应用](evals.md)，深入理解评估流程
- **自定义指标**。使用简单的装饰器[创建自己的指标](../concepts/metrics/overview/index.md#output-types)
- **集成到生产流程**。将[评估纳入 CI/CD 流水线](../howtos/index.md)
- **RAG 评估**。使用专门的指标评估 [RAG 系统](rag_eval.md)
- **智能体评估**。探索 [AI 智能体评估](../howtos/applications/text2sql.md)
- **测试数据生成**。为评估[生成合成测试数据集](rag_testset_generation.md)

## 获取帮助 { #getting-help }

- 📚 [完整文档](https://docs.ragas.io/)
- 💬 [加入 Discord 社区](https://discord.gg/5djav8GGNZ)
- 🐛 [反馈问题](https://github.com/vibrantlabsai/ragas/issues)
