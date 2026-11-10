# AG-UI

[AG-UI](https://docs.ag-ui.com/) 是将智能体更新流式传递给用户界面的事件协议，统一了消息、工具调用和状态事件，便于连接不同智能体运行时与可视化前端。`ragas.integrations.ag_ui` 可将这些事件流转换为 Ragas 消息，并通过新版 `@experiment` 装饰器对运行中的 AG-UI 端点评估。

本指南假设已有兼容 AG-UI 的智能体运行，例如使用 Google ADK、PydanticAI 或 CrewAI 构建，并且你已熟悉 Ragas 数据集。

## 安装集成 { #install-the-integration }

AG-UI 辅助功能作为可选扩展提供，请与评估 LLM 的依赖一起安装。在 Jupyter 或 IPython 中，还需安装 `nest_asyncio` 以复用事件循环。

```bash
pip install "ragas[ag-ui]" python-dotenv nest_asyncio
```

配置评估 LLM 的凭据，例如使用 OpenAI 时，

```bash
# .env
OPENAI_API_KEY=sk-...
```

运行示例前，在 Python 中加载环境变量。

```python
from dotenv import load_dotenv
import nest_asyncio

load_dotenv()

# If you're inside Jupyter/IPython, patch the running event loop once.
nest_asyncio.apply()
```

## 构建实验数据集 { #build-an-experiment-dataset }

`Dataset` 支持单轮或多轮样本，可以测试自由文本问答，也可以测试包含工具调用的长对话。

### 单轮样本 { #single-turn-samples }

只需评估最终答案文本时，通过 `Dataset.from_pandas()` 提供 `user_input` 和 `reference` 列。

```python
import pandas as pd
from ragas.dataset import Dataset

scientist_questions = Dataset.from_pandas(
    pd.DataFrame([
        {
            "user_input": "Who originated the theory of relativity?",
            "reference": "Albert Einstein originated the theory of relativity.",
        },
        {
            "user_input": "Who discovered penicillin and when?",
            "reference": "Alexander Fleming discovered penicillin in 1928.",
        },
    ]),
    name="scientist_questions",
    backend="inmemory",
)
```

### 包含预期工具调用的多轮样本 { #multi-turn-samples-with-tool-expectations }

需要评估中间行为，例如工具是否正确调用、用户目标是否完成时，将对话列表作为 `user_input`。预期工具调用用 JSON 提供，也可以附加目标准确率所需的参考结果。

```python
import json
import pandas as pd
from ragas.dataset import Dataset
from ragas.messages import HumanMessage

weather_queries = Dataset.from_pandas(
    pd.DataFrame([
        {
            "user_input": [HumanMessage(content="What's the weather in Paris?")],
            "reference_tool_calls": json.dumps([
                {"name": "get_weather", "args": {"location": "Paris"}}
            ]),
            # Expected outcome for AgentGoalAccuracyWithReference
            "reference": "The user received the current weather conditions for Paris.",
        },
        {
            "user_input": [HumanMessage(content="Is it raining in London right now?")],
            "reference_tool_calls": json.dumps([
                {"name": "get_weather", "args": {"location": "London"}}
            ]),
            "reference": "The user received the current weather conditions for London.",
        },
    ]),
    name="weather_queries",
    backend="inmemory",
)
```

### 从 CSV 加载 { #loading-from-csv }

大型数据集可以保存为 CSV，再通过 Dataset API 加载。

```python
from ragas.dataset import Dataset

dataset = Dataset.load(
    name="scientist_biographies",
    backend="local/csv",
    root_dir="./test_data",
)
```

## 选择指标与评估模型 { #choose-metrics-and-evaluator-model }

本集成支持 Ragas 各类指标。使用新版 collections 并加入自定义检查时，为评估提示词创建 Instructor 兼容的 LLM，并通过同步 OpenAI 客户端创建嵌入模型。

```python
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    AgentGoalAccuracyWithReference,
    AnswerRelevancy,
    FactualCorrectness,
    ToolCallF1,
)

async_llm_client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o-mini", client=async_llm_client)

# AnswerRelevancy's embeddings still run synchronously, so pair it with a sync client.
embedding_client = OpenAI()
evaluator_embeddings = embedding_factory(
    "openai", model="text-embedding-3-small", client=embedding_client, interface="modern"
)

conciseness_metric = DiscreteMetric(
    name="conciseness",
    allowed_values=["verbose", "concise"],
    prompt=(
        "Is the response concise and efficiently conveys information?\n\n"
        "Response: {response}\n\n"
        "Answer with only 'verbose' or 'concise'."
    ),
)

# Metrics for single-turn Q&A evaluation
qa_metrics = [
    FactualCorrectness(
        llm=evaluator_llm, mode="f1", atomicity="high", coverage="high"
    ),
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=2),
    conciseness_metric,
]

# Metrics for multi-turn agent evaluation
# - ToolCallF1: Rule-based metric for tool call accuracy
# - AgentGoalAccuracyWithReference: LLM-based metric for goal achievement
tool_metrics = [
    ToolCallF1(),
    AgentGoalAccuracyWithReference(llm=evaluator_llm),
]
```

## 使用 @experiment 运行实验 { #run-experiments-with-experiment }

`run_ag_ui_row()` 调用端点，并为每条数据补充智能体响应。结合 `@experiment` 即可构建评估流水线。

> ⚠️ 端点必须提供 AG-UI SSE 事件流，常见路径为 `/chat`、`/agent` 或 `/agentic_chat`。

### 基础单轮评估 { #basic-single-turn-evaluation }

在 Jupyter 或 IPython 中，执行 `nest_asyncio.apply()` 后使用顶层 `await`，避免事件循环已运行的错误。普通脚本可使用 `asyncio.run`。

```python
from ragas import experiment
from ragas.integrations.ag_ui import run_ag_ui_row
from ragas.metrics.collections import FactualCorrectness

@experiment()
async def factual_experiment(row):
    # Call AG-UI endpoint and get enriched row
    enriched = await run_ag_ui_row(row, "http://localhost:8000/chat")

    # Score with metrics
    score = await FactualCorrectness(llm=evaluator_llm).ascore(
        response=enriched["response"],
        reference=row["reference"],
    )

    return {**enriched, "factual_correctness": score.value}

# Run the experiment against the dataset
# In Jupyter/IPython (after calling nest_asyncio.apply())
factual_result = await factual_experiment.arun(
    scientist_questions,
    name="scientist_qa_eval"
)

# In a standalone script, use:
# factual_result = asyncio.run(factual_experiment.arun(scientist_questions, name="scientist_qa_eval"))

factual_result.to_pandas()
```

结果 DataFrame 包含逐样本得分、原始响应和检索上下文（工具结果）。实验框架会自动保存结果，也可以通过 pandas 导出 CSV。

### 多轮工具评估 { #multi-turn-tool-evaluation }

将对话消息和参考工具调用直接传给指标。

```python
import json
from ragas import experiment
from ragas.integrations.ag_ui import run_ag_ui_row
from ragas.messages import ToolCall
from ragas.metrics.collections import AgentGoalAccuracyWithReference, ToolCallF1

@experiment()
async def tool_experiment(row):
    # Call AG-UI endpoint and get enriched row
    enriched = await run_ag_ui_row(row, "http://localhost:8000/chat")

    # Parse reference_tool_calls from JSON string (e.g., from CSV)
    ref_tool_calls_raw = row.get("reference_tool_calls")
    if isinstance(ref_tool_calls_raw, str):
        ref_tool_calls = [ToolCall(**tc) for tc in json.loads(ref_tool_calls_raw)]
    else:
        ref_tool_calls = ref_tool_calls_raw or []

    # Score with tool metrics using the modern collections API
    f1_result = await ToolCallF1().ascore(
        user_input=enriched["messages"],
        reference_tool_calls=ref_tool_calls,
    )
    goal_result = await AgentGoalAccuracyWithReference(llm=evaluator_llm).ascore(
        user_input=enriched["messages"],
        reference=row.get("reference", ""),
    )

    return {
        **enriched,
        "tool_call_f1": f1_result.value,
        "agent_goal_accuracy": goal_result.value,
    }

# Run the experiment
# In Jupyter/IPython
tool_result = await tool_experiment.arun(
    weather_queries,
    name="weather_tool_eval"
)

# Or in a script
# tool_result = asyncio.run(tool_experiment.arun(weather_queries, name="weather_tool_eval"))

tool_result.to_pandas()
```

请求失败时，实验记录错误并为该样本返回占位值，继续处理剩余样本。

## 直接处理 AG-UI 事件 { #working-directly-with-ag-ui-events }

也可以单独收集运行记录或预发布环境中的事件日志，再离线评估。转换工具使用与 `run_ag_ui_row()` 相同的解析逻辑。

```python
from ragas.integrations.ag_ui import convert_to_ragas_messages
from ag_ui.core import TextMessageChunkEvent

events = [
    TextMessageChunkEvent(
        message_id="assistant-1",
        role="assistant",
        delta="Hello from AG-UI!",
        timestamp="2024-12-01T00:00:00Z",
    )
]

ragas_messages = convert_to_ragas_messages(events, metadata=True)
```

已有 `MessagesSnapshotEvent` 时，可以直接调用 `convert_messages_snapshot`。

```python
from ragas.integrations.ag_ui import convert_messages_snapshot
from ag_ui.core import MessagesSnapshotEvent, UserMessage, AssistantMessage

snapshot = MessagesSnapshotEvent(
    messages=[
        UserMessage(id="msg-1", content="Hello?"),
        AssistantMessage(id="msg-2", content="Hi! How can I help you today?"),
    ]
)

ragas_messages = convert_messages_snapshot(snapshot)
```

转换后的消息可用于自定义评估工作流，也可以直接传给指标评分函数。

## 数据提取工具 { #extraction-helpers }

以下辅助函数可从消息中提取所需数据。

```python
from ragas.integrations.ag_ui import (
    extract_response,    # Get concatenated AI response text
    extract_tool_calls,  # Get all tool calls from AI messages
    extract_contexts,    # Get tool results/contexts
)

messages = convert_to_ragas_messages(events)

response = extract_response(messages)      # "Hello! The weather is sunny."
tool_calls = extract_tool_calls(messages)  # [ToolCall(name="get_weather", args={"location": "SF"})]
contexts = extract_contexts(messages)      # ["Sunny, 72F in San Francisco"]
```

## 生产实验建议 { #tips-for-production-experiments }

- **自定义请求头**，通过 `run_ag_ui_row()` 的 `extra_headers` 传入身份验证 token 或租户 ID。
- **超时**，智能体工具耗时较长时调整 `timeout`。
- **元数据调试**，设置 `metadata=True`，保留 run、thread 和 message ID，便于追踪。
- **实验命名**，为 `.arun()` 的 `name` 使用描述性名称。

完整示例位于 `examples/ragas_examples/ag_ui_agent_experiments/experiments.py`，包含以下内容。

- 配置端点的 CLI 参数
- CSV 测试数据集
- 日志与错误处理
- 带时间戳的结果输出

交互式 notebook 位于 `howtos/integrations/ag_ui.ipynb`。

## API 参考 { #api-reference }

### 主要 API { #primary-api }

- **`run_ag_ui_row(row, endpoint_url, ...)`**，对端点运行单条样本，返回补充 response、messages、tool_calls 和 contexts 的数据。

### 转换函数 { #conversion-functions }

- **`convert_to_ragas_messages(events, metadata=False)`**，将 AG-UI 事件序列转换为 Ragas 消息
- **`convert_messages_snapshot(snapshot, metadata=False)`**，将消息快照转换为 Ragas 消息
- **`convert_messages_to_ag_ui(messages)`**，将 Ragas 消息转换为 AG-UI 格式

### 提取函数 { #extraction-helpers_1 }

- **`extract_response(messages)`**，提取并拼接 AI 回答文本
- **`extract_tool_calls(messages)`**，提取 AI 消息中的全部工具调用
- **`extract_contexts(messages)`**，提取工具结果或上下文

### 底层接口 { #low-level }

- **`call_ag_ui_endpoint(endpoint_url, user_input, ...)`**，调用 AG-UI 端点并收集事件流
- **`AGUIEventCollector`**，收集流式事件并重建消息
