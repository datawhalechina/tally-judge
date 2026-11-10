# AG-UI 集成 { #ag-ui-integration }
Ragas 可以评估通过 [AG-UI 协议](https://docs.ag-ui.com/)流式发送事件的智能体。本教程使用新版 `@experiment` 装饰器，介绍数据集构建、指标配置与端点评分。

## 前置条件 { #prerequisites }
- 安装依赖，运行 `pip install "ragas[ag-ui]" python-dotenv nest_asyncio`
- 在本地启动兼容 AG-UI 的智能体，例如 Google ADK、PydanticAI 或 CrewAI
- 创建 `.env`，配置评估模型凭据，例如 `OPENAI_API_KEY`、`GOOGLE_API_KEY`
- 在 notebook 中调用 `nest_asyncio.apply()`，以便直接 `await` 协程


```python
# !pip install "ragas[ag-ui]" python-dotenv nest_asyncio
```

## 导入与环境配置 { #imports-and-environment-setup }
加载环境变量，并导入本教程使用的类。


```python
import json

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display

from ragas.dataset import Dataset
from ragas.messages import HumanMessage

load_dotenv()
# Patch the existing notebook loop so we can await coroutines safely
nest_asyncio.apply()
```

## 构建单轮实验数据 { #build-single-turn-experiment-data }
只评估最终答案时，通过 `Dataset.from_pandas()` 创建包含 `user_input` 与 `reference` 的样本。


```python
scientist_questions = Dataset.from_pandas(
    pd.DataFrame(
        [
            {
                "user_input": "Who originated the theory of relativity?",
                "reference": "Albert Einstein originated the theory of relativity.",
            },
            {
                "user_input": "Who discovered penicillin and when?",
                "reference": "Alexander Fleming discovered penicillin in 1928.",
            },
        ]
    ),
    name="scientist_questions",
    backend="inmemory",
)

scientist_questions
```

## 构建多轮对话 { #build-multi-turn-conversations }

工具使用与目标达成评估需要以下内容。
- `reference_tool_calls`，JSON 格式的预期工具调用，用于 `ToolCallF1`
- `reference`，预期结果描述，用于 `AgentGoalAccuracyWithReference`


```python
weather_queries = Dataset.from_pandas(
    pd.DataFrame(
        [
            {
                "user_input": [HumanMessage(content="What's the weather in Paris?")],
                "reference_tool_calls": json.dumps(
                    [{"name": "get_weather", "args": {"location": "Paris"}}]
                ),
                # Expected outcome - phrased to match what LLM extracts as end_state
                "reference": "The AI provided the current weather conditions for Paris.",
            },
            {
                "user_input": [
                    HumanMessage(content="Is it raining in London right now?")
                ],
                "reference_tool_calls": json.dumps(
                    [{"name": "get_weather", "args": {"location": "London"}}]
                ),
                "reference": "The AI provided the current weather conditions for London.",
            },
        ]
    ),
    name="weather_queries",
    backend="inmemory",
)

weather_queries
```

## 配置指标与评估 LLM { #configure-metrics-and-the-evaluator-llm }

单轮问答使用以下指标。
- `FactualCorrectness`，对照参考答案检查回答事实
- `AnswerRelevancy`，衡量回答与问题的相关性
- `DiscreteMetric`，自定义简洁性指标

多轮智能体实验使用以下指标。
- `ToolCallF1`，基于规则比较实际与预期工具调用
- `AgentGoalAccuracyWithReference`，使用 LLM 判断是否完成用户目标


```python
from openai import AsyncOpenAI

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    AgentGoalAccuracyWithReference,
    AnswerRelevancy,
    FactualCorrectness,
    ToolCallF1,
)

# Async client for evaluator prompts
async_llm_client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o-mini", client=async_llm_client)

embedding_client = AsyncOpenAI()
evaluator_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=embedding_client,
    interface="modern",
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

# Metrics for single-turn Q&A experiments
qa_metrics = [
    FactualCorrectness(
        llm=evaluator_llm,
        mode="f1",
        atomicity="high",
        coverage="high",
    ),
    AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        strictness=2,
    ),
    conciseness_metric,
]

# Metrics for multi-turn agent experiments
# - ToolCallF1: Rule-based metric for tool call accuracy
# - AgentGoalAccuracyWithReference: LLM-based metric for goal achievement
tool_metrics = [
    ToolCallF1(),
    AgentGoalAccuracyWithReference(llm=evaluator_llm),
]
```

## 对运行中的 AG-UI 端点执行实验 { #run-experiments-against-a-live-ag-ui-endpoint }
设置智能体端点 URL。`run_ag_ui_row()` 调用端点并返回补充后的样本数据，结合 `@experiment` 构建流水线。

准备好后切换示例中的运行标志。Jupyter/IPython 中执行 `nest_asyncio.apply()` 后，可以直接 await 实验。


```python
AG_UI_ENDPOINT = "http://localhost:8000"  # Update to match your agent

RUN_FACTUAL_EXPERIMENT = True
RUN_TOOL_EXPERIMENT = True
```


```python
from ragas import experiment
from ragas.integrations.ag_ui import run_ag_ui_row


@experiment()
async def factual_experiment(row):
    """Single-turn Q&A experiment with factual correctness scoring."""
    # Call AG-UI endpoint and get enriched row
    enriched = await run_ag_ui_row(row, AG_UI_ENDPOINT, metadata=True)

    # Score with factual correctness metric
    fc_result = await qa_metrics[0].ascore(
        response=enriched["response"],
        reference=row["reference"],
    )

    # Score with answer relevancy metric
    ar_result = await qa_metrics[1].ascore(
        user_input=row["user_input"],
        response=enriched["response"],
    )

    # Score with conciseness metric
    concise_result = await conciseness_metric.ascore(
        response=enriched["response"],
        llm=evaluator_llm,
    )

    return {
        **enriched,
        "factual_correctness": fc_result.value,
        "answer_relevancy": ar_result.value,
        "conciseness": concise_result.value,
    }


if RUN_FACTUAL_EXPERIMENT:
    # Run the experiment against the dataset
    factual_result = await factual_experiment.arun(
        scientist_questions, name="scientist_qa_experiment"
    )
    display(factual_result.to_pandas())
```


```python
from ragas.messages import ToolCall


@experiment()
async def tool_experiment(row):
    """Multi-turn experiment with tool call and goal accuracy scoring."""
    # Call AG-UI endpoint and get enriched row
    enriched = await run_ag_ui_row(row, AG_UI_ENDPOINT)

    # Parse reference_tool_calls from JSON string (e.g., from CSV)
    ref_tool_calls_raw = row.get("reference_tool_calls")
    if isinstance(ref_tool_calls_raw, str):
        ref_tool_calls = [ToolCall(**tc) for tc in json.loads(ref_tool_calls_raw)]
    else:
        ref_tool_calls = ref_tool_calls_raw or []

    # Score with tool metrics using the modern collections API
    f1_result = await tool_metrics[0].ascore(
        user_input=enriched["messages"],
        reference_tool_calls=ref_tool_calls,
    )
    goal_result = await tool_metrics[1].ascore(
        user_input=enriched["messages"],
        reference=row.get("reference", ""),
    )

    return {
        **enriched,
        "tool_call_f1": f1_result.value,
        "agent_goal_accuracy": goal_result.value,
    }


if RUN_TOOL_EXPERIMENT:
    # Run the experiment against the dataset
    tool_result = await tool_experiment.arun(
        weather_queries, name="weather_tool_experiment"
    )
    display(tool_result.to_pandas())
```

## 进阶，底层控制 { #advanced-lower-level-control }

建议优先使用 `run_ag_ui_row()`。需要更多控制时，可以直接调用底层 `call_ag_ui_endpoint()`。

底层方式支持以下定制。
- 自定义事件处理
- 为每行设置端点配置
- 实现自定义消息处理
- 添加日志与调试信息


```python
from ragas.integrations.ag_ui import (
    call_ag_ui_endpoint,
    convert_to_ragas_messages,
    extract_response,
)


@experiment()
async def custom_ag_ui_experiment(row):
    """
    Custom experiment function with full control over endpoint calls.
    """
    # Call the AG-UI endpoint directly (lower-level than run_ag_ui_row)
    events = await call_ag_ui_endpoint(
        endpoint_url=AG_UI_ENDPOINT,
        user_input=row["user_input"],
        timeout=60.0,
    )

    # Convert AG-UI events to Ragas messages
    messages = convert_to_ragas_messages(events, metadata=True)

    # Extract response using helper (or custom logic)
    response = extract_response(messages)

    # Score with a custom metric
    score_result = await conciseness_metric.ascore(
        response=response,
        llm=evaluator_llm,
    )

    # Return result with custom fields
    return {
        **row,
        "response": response or "[No response]",
        "message_count": len(messages),
        "conciseness": score_result.value,
    }
```

在数据集上运行自定义实验。`@experiment` 提供 `.arun()`，支持并行执行和自动收集结果。


```python
RUN_CUSTOM_EXPERIMENT = True

if RUN_CUSTOM_EXPERIMENT:
    # Run the custom experiment
    custom_result = await custom_ag_ui_experiment.arun(
        scientist_questions, name="custom_ag_ui_experiment"
    )
    display(custom_result.to_pandas())
```

### API 比较 { #api-comparison }

| API 层级 | 函数 | 使用场景 |
|-----------|----------|-------------|
| 高层 | `run_ag_ui_row()` | 标准实验，自动处理调用、转换与提取 |
| 底层 | `call_ag_ui_endpoint()` + `convert_to_ragas_messages()` | 自定义事件处理、逐行端点配置和进阶调试 |

两种方式都支持 `@experiment`，可根据所需控制程度选择。
