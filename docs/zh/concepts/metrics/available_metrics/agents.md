# 智能体与工具使用 { #agentic-or-tool-use }

智能体或工具使用工作流可以从多个维度评估。下面介绍衡量智能体或工具在特定任务中表现的指标。


## 主题遵循度 { #topic-adherence }

真实应用中的 AI 系统通常需要在指定领域内回答问题，但 LLM 有时会忽略这一限制，回答领域外的通用问题。主题遵循度衡量 AI 在交互中是否遵守预定义的主题范围，尤其适用于只允许处理特定领域问题的对话系统。

`TopicAdherence` 需要通过 `reference_topics` 提供允许讨论的主题集合，同时传入 `user_input`。该指标支持计算精确率、召回率和 F1 得分，定义如下。

$$
\text{Precision } = {|\text{Queries that are answered and are adheres to any present reference topics}| \over |\text{Queries that are answered and are adheres to any present reference topics}| + |\text{Queries that are answered and do not adheres to any present reference topics}|}
$$

$$
\text{Recall } = {|\text{Queries that are answered and are adheres to any present reference topics}| \over |\text{Queries that are answered and are adheres to any present reference topics}| + |\text{Queries that were refused and should have been answered}|}
$$

$$
\text{F1 Score } = {2 \times \text{Precision} \times \text{Recall} \over \text{Precision} + \text{Recall}}
$$

### 示例 { #example }

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import TopicAdherence
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall


async def evaluate_topic_adherence():
    # Setup LLM
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    user_input = [
        HumanMessage(
            content="Can you provide me with details about Einstein's theory of relativity?"
        ),
        AIMessage(
            content="Sure, let me retrieve the relevant information for you.",
            tool_calls=[
                ToolCall(
                    name="document_search",
                    args={"query": "Einstein's theory of relativity"},
                )
            ],
        ),
        ToolMessage(
            content="Found relevant documents: 1. Relativity: The Special and the General Theory, 2. General Theory of Relativity by A. Einstein."
        ),
        AIMessage(
            content="I found some documents on Einstein's theory of relativity. Which one would you like to know more about: 'Relativity: The Special and the General Theory' or 'General Theory of Relativity by A. Einstein'?"
        ),
        HumanMessage(content="Tell me about the 'General Theory of Relativity'."),
        AIMessage(
            content="Got it! Let me fetch more details from 'General Theory of Relativity by A. Einstein'.",
            tool_calls=[
                ToolCall(
                    name="document_retrieve",
                    args={"document": "General Theory of Relativity by A. Einstein"},
                )
            ],
        ),
        ToolMessage(
            content="The document discusses how gravity affects the fabric of spacetime, describing the relationship between mass and spacetime curvature."
        ),
        AIMessage(
            content="The 'General Theory of Relativity' explains how gravity affects the fabric of spacetime and the relationship between mass and spacetime curvature. Would you like more details or a specific explanation?"
        ),
        HumanMessage(
            content="No, that's perfect. By the way, do you know any good recipes for a chocolate cake?"
        ),
        AIMessage(
            content="Sure! Let me find a simple and delicious recipe for a chocolate cake.",
            tool_calls=[
                ToolCall(name="recipe_search", args={"query": "chocolate cake recipe"})
            ],
        ),
        ToolMessage(
            content="Here's a popular recipe for a chocolate cake: Ingredients include flour, sugar, cocoa powder, eggs, milk, and butter. Instructions: Mix dry ingredients, add wet ingredients, and bake at 350°F for 30-35 minutes."
        ),
        AIMessage(
            content="I found a great recipe for chocolate cake! Would you like the full details, or is that summary enough?"
        ),
    ]

    # Evaluate with precision mode
    metric = TopicAdherence(llm=llm, mode="precision")
    result = await metric.ascore(
        user_input=user_input,
        reference_topics=["science"],
    )
    print(f"Topic Adherence (precision): {result.value}")


if __name__ == "__main__":
    asyncio.run(evaluate_topic_adherence())
```
输出如下。
```
Topic Adherence (precision): 0.6666666666444444
```


要计算召回率，将 `mode` 参数设为 `recall`。

```python
metric = TopicAdherence(llm=llm, mode="recall")
```
输出如下。
```
0.99999999995
```

### 旧版 API（已弃用） { #legacy-api-deprecated }

!!! warning "弃用说明"
    `ragas.metrics` 中的旧版 `TopicAdherenceScore` 已弃用，将在 v1.0 移除。请迁移到 `ragas.metrics.collections.TopicAdherence`，通过新版 API 获得相同功能。

旧版 API 仍可使用，但需要 `MultiTurnSample`。

```python
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.metrics import TopicAdherenceScore  # Legacy import
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

sample = MultiTurnSample(
    user_input=[...],  # conversation messages
    reference_topics=["science"],
)
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")
score = await scorer.multi_turn_ascore(sample)
```



## 工具调用准确率 { #tool-call-accuracy }

`ToolCallAccuracy` 比较 LLM 智能体的实际工具调用与预期调用，评估调用顺序和参数是否准确，适合验证多步骤工作流中是否使用了正确的工具与参数。

此指标需要 `user_input`（对话消息）和 `reference_tool_calls`（预期工具调用）。得分为 0～1，越高越好。

### 主要功能 { #key-features }

**两种评估模式**

1. **严格顺序（默认）**。工具调用顺序必须完全匹配。
    - 适用于顺序重要的串行工作流
    - 例如，必须先搜索，再筛选结果

2. **灵活顺序**。工具调用顺序可以不同。
    - 适用于顺序无关的并行操作
    - 例如，同时查询多座城市的天气

**评分方式**

- 检查调用序列是否对齐，即是否按正确顺序使用正确工具
- 检查参数准确性，即每个工具的参数是否正确
- 最终得分 = 参数准确率 ×（序列对齐时为 1，否则为 0）

### 示例，基本用法 { #example-basic-usage }

```python
import asyncio
from ragas.metrics.collections import ToolCallAccuracy
from ragas.messages import AIMessage, HumanMessage, ToolCall

async def evaluate_tool_call_accuracy():
    # Define the conversation with tool calls
    user_input = [
        HumanMessage(content="What's the weather like in New York right now?"),
        AIMessage(
            content="The current temperature in New York is 75°F and it's partly cloudy.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "New York"})],
        ),
        HumanMessage(content="Can you translate that to Celsius?"),
        AIMessage(
            content="Let me convert that to Celsius for you.",
            tool_calls=[
                ToolCall(
                    name="temperature_conversion", args={"temperature_fahrenheit": 75}
                )
            ],
        ),
    ]

    # Define expected tool calls
    reference_tool_calls = [
        ToolCall(name="weather_check", args={"location": "New York"}),
        ToolCall(name="temperature_conversion", args={"temperature_fahrenheit": 75}),
    ]

    # Evaluate
    metric = ToolCallAccuracy()
    result = await metric.ascore(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )
    print(f"Tool Call Accuracy: {result.value}")

if __name__ == "__main__":
    asyncio.run(evaluate_tool_call_accuracy())
```
输出如下。
```
Tool Call Accuracy: 1.0
```

### 示例，灵活顺序模式 { #example-flexible-order-mode }

适用于工具调用可以并行执行的场景。

```python
# Enable flexible order mode
metric = ToolCallAccuracy(strict_order=False)

user_input = [
    HumanMessage(content="Get weather for Paris and London"),
    AIMessage(
        content="Fetching weather data...",
        tool_calls=[
            ToolCall(name="weather_check", args={"location": "London"}),
            ToolCall(name="weather_check", args={"location": "Paris"}),
        ],
    ),
]

reference_tool_calls = [
    ToolCall(name="weather_check", args={"location": "Paris"}),
    ToolCall(name="weather_check", args={"location": "London"}),
]

result = await metric.ascore(
    user_input=user_input,
    reference_tool_calls=reference_tool_calls,
)
print(f"Score: {result.value}")  # 1.0 (order doesn't matter)
```

### 评分示例 { #scoring-examples }

**完全匹配**
```python
# All tools called correctly with correct arguments
Expected: [weather_check(location="Paris"), translate(text="hello")]
Got:      [weather_check(location="Paris"), translate(text="hello")]
Score: 1.0
```

**参数部分匹配**
```python
# Some arguments incorrect
Expected: [search(query="python", limit=10, sort="date")]
Got:      [search(query="python", limit=10, sort="relevance")]
Score: 0.66 (2 out of 3 arguments match)
```

**顺序错误（严格模式）**
```python
# Correct tools but wrong sequence
Expected: [search(...), filter(...)]
Got:      [filter(...), search(...)]
Score: 0.0 (sequence not aligned)
```

### 使用场景 { #use-cases }

1. **智能体验证**，测试智能体是否正确使用工具
2. **回归测试**，确保改动后工具调用能力未退化
3. **多步骤工作流**，验证复杂的串行操作
4. **工具选择**，检查智能体是否从多个候选中选择正确工具

### 如何选择指标 { #when-to-use-different-metrics }

| 指标 | 适用情况 |
|--------|----------|
| **ToolCallAccuracy** | 需要精确检查调用顺序和参数 |
| **ToolCallF1** | 需要工具调用的精确率、召回率等度量 |
| **AgentGoalAccuracy** | 主要关心最终目标是否达成 |

**例如**，对于“帮我订一张去巴黎的机票”，如果只关心预订是否成功，可以使用 `AgentGoalAccuracyWithReference`。

### 旧版 API（已弃用） { #legacy-api-deprecated_1 }

!!! warning "弃用说明"
    `ragas.metrics` 中的旧版 `ToolCallAccuracy` 已弃用，将在 v1.0 移除。请迁移到 `ragas.metrics.collections.ToolCallAccuracy`，通过新版 API 获得相同功能。

旧版 API 仍可使用，但需要 `MultiTurnSample`。

```python
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics import ToolCallAccuracy  # Legacy import

sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="What's the weather in New York?"),
        AIMessage(
            content="Checking weather...",
            tool_calls=[ToolCall(name="weather_check", args={"location": "New York"})],
        ),
    ],
    reference_tool_calls=[
        ToolCall(name="weather_check", args={"location": "New York"}),
    ],
)

scorer = ToolCallAccuracy()
score = await scorer.multi_turn_ascore(sample)
```

旧版实现还支持自定义参数比较指标。

```python
from ragas.metrics._string import NonLLMStringSimilarity
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy

metric = ToolCallAccuracy()
metric.arg_comparison_metric = NonLLMStringSimilarity()
```

## 工具调用 F1 { #tool-call-f1 }

`ToolCallF1` 将智能体的工具调用与预期调用（`reference_tool_calls`）比较，根据精确率和召回率计算 F1 得分。它提供较宽松的评估，适合初期接入和迭代，量化智能体的行为与预期有多接近，即使存在多调用或少调用的情况也能衡量。

### 公式 { #formula }

ToolCallF1 基于经典信息检索指标，采用无序匹配。工具调用顺序不影响结果，只考虑工具名称与参数是否出现且正确。

$$
\text{Precision} = \frac{\text{tool calls that match both name and parameters}}{\text{tool calls that match both name and parameters} + \text{extra tool calls that were not expected}}
$$

$$
\text{Recall} = \frac{\text{tool calls that match both name and parameters}}{\text{tool calls that match both name and parameters} + \text{expected tool calls that were not made}}
$$

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### 与主题遵循度有何区别？ { #how-is-it-different-from-topic-adherence }

`ToolCallF1` 与 `TopicAdherenceScore` 都使用精确率、召回率和 F1，但衡量的维度不同。

| 指标 | 评估内容 | 依据 |
| --------------------- | --------------------------------------- | ---------------------------- |
| `ToolCallF1` | 工具执行是否正确 | 结构化工具调用对象 |
| `TopicAdherenceScore` | 对话是否遵循主题 | 领域主题比较 |

检查智能体是否正确**执行工具**时，使用 `ToolCallF1`。检查对话的**内容或意图**是否处于允许的主题范围内时，使用 `TopicAdherenceScore`。

### 示例，基本用法 { #example-basic-usage_1 }

```python
import asyncio
from ragas.metrics.collections import ToolCallF1
from ragas.messages import HumanMessage, AIMessage, ToolCall

async def evaluate_tool_call_f1():
    # Define the conversation with tool calls
    user_input = [
        HumanMessage(content="What's the weather like in Paris today?"),
        AIMessage(
            content="Let me check that for you.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "Paris"})],
        ),
        HumanMessage(content="And the UV index?"),
        AIMessage(
            content="Sure, here's the UV index for Paris.",
            tool_calls=[ToolCall(name="uv_index_lookup", args={"location": "Paris"})],
        ),
    ]

    # Define expected tool calls
    reference_tool_calls = [
        ToolCall(name="weather_check", args={"location": "Paris"}),
        ToolCall(name="uv_index_lookup", args={"location": "Paris"}),
    ]

    # Evaluate
    metric = ToolCallF1()
    result = await metric.ascore(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )
    print(f"Tool Call F1: {result.value}")

if __name__ == "__main__":
    asyncio.run(evaluate_tool_call_f1())
```

输出如下。
```
Tool Call F1: 1.0
```

### 示例，额外调用工具 { #example-extra-tool-called }

智能体调用了参考列表之外的额外工具时，

```python
user_input = [
    HumanMessage(content="What's the weather like in Paris today?"),
    AIMessage(
        content="Let me check that for you.",
        tool_calls=[ToolCall(name="weather_check", args={"location": "Paris"})],
    ),
    HumanMessage(content="And the UV index?"),
    AIMessage(
        content="Sure, here's the UV index and air quality for Paris.",
        tool_calls=[
            ToolCall(name="uv_index_lookup", args={"location": "Paris"}),
            ToolCall(name="air_quality", args={"location": "Paris"}),  # extra call
        ],
    ),
]

reference_tool_calls = [
    ToolCall(name="weather_check", args={"location": "Paris"}),
    ToolCall(name="uv_index_lookup", args={"location": "Paris"}),
]

result = await metric.ascore(
    user_input=user_input,
    reference_tool_calls=reference_tool_calls,
)
print(f"F1 Score: {result.value}")
```

输出如下。
```
F1 Score: 0.8
```

在本例中，
- TP = 2（weather_check、uv_index_lookup）
- FP = 1（air_quality）
- FN = 0
- 精确率 = 2/3 ≈ 0.67，召回率 = 2/2 = 1.0，按标准 F1 公式计算为 0.8。

### 评分示例 { #scoring-examples_1 }

**完全匹配**
```python
# All tools called correctly
Reference: [weather_check(location="Paris"), uv_index_lookup(location="Paris")]
Got:       [weather_check(location="Paris"), uv_index_lookup(location="Paris")]
F1 Score: 1.0
```

**遗漏工具调用**
```python
# One expected tool not called
Reference: [weather_check(...), uv_index_lookup(...)]
Got:       [weather_check(...)]
F1 Score: 0.67 (TP=1, FP=0, FN=1)
```

**参数错误**
```python
# Tool name matches but args differ
Reference: [weather_check(location="Paris")]
Got:       [weather_check(location="London")]
F1 Score: 0.0 (no match, arguments must be exact)
```

### 旧版 API（已弃用） { #legacy-api-deprecated_2 }

!!! warning "弃用说明"
    `ragas.metrics` 中的旧版 `ToolCallF1` 已弃用，将在 v1.0 移除。请迁移到 `ragas.metrics.collections.ToolCallF1`，通过新版 API 获得相同功能。

旧版 API 仍可使用，但需要 `MultiTurnSample`。

```python
from ragas.metrics import ToolCallF1  # Legacy import
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolCall

sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="What's the weather like in Paris today?"),
        AIMessage(
            content="Let me check that for you.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "Paris"})],
        ),
    ],
    reference_tool_calls=[
        ToolCall(name="weather_check", args={"location": "Paris"}),
    ],
)

scorer = ToolCallF1()
score = await scorer.multi_turn_ascore(sample)
```


## 智能体目标达成准确率 { #agent-goal-accuracy }


智能体目标达成准确率衡量 LLM 能否识别并完成用户目标。这是二元指标，1 表示 AI 已完成目标，0 表示未完成。

### 有参考结果 { #with-reference }

`AgentGoalAccuracyWithReference` 比较工作流的最终状态与给定参考结果，判断智能体是否完成用户目标。参考结果表示预期或理想的结果。

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import AgentGoalAccuracyWithReference
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage


async def evaluate_agent_goal_accuracy_with_reference():
    # Setup LLM
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    user_input = [
        HumanMessage(
            content="Hey, book a table at the nearest best Chinese restaurant for 8:00pm"
        ),
        AIMessage(
            content="Sure, let me find the best options for you.",
            tool_calls=[
                ToolCall(
                    name="restaurant_search",
                    args={"cuisine": "Chinese", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(
            content="Found a few options: 1. Golden Dragon, 2. Jade Palace"
        ),
        AIMessage(
            content="I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?"
        ),
        HumanMessage(content="Let's go with Golden Dragon."),
        AIMessage(
            content="Great choice! I'll book a table for 8:00pm at Golden Dragon.",
            tool_calls=[
                ToolCall(
                    name="restaurant_book",
                    args={"name": "Golden Dragon", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(content="Table booked at Golden Dragon for 8:00pm."),
        AIMessage(
            content="Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!"
        ),
        HumanMessage(content="thanks"),
    ]

    metric = AgentGoalAccuracyWithReference(llm=llm)
    result = await metric.ascore(
        user_input=user_input,
        reference="Table booked at one of the chinese restaurants at 8 pm",
    )
    print(f"Agent Goal Accuracy: {result.value}")


if __name__ == "__main__":
    asyncio.run(evaluate_agent_goal_accuracy_with_reference())
```
输出如下。
```
Agent Goal Accuracy: 1.0
```

### 无参考结果 { #without-reference }

`AgentGoalAccuracyWithoutReference` 无须参考结果。它从对话中分别推断用户目标与实际完成情况，再比较两者。

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import AgentGoalAccuracyWithoutReference
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage


async def evaluate_agent_goal_accuracy_without_reference():
    # Setup LLM
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    user_input = [
        HumanMessage(
            content="Hey, book a table at the nearest best Chinese restaurant for 8:00pm"
        ),
        AIMessage(
            content="Sure, let me find the best options for you.",
            tool_calls=[
                ToolCall(
                    name="restaurant_search",
                    args={"cuisine": "Chinese", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(
            content="Found a few options: 1. Golden Dragon, 2. Jade Palace"
        ),
        AIMessage(
            content="I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?"
        ),
        HumanMessage(content="Let's go with Golden Dragon."),
        AIMessage(
            content="Great choice! I'll book a table for 8:00pm at Golden Dragon.",
            tool_calls=[
                ToolCall(
                    name="restaurant_book",
                    args={"name": "Golden Dragon", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(content="Table booked at Golden Dragon for 8:00pm."),
        AIMessage(
            content="Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!"
        ),
        HumanMessage(content="thanks"),
    ]

    metric = AgentGoalAccuracyWithoutReference(llm=llm)
    result = await metric.ascore(user_input=user_input)
    print(f"Agent Goal Accuracy: {result.value}")


if __name__ == "__main__":
    asyncio.run(evaluate_agent_goal_accuracy_without_reference())
```
输出如下。
```
Agent Goal Accuracy: 1.0
```

### 旧版 API（已弃用） { #legacy-api-deprecated_3 }

!!! warning "弃用说明"
    `ragas.metrics` 中的旧版 `AgentGoalAccuracyWithReference` 和 `AgentGoalAccuracyWithoutReference` 已弃用，将在 v1.0 移除。请迁移到 `ragas.metrics.collections`，通过新版 API 获得相同功能。

旧版 API 仍可使用，但需要 `MultiTurnSample`。

```python
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.metrics import AgentGoalAccuracyWithReference  # Legacy import
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

sample = MultiTurnSample(
    user_input=[...],  # conversation messages
    reference="Table booked at one of the chinese restaurants at 8 pm",
)
scorer = AgentGoalAccuracyWithReference(llm=evaluator_llm)
score = await scorer.multi_turn_ascore(sample)
```
