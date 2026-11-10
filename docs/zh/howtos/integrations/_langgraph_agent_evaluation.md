# 构建并评估查询金属价格的 ReAct 智能体 { #building-and-evaluating-a-react-agent-for-fetching-metal-prices }

AI 智能体可自主调用 API、检索实时数据并完成用户任务，应用于金融、电商和客服等领域。评估有助于确认它们处理不同输入时的有效性与准确性。

本教程将完成以下工作。

1. 构建查询金属价格的 [ReAct 智能体](https://arxiv.org/abs/2210.03629)。
2. 配置评估流水线，跟踪关键指标。
3. 使用不同查询运行并评估智能体。

点击[此链接](https://colab.research.google.com/github/vibrantlabsai/ragas/blob/main/docs/howtos/integrations/langgraph_agent_evaluation.ipynb)，在 Google Colab 打开 notebook。

## 前置条件 { #prerequisites }
- Python 3.8+
- 了解 LangGraph、LangChain 与 LLM 的基础知识

## 安装 Ragas 与其他依赖 { #installing-ragas-and-other-dependencies }
使用 pip 安装 Ragas 和 LangGraph。


```python
%pip install langgraph==0.2.44
%pip install ragas
%pip install nltk
```

## 构建 ReAct 智能体 { #building-the-react-agent }

### 初始化外部组件 { #initializing-external-components }
配置外部组件有两种方式。

1. 使用真实 API 密钥。

    - 在 [metals.dev](https://metals.dev/) 注册，获取 API 密钥。

2. 模拟 API 响应。

    - 使用预定义 JSON 对象模拟 API 响应，方便快速体验。


根据需要选择一种方式。

### 使用预定义 JSON 模拟 API 响应 { #predefined-json-object-to-simulate-api-response }
希望快速开始时，可以直接使用以下 JSON 对象模拟响应。


```python
metal_price = {
    "gold": 88.1553,
    "silver": 1.0523,
    "platinum": 32.169,
    "palladium": 35.8252,
    "lbma_gold_am": 88.3294,
    "lbma_gold_pm": 88.2313,
    "lbma_silver": 1.0545,
    "lbma_platinum_am": 31.99,
    "lbma_platinum_pm": 32.2793,
    "lbma_palladium_am": 36.0088,
    "lbma_palladium_pm": 36.2017,
    "mcx_gold": 93.2689,
    "mcx_gold_am": 94.281,
    "mcx_gold_pm": 94.1764,
    "mcx_silver": 1.125,
    "mcx_silver_am": 1.1501,
    "mcx_silver_pm": 1.1483,
    "ibja_gold": 93.2713,
    "copper": 0.0098,
    "aluminum": 0.0026,
    "lead": 0.0021,
    "nickel": 0.0159,
    "zinc": 0.0031,
    "lme_copper": 0.0096,
    "lme_aluminum": 0.0026,
    "lme_lead": 0.002,
    "lme_nickel": 0.0158,
    "lme_zinc": 0.0031,
}
```

### 定义 get_metal_price 工具 { #define-the-get_metal_price-tool }

使用 LangChain 的 `@tool` 装饰器创建 get_metal_price 工具，获取指定金属的价格。

要使用 metals.dev 实时数据，可修改函数，向 API 发送真实请求。


```python
from langchain_core.tools import tool


# Define the tools for the agent to use
@tool
def get_metal_price(metal_name: str) -> float:
    """Fetches the current per gram price of the specified metal.

    Args:
        metal_name : The name of the metal (e.g., 'gold', 'silver', 'platinum').

    Returns:
        float: The current price of the metal in dollars per gram.

    Raises:
        KeyError: If the specified metal is not found in the data source.
    """
    try:
        metal_name = metal_name.lower().strip()
        if metal_name not in metal_price:
            raise KeyError(
                f"Metal '{metal_name}' not found. Available metals: {', '.join(metal_price['metals'].keys())}"
            )
        return metal_price[metal_name]
    except Exception as e:
        raise Exception(f"Error fetching metal price: {str(e)}")
```

### 将工具绑定到 LLM { #binding-the-tool-to-the-llm }
将工具绑定到 ChatOpenAI，使智能体能够根据用户请求调用外部工具，获取数据并执行操作。


```python
from langchain_openai import ChatOpenAI

tools = [get_metal_price]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
```

LangGraph 的状态用于跟踪图执行过程中的信息，并在节点之间传递。各节点运行时，状态也会更新。

例如，对话系统通过状态记录消息，每次生成新消息后加入历史记录，并将更新后的状态传给后续节点。

### 定义状态 { #defining-the-state }
定义保存消息列表的状态类，每次生成消息后追加到列表，持续更新对话历史。


```python
from langgraph.graph import END
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### 定义 should_continue 函数 { #defining-the-should_continue-function }
`should_continue` 判断是否继续调用工具，或结束对话。它检查最后一条消息是否含有工具调用，例如请求金属价格。

- 包含工具调用时，转到 tools 节点继续执行。
- 没有工具调用时，进入 END 状态，结束对话。


```python
# Define the function that determines whether to continue or not
def should_continue(state: GraphState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

### 调用模型 { #calling-the-model }
`call_model` 接收当前对话状态，调用 LLM 并返回生成的回答。


```python
# Define the function that calls the model
def call_model(state: GraphState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

### 创建助手节点 { #creating-the-assistant-node }
`assistant` 节点处理当前对话状态，判断下一步行动，并调用 LLM 生成合适的响应。


```python
# Node
def assistant(state: GraphState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 创建工具节点 { #creating-the-tool-node }
`tool_node` 根据状态与对话需要，调用前面定义的工具，例如获取金属价格。


```python
from langgraph.prebuilt import ToolNode

# Node
tools = [get_metal_price]
tool_node = ToolNode(tools)
```

### 构建图 { #building-the-graph }
使用 StateGraph 连接节点与边。每个节点代表一个处理步骤，例如助手或工具调用，边决定步骤间的执行流程。


```python
from langgraph.graph import START, StateGraph
from IPython.display import Image, display

# Define a new graph for the agent
builder = StateGraph(GraphState)

# Define the two nodes we will cycle between
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

# Set the entrypoint as `agent`
builder.add_edge(START, "assistant")

# Making a conditional edge
# should_continue will determine which node is called next.
builder.add_conditional_edges("assistant", should_continue, ["tools", END])

# Making a normal edge from `tools` to `agent`.
# The `agent` node will be called after the `tool`.
builder.add_edge("tools", "assistant")

# Compile and display the graph for a visual overview
react_graph = builder.compile()
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```



![智能体工作流](../../../howtos/integrations/_langgraph_agent_evaluation_files/_langgraph_agent_evaluation_23_0.jpg)



通过示例查询测试智能体，让它获取铜的价格。


```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="What is the price of copper?")]
result = react_graph.invoke({"messages": messages})
```


```python
result["messages"]
```




    [HumanMessage(content='What is the price of copper?', id='4122f5d4-e298-49e8-a0e0-c98adda78c6c'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_DkVQBK4UMgiXrpguUS2qC4mA', 'function': {'arguments': '{"metal_name":"copper"}', 'name': 'get_metal_price'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 116, 'total_tokens': 134, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0f77b156-e43e-4c1e-bd3a-307333eefb68-0', tool_calls=[{'name': 'get_metal_price', 'args': {'metal_name': 'copper'}, 'id': 'call_DkVQBK4UMgiXrpguUS2qC4mA', 'type': 'tool_call'}], usage_metadata={'input_tokens': 116, 'output_tokens': 18, 'total_tokens': 134}),
     ToolMessage(content='0.0098', name='get_metal_price', id='422c089a-6b76-4e48-952f-8925c3700ae3', tool_call_id='call_DkVQBK4UMgiXrpguUS2qC4mA'),
     AIMessage(content='The price of copper is $0.0098 per gram.', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 148, 'total_tokens': 162, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'stop', 'logprobs': None}, id='run-67cbf98b-4fa6-431e-9ce4-58697a76c36e-0', usage_metadata={'input_tokens': 148, 'output_tokens': 14, 'total_tokens': 162})]



### 将消息转为 Ragas 评估格式 { #converting-messages-to-ragas-evaluation-format }

当前 GraphState 使用列表保存用户、AI 与外部工具之间的消息，每条消息为 LangChain 格式对象。

```python
# Implementation of Graph State
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

智能体运行时会持续向列表添加消息，但 Ragas 评估需要其专用的消息结构。

因此需要将 LangChain 消息转成 Ragas 消息，再使用内置工具评估智能体。

**目标**是将 HumanMessage、AIMessage 和 ToolMessage 等 LangChain 对象列表，转成 Ragas 可识别的格式。

Ragas 提供 [convert_to_ragas_messages][ragas.integrations.langgraph.convert_to_ragas_messages] 完成转换。

用法如下。


```python
from ragas.integrations.langgraph import convert_to_ragas_messages

# Assuming 'result["messages"]' contains the list of LangChain messages
ragas_trace = convert_to_ragas_messages(result["messages"])
```


```python
ragas_trace  # List of Ragas messages
```




    [HumanMessage(content='What is the price of copper?', metadata=None, type='human'),
     AIMessage(content='', metadata=None, type='ai', tool_calls=[ToolCall(name='get_metal_price', args={'metal_name': 'copper'})]),
     ToolMessage(content='0.0098', metadata=None, type='tool'),
     AIMessage(content='The price of copper is $0.0098 per gram.', metadata=None, type='ai', tool_calls=None)]



## 评估智能体表现 { #evaluating-the-agents-performance }

本教程使用以下指标。

- [工具调用准确率](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#tool-call-accuracy)，衡量智能体能否识别并调用完成任务所需的工具。

- [智能体目标达成准确率](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/#agent-goal-accuracy)，衡量是否识别并完成用户目标，完成为 1，未完成为 0。


先运行几个查询，并准备对应的标准标签。

### 工具调用准确率 { #tool-call-accuracy }


```python
from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages
import ragas.messages as r


ragas_trace = convert_to_ragas_messages(
    messages=result["messages"]
)  # List of Ragas messages converted using the Ragas function

sample = MultiTurnSample(
    user_input=ragas_trace,
    reference_tool_calls=[
        r.ToolCall(name="get_metal_price", args={"metal_name": "copper"})
    ],
)

tool_accuracy_scorer = ToolCallAccuracy()
await tool_accuracy_scorer.multi_turn_ascore(sample)
```




    1.0



工具调用准确率为 1，因为 LLM 正确调用 get_metal_price，并传入正确的金属参数 copper。

### 智能体目标达成准确率 { #agent-goal-accuracy }


```python
messages = [HumanMessage(content="What is the price of 10 grams of silver?")]

result = react_graph.invoke({"messages": messages})
```


```python
result["messages"]  # List of LangChain messages
```




    [HumanMessage(content='What is the price of 10 grams of silver?', id='51a469de-5b7c-4d01-ab71-f8db64c8da49'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_rdplOo95CRwo3mZcPu4dmNxG', 'function': {'arguments': '{"metal_name":"silver"}', 'name': 'get_metal_price'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 120, 'total_tokens': 137, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-3bb60e27-1275-41f1-a46e-03f77984c9d8-0', tool_calls=[{'name': 'get_metal_price', 'args': {'metal_name': 'silver'}, 'id': 'call_rdplOo95CRwo3mZcPu4dmNxG', 'type': 'tool_call'}], usage_metadata={'input_tokens': 120, 'output_tokens': 17, 'total_tokens': 137}),
     ToolMessage(content='1.0523', name='get_metal_price', id='0b5f9260-df26-4164-b042-6df2e869adfb', tool_call_id='call_rdplOo95CRwo3mZcPu4dmNxG'),
     AIMessage(content='The current price of silver is approximately $1.0523 per gram. Therefore, the price of 10 grams of silver would be about $10.52.', response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 151, 'total_tokens': 185, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0ba0d124f1', 'finish_reason': 'stop', 'logprobs': None}, id='run-93e38f71-cc9d-41d6-812a-bfad9f9231b2-0', usage_metadata={'input_tokens': 151, 'output_tokens': 34, 'total_tokens': 185})]




```python
from ragas.integrations.langgraph import convert_to_ragas_messages

ragas_trace = convert_to_ragas_messages(
    result["messages"]
)  # List of Ragas messages converted using the Ragas function
ragas_trace
```




    [HumanMessage(content='What is the price of 10 grams of silver?', metadata=None, type='human'),
     AIMessage(content='', metadata=None, type='ai', tool_calls=[ToolCall(name='get_metal_price', args={'metal_name': 'silver'})]),
     ToolMessage(content='1.0523', metadata=None, type='tool'),
     AIMessage(content='The current price of silver is approximately $1.0523 per gram. Therefore, the price of 10 grams of silver would be about $10.52.', metadata=None, type='ai', tool_calls=None)]




```python
from ragas.dataset_schema import MultiTurnSample
from ragas.metrics import AgentGoalAccuracyWithReference
from ragas.llms import LangchainLLMWrapper


sample = MultiTurnSample(
    user_input=ragas_trace,
    reference="Price of 10 grams of silver",
)

scorer = AgentGoalAccuracyWithReference()

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
scorer.llm = evaluator_llm
await scorer.multi_turn_ascore(sample)
```




    1.0



目标达成准确率为 1，因为 LLM 完成了查询 10 克白银价格的目标。

## 下一步 { #whats-next }
🎉 现在已经掌握如何通过 Ragas 评估智能体。
