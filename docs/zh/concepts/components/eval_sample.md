# 评估样本 { #evaluation-sample }

评估样本是一条结构化数据，用于衡量 LLM 应用在具体场景中的表现。它代表一次交互，或 AI 应用需要处理的一个具体用例。Ragas 使用 `SingleTurnSample` 与 `MultiTurnSample` 类表示评估样本。

## SingleTurnSample
`SingleTurnSample` 表示用户与 LLM 之间的单轮交互，以及用于评估的预期结果。适用于只有一组问答的评估，也可以包含上下文或参考信息。


### 示例 { #example }
以下示例为 RAG 应用中的单轮交互创建 `SingleTurnSample` 实例。用户提出问题，AI 给出回答。该实例包含检索上下文、参考答案和评分标准等信息。
```python
from ragas import SingleTurnSample

# User's question
user_input = "What is the capital of France?"

# Retrieved contexts (e.g., from a knowledge base or search engine)
retrieved_contexts = ["Paris is the capital and most populous city of France."]

# AI's response
response = "The capital of France is Paris."

# Reference answer (ground truth)
reference = "Paris"

# Evaluation rubric
rubric = {
    "accuracy": "Correct",
    "completeness": "High",
    "fluency": "Excellent"
}

# Create the SingleTurnSample instance
sample = SingleTurnSample(
    user_input=user_input,
    retrieved_contexts=retrieved_contexts,
    response=response,
    reference=reference,
    rubric=rubric
)
```

## MultiTurnSample

`MultiTurnSample` 表示人类用户、AI 以及可选工具之间的多轮交互，并记录用于评估的预期结果，适合评估对话智能体的复杂交互。在 `MultiTurnSample` 中，`user_input` 是一组消息，构成人类用户与 AI 系统之间的多轮对话。消息分别由 `HumanMessage`、`AIMessage` 和 `ToolMessage` 类的实例表示。


### 示例 { #example_1 }
以下示例创建一个 `MultiTurnSample` 实例来评估多轮交互。用户想知道纽约市当前的天气，AI 助手调用天气 API 工具获取信息，再向用户作答。


```python
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

# User asks about the weather in New York City
user_message = HumanMessage(content="What's the weather like in New York City today?")

# AI decides to use a weather API tool to fetch the information
ai_initial_response = AIMessage(
    content="Let me check the current weather in New York City for you.",
    tool_calls=[ToolCall(name="WeatherAPI", args={"location": "New York City"})]
)

# Tool provides the weather information
tool_response = ToolMessage(content="It's sunny with a temperature of 75°F in New York City.")

# AI delivers the final response to the user
ai_final_response = AIMessage(content="It's sunny and 75 degrees Fahrenheit in New York City today.")

# Combine all messages into a list to represent the conversation
conversation = [
    user_message,
    ai_initial_response,
    tool_response,
    ai_final_response
]
```

然后使用这段对话创建 `MultiTurnSample` 对象，并按需加入参考回答和评分标准。
```python
from ragas import MultiTurnSample
# Reference response for evaluation purposes
reference_response = "Provide the current weather in New York City to the user."


# Create the MultiTurnSample instance
sample = MultiTurnSample(
    user_input=conversation,
    reference=reference_response,
)
```
