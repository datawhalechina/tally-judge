# 评估 LlamaIndex 智能体 { #evaluating-llamaindex-agents }

构建能使用工具、作出决策的智能体之后，还需要检验其准确性、可靠性与效率。[LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/agent/) 支持 [FunctionAgent](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)、[CodeActAgent](https://docs.llamaindex.ai/en/stable/examples/agent/code_act_agent/) 和 [ReActAgent](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/) 等类型。本教程结合 Ragas 内置与自定义指标，介绍它们的评估方法。

下面开始。

教程分为三个部分。

1. **使用内置 Ragas 指标评估**
   介绍 AgentGoalAccuracy 与 Tool Call Accuracy，分别衡量用户目标是否达成，以及是否按正确顺序选择和调用工具。

2. **使用自定义指标评估 CodeActAgent**
   为 LlamaIndex 的 CodeActAgent 构建专门指标，评估代码生成型智能体的能力与任务要求。

3. **评估查询引擎工具**
   通过 Ragas 的 RAG 指标，检查智能体访问外部信息系统时的检索效果与回答质量。

## Ragas 智能体指标 { #ragas-agentic-metrics }

先创建一个仅包含单个 LlamaIndex FunctionAgent 的简单工作流，演示基础评估功能。

??? note "点击查看 FunctionAgent 配置"

    ```python
    from llama_index.llms.openai import OpenAI


    async def send_message(to: str, content: str) -> str:
        """Dummy function to simulate sending an email."""
        return f"Successfully sent mail to {to}"

    llm = OpenAI(model="gpt-4o-mini")
    ```


    ```python
    from llama_index.core.agent.workflow import FunctionAgent

    agent = FunctionAgent(
        tools=[send_message],
        llm=llm,
        system_prompt="You are a helpful assistant of Jane",
    )
    ```

### 智能体目标达成准确率 { #agent-goal-accuracy }

智能体的价值在于理解并完成用户目标。Agent Goal Accuracy 直接衡量这一能力，反映它能否正确理解需求并采取合适行动。

Ragas 提供两种变体。

- [AgentGoalAccuracyWithReference](../../concepts/metrics/available_metrics/agents.md#with-reference)，比较最终结果与预定义预期结果，返回 1 或 0。
- [AgentGoalAccuracyWithoutReference](../../concepts/metrics/available_metrics/agents.md#without-reference)，从对话推断用户意图，再检查目标是否达成，返回 1 或 0。

预期结果明确时适合有参考版本，例如受控测试或已有标准答案的数据评估。


```python
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
	AgentStream,
    ToolCall as LlamaToolCall,
    ToolCallResult,
)

handler =  agent.run(user_msg="Send a message to jhon asking for a meeting")

events = []

async for ev in handler.stream_events():
    if isinstance(ev, (AgentInput, AgentOutput, LlamaToolCall, ToolCallResult)):
        events.append(ev)
    elif isinstance(ev, AgentStream):
        print(f"{ev.delta}", end="", flush=True)
    elif isinstance(ev, ToolCallResult):
        print(
            f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}"
        )

response = await handler
```
输出如下。
```
I have successfully sent a message to Jhon asking for a meeting.
```

```python
from ragas.integrations.llama_index import convert_to_ragas_messages

ragas_messages = convert_to_ragas_messages(events)
```


```python
from ragas.metrics import AgentGoalAccuracyWithoutReference
from ragas.llms import LlamaIndexLLMWrapper
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import ToolCall as RagasToolCall

evaluator_llm = LlamaIndexLLMWrapper(llm=llm)

sample = MultiTurnSample(
    user_input=ragas_messages,
)

agent_goal_accuracy_without_reference = AgentGoalAccuracyWithoutReference(llm=evaluator_llm)
await agent_goal_accuracy_without_reference.multi_turn_ascore(sample)
```
输出如下。
```
1.0
```

```python
from ragas.metrics import AgentGoalAccuracyWithReference

sample = MultiTurnSample(
    user_input=ragas_messages,
    reference="Successfully sent a message to Jhon asking for a meeting"
)


agent_goal_accuracy_with_reference = AgentGoalAccuracyWithReference(llm=evaluator_llm)
await agent_goal_accuracy_with_reference.multi_turn_ascore(sample)
```
输出如下。
```
1.0
```

### 工具调用准确率 { #tool-call-accuracy }

智能体能否在适当时机选择并使用正确工具，直接影响工作流效果。工具调用准确率检查工具选择、调用顺序与用户任务是否一致。

- [ToolCallAccuracy](../../concepts/metrics/available_metrics/agents.md#tool-call-accuracy) 将实际工具使用与参考序列比较。工具选择或严格模式下的顺序不匹配时，返回 0，表示未遵循预期调用路径。


```python
from ragas.metrics import ToolCallAccuracy

sample = MultiTurnSample(
    user_input=ragas_messages,
    reference_tool_calls=[
        RagasToolCall(
            name="send_message",
            args={'to': 'jhon', 'content': 'Hi Jhon,\n\nI hope this message finds you well. I would like to schedule a meeting to discuss some important matters. Please let me know your availability.\n\nBest regards,\nJane'},
        ),
    ],
)

tool_accuracy_scorer = ToolCallAccuracy()
await tool_accuracy_scorer.multi_turn_ascore(sample)
```
输出如下。
```
1.0
```


## 评估 LlamaIndex CodeAct 智能体 { #evaluating-llamaindex-codeact-agents }

LlamaIndex 提供受 CodeAct 论文启发的预构建 CodeActAgent，通过生成并执行代码采取行动，通常使用 Python 等高级语言。这种方式具有以下特点。

- 可组合性，支持函数嵌套与复用。
- 对象管理，能够自然地保存与处理操作输出，例如 image = generate_image()。
- 通用性，可以表达各种计算任务。
- 训练数据基础，LLM 已从训练数据中学习代码，便于使用这一交互方式。

??? note "点击查看 CodeActAgent 配置"

    ### 定义函数 { #defining-functions }

    ```python
    from llama_index.llms.openai import OpenAI

    # Configure the LLM
    llm = OpenAI(model="gpt-4o-mini")


    # Define a few helper functions
    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b


    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b


    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b


    def divide(a: int, b: int) -> float:
        """Divide two numbers"""
        return a / b
    ```

    ### 创建代码执行器 { #create-a-code-executor }

    CodeActAgent 需要提供特定的 code_execute_fn，执行智能体生成的代码。


    ```python
    from typing import Any, Dict, Tuple
    import io
    import contextlib
    import ast
    import traceback


    class SimpleCodeExecutor:
        """
        A simple code executor that runs Python code with state persistence.

        This executor maintains a global and local state between executions,
        allowing for variables to persist across multiple code runs.

        NOTE: not safe for production use! Use with caution.
        """

        def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]):
            """
            Initialize the code executor.

            Args:
                locals: Local variables to use in the execution context
                globals: Global variables to use in the execution context
            """
            # State that persists between executions
            self.globals = globals
            self.locals = locals

        def execute(self, code: str) -> Tuple[bool, str, Any]:
            """
            Execute Python code and capture output and return values.

            Args:
                code: Python code to execute

            Returns:
                Dict with keys `success`, `output`, and `return_value`
            """
            # Capture stdout and stderr
            stdout = io.StringIO()
            stderr = io.StringIO()

            output = ""
            return_value = None
            try:
                # Execute with captured output
                with contextlib.redirect_stdout(
                    stdout
                ), contextlib.redirect_stderr(stderr):
                    # Try to detect if there's a return value (last expression)
                    try:
                        tree = ast.parse(code)
                        last_node = tree.body[-1] if tree.body else None

                        # If the last statement is an expression, capture its value
                        if isinstance(last_node, ast.Expr):
                            # Split code to add a return value assignment
                            last_line = code.rstrip().split("\n")[-1]
                            exec_code = (
                                code[: -len(last_line)]
                                + "\n__result__ = "
                                + last_line
                            )

                            # Execute modified code
                            exec(exec_code, self.globals, self.locals)
                            return_value = self.locals.get("__result__")
                        else:
                            # Normal execution
                            exec(code, self.globals, self.locals)
                    except:
                        # If parsing fails, just execute the code as is
                        exec(code, self.globals, self.locals)

                # Get output
                output = stdout.getvalue()
                if stderr.getvalue():
                    output += "\n" + stderr.getvalue()

            except Exception as e:
                # Capture exception information
                output = f"Error: {type(e).__name__}: {str(e)}\n"
                output += traceback.format_exc()

            if return_value is not None:
                output += "\n\n" + str(return_value)

            return output
    ```


    ```python
    code_executor = SimpleCodeExecutor(
        # give access to our functions defined above
        locals={
            "add": add,
            "subtract": subtract,
            "multiply": multiply,
            "divide": divide,
        },
        globals={
            # give access to all builtins
            "__builtins__": __builtins__,
            # give access to numpy
            "np": __import__("numpy"),
        },
    )
    ```

    ### 配置 CodeAct 智能体 { #setup-the-codeact-agent }


    ```python
    from llama_index.core.agent.workflow import CodeActAgent
    from llama_index.core.workflow import Context

    agent = CodeActAgent(
        code_execute_fn=code_executor.execute,
        llm=llm,
        tools=[add, subtract, multiply, divide],
    )

    # context to hold the agent's session/state/chat history
    ctx = Context(agent)
    ```

### 运行并评估 CodeAct 智能体 { #running-and-evaluating-the-codeact-agent }


```python
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
)

handler = agent.run("Calculate the sum of the first 10 fibonacci numbers", ctx=ctx)

events = []

async for event in handler.stream_events():
    if isinstance(event, (AgentInput, AgentOutput, ToolCall, ToolCallResult)):
        events.append(event)
    elif isinstance(event, AgentStream):
        print(f"{event.delta}", end="", flush=True)
```

    The first 10 Fibonacci numbers are 0, 1, 1, 2, 3, 5, 8, 13, 21, and 34. I will calculate their sum.

    <execute>
    def fibonacci(n):
        fib_sequence = [0, 1]
        for i in range(2, n):
            next_fib = fib_sequence[-1] + fib_sequence[-2]
            fib_sequence.append(next_fib)
        return fib_sequence

    # Calculate the first 10 Fibonacci numbers
    first_10_fib = fibonacci(10)

    # Calculate the sum of the first 10 Fibonacci numbers
    sum_fib = sum(first_10_fib)
    print(sum_fib)
    </execute>The sum of the first 10 Fibonacci numbers is 88.

### 提取 ToolCall { #extract-the-toolcall }


```python
CodeAct_agent_tool_call = events[2]
agent_code = CodeAct_agent_tool_call.tool_kwargs["code"]

print(agent_code)
```
输出如下。
```
    def fibonacci(n):
        fib_sequence = [0, 1]
        for i in range(2, n):
            next_fib = fib_sequence[-1] + fib_sequence[-2]
            fib_sequence.append(next_fib)
        return fib_sequence

    # Calculate the first 10 Fibonacci numbers
    first_10_fib = fibonacci(10)

    # Calculate the sum of the first 10 Fibonacci numbers
    sum_fib = sum(first_10_fib)
    print(sum_fib)
```

评估 CodeAct 智能体时，可以先检查代码能否编译、参数选择是否正确等基础功能，再逐步增加更复杂的评估。

Ragas 的自定义指标支持随着需求变化不断细化评估。

- [AspectCritic](../../concepts/metrics/available_metrics/aspect_critic.md)，通过 LLM 判断回答是否满足用户指定标准，返回通过/失败。
- [RubricScoreMetric](../../concepts/metrics/available_metrics/general_purpose.md#rubrics-based-criteria-scoring)，按照预定义的质量细则和离散评分等级，跨多个维度一致地评估回答。


```python
def is_compilable(code_str: str, mode="exec") -> bool:
    try:
        compile(code_str, "<string>", mode)
        return True
    except Exception:
        return False

is_compilable(agent_code)
```
输出如下。
```
True
```



```python
from ragas.metrics import AspectCritic
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LlamaIndexLLMWrapper

llm = OpenAI(model="gpt-4o-mini")
evaluator_llm = LlamaIndexLLMWrapper(llm=llm)

correct_tool_args = AspectCritic(
    name="correct_tool_args",
    llm=evaluator_llm,
    definition="Score 1 if the tool arguements use in the tool call are correct and 0 otherwise",
)

sample = SingleTurnSample(
    user_input="Calculate the sum of the first 10 fibonacci numbers",
    response=agent_code,
)

await correct_tool_args.single_turn_ascore(sample)
```
输出如下。
```
1
```


## 评估查询引擎工具 { #evaluating-query-engine-tool }

使用 Ragas 时，先将数据整理为合适格式。智能体中的查询引擎工具可以按普通 RAG 系统评估。

从用户交互事件流中提取查询引擎的每次调用，构建 Ragas 评估数据集，再应用相关指标。本节为 FunctionAgent 配置两个工具，分别查询 Lyft 和 Uber 的 2021 年 10-K 年报。

??? note "点击查看智能体配置"

    ### 设置 LLM { #setting-the-llms }

    ```python
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core import Settings

    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    ```

    ### 构建查询引擎工具 { #build-query-engine-tools }


    ```python
    from llama_index.core import StorageContext, load_index_from_storage

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/lyft"
        )
        lyft_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/uber"
        )
        uber_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False
    ```


    ```python
    !mkdir -p 'data/10k/'
    !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
    !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'
    ```

    ```python
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

    if not index_loaded:
        # load data
        lyft_docs = SimpleDirectoryReader(
            input_files=["./data/10k/lyft_2021.pdf"]
        ).load_data()
        uber_docs = SimpleDirectoryReader(
            input_files=["./data/10k/uber_2021.pdf"]
        ).load_data()

        # build index
        lyft_index = VectorStoreIndex.from_documents(lyft_docs)
        uber_index = VectorStoreIndex.from_documents(uber_docs)

        # persist index
        lyft_index.storage_context.persist(persist_dir="./storage/lyft")
        uber_index.storage_context.persist(persist_dir="./storage/uber")
    ```


    ```python
    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)
    ```


    ```python
    from llama_index.core.tools import QueryEngineTool

    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=lyft_engine,
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=uber_engine,
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ]
    ```


    ### 配置智能体 { #agent-setup }


    ```python
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
    from llama_index.core.workflow import Context

    agent = FunctionAgent(tools=query_engine_tools, llm=OpenAI(model="gpt-4o-mini"))

    # context to hold the session/state
    ctx = Context(agent)
    ```

### 运行并评估智能体 { #running-and-evaluating-agents }


```python
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

handler = agent.run("What's the revenue for Lyft in 2021 vs Uber?", ctx=ctx)

events = []

async for ev in handler.stream_events():
    if isinstance(ev, (AgentInput, AgentOutput, ToolCall, ToolCallResult)):
        events.append(ev)
    elif isinstance(ev, AgentStream):
        print(ev.delta, end="", flush=True)

response = await handler
```
输出如下。
```
In 2021, Lyft generated a total revenue of $3.21 billion, while Uber's total revenue was significantly higher at $17.455 billion.
```

提取用户交互中查询引擎调用对应的全部 `ToolCallResult`，从事件流构建 RAG 评估数据集。


```python
from ragas.dataset_schema import SingleTurnSample

ragas_samples = []

for event in events:
	if isinstance(event, ToolCallResult):
		if event.tool_name in ["lyft_10k", "uber_10k"]:
			sample = SingleTurnSample(
				user_input=event.tool_kwargs["input"],
				response=event.tool_output.content,
				retrieved_contexts=[node.text for node in event.tool_output.raw_output.source_nodes]
				)
			ragas_samples.append(sample)
```


```python
from ragas.dataset_schema import EvaluationDataset

dataset = EvaluationDataset(samples=ragas_samples)
dataset.to_pandas()
```
输出如下。

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What was the total revenue for Uber in the yea...</td>
      <td>[Financial and Operational Highlights\nYear En...</td>
      <td>The total revenue for Uber in the year 2021 wa...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What was the total revenue for Lyft in the yea...</td>
      <td>[Significant items\n subject to estimates and ...</td>
      <td>The total revenue for Lyft in the year 2021 wa...</td>
    </tr>
  </tbody>
</table>
</div>



生成的数据集默认不包含参考答案，因此先使用无须参考答案的指标。需要有参考评估时，可以补充 reference 列，再应用对应指标。

### 使用 Ragas 的 RAG 指标评估 { #evaluating-using-ragas-rag-metrics }

重点检查查询引擎的检索质量与幻觉情况，使用忠实度和上下文相关性两个指标，更多详情见[指标文档](../../concepts/metrics/available_metrics/index.md)。

这一评估方式有助于定位检索或回答生成中影响整体表现的问题。
- [忠实度](../../concepts/metrics/available_metrics/faithfulness.md)，判断生成回答中的事实是否得到检索上下文支持。
- [上下文相关性](../../concepts/metrics/available_metrics/nvidia_metrics.md#context-relevance)，通过两次 LLM 评判，衡量检索信息是否切合用户查询。



```python
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRelevance
from ragas.llms import LlamaIndexLLMWrapper
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")
evaluator_llm = LlamaIndexLLMWrapper(llm=llm)

faithfulness = Faithfulness(llm=evaluator_llm)
context_precision = ContextRelevance(llm=evaluator_llm)

result = evaluate(dataset, metrics=[faithfulness, context_precision])
```
```
Evaluating: 100%|██████████| 4/4 [00:03<00:00,  1.19it/s]
```


```python
result.to_pandas()
```
输出如下。

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>faithfulness</th>
      <th>nv_context_relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What was the total revenue for Uber in the yea...</td>
      <td>[Financial and Operational Highlights\nYear En...</td>
      <td>The total revenue for Uber in the year 2021 wa...</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What was the total revenue for Lyft in the yea...</td>
      <td>[Significant items\n subject to estimates and ...</td>
      <td>The total revenue for Lyft in the year 2021 wa...</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
