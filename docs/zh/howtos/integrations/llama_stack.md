# 使用 Llama 4 评估 LlamaStack 网页搜索回答的依据 { #evaluating-llamastack-web-search-groundedness-with-llama-4 }

本教程评估 LlamaStack 网页搜索智能体生成的回答是否有充分依据。[LlamaStack](https://llama-stack.readthedocs.io/en/latest/) 是 Meta 维护的开源框架，用于简化 LLM 应用开发与部署。本例使用 Ragas 指标，并以 Meta Llama 4 Maverick 作为裁判。

## 配置并运行 LlamaStack 服务 { #setup-and-running-a-llamastack-server }

以下命令安装使用 Together 推理服务商所需的全部依赖。

使用 conda 时，
```shell
!pip install ragas langchain-together uv
!uv run --with llama-stack llama stack build --template together --image-type conda
```

使用 venv 时，
```shell
!pip install ragas langchain-together uv
!uv run --with llama-stack llama stack build --template together --image-type venv
```


```python
import os
import subprocess


def run_llama_stack_server_background():
    log_file = open("llama_stack_server.log", "w")
    process = subprocess.Popen(
        "uv run --with llama-stack llama stack run together --image-type venv",
        shell=True,
        stdout=log_file,
        stderr=log_file,
        text=True,
    )

    print(f"Starting LlamaStack server with PID: {process.pid}")
    return process


def wait_for_server_to_start():
    import requests
    from requests.exceptions import ConnectionError
    import time

    url = "http://0.0.0.0:8321/v1/health"
    max_retries = 30
    retry_interval = 1

    print("Waiting for server to start", end="")
    for _ in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("\nServer is ready!")
                return True
        except ConnectionError:
            print(".", end="", flush=True)
            time.sleep(retry_interval)

    print("\nServer failed to start after", max_retries * retry_interval, "seconds")
    return False


# use this helper if needed to kill the server
def kill_llama_stack_server():
    # Kill any existing llama stack server processes
    os.system(
        "ps aux | grep -v grep | grep llama_stack.distribution.server.server | awk '{print $2}' | xargs kill -9"
    )
```

## 启动 LlamaStack 服务 { #starting-the-llamastack-server }


```python
server_process = run_llama_stack_server_background()
assert wait_for_server_to_start()
```
```
Starting LlamaStack server with PID: 95508
Waiting for server to start....
Server is ready!
```


## 构建搜索智能体 { #building-a-search-agent }


```python
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger

client = LlamaStackClient(
    base_url="http://0.0.0.0:8321",
)

agent = Agent(
    client,
    model="meta-llama/Llama-3.1-8B-Instruct",
    instructions="You are a helpful assistant. Use web search tool to answer the questions.",
    tools=["builtin::websearch"],
)
user_prompts = [
    "In which major did Demis Hassabis complete his undergraduate degree? Search the web for the answer.",
    "Ilya Sutskever is one of the key figures in AI. From which institution did he earn his PhD in machine learning? Search the web for the answer.",
    "Sam Altman, widely known for his role at OpenAI, was born in which American city? Search the web for the answer.",
]

session_id = agent.create_session("test-session")


for prompt in user_prompts:
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        session_id=session_id,
    )
    for log in AgentEventLogger().log(response):
        log.print()
```

查看智能体的详细执行步骤，了解其表现。


```python
session_response = client.agents.session.retrieve(
    session_id=session_id,
    agent_id=agent.agent_id,
)
```

## 评估智能体回答 { #evaluate-agent-responses }

需要构建 [EvaluationDataset](../../concepts/components/eval_dataset.md)，并选择衡量回答依据充分性的指标。Ragas 提供多种内置指标，可评价检索与生成的不同方面。

这里使用以下两个指标。

1. [忠实度](../../concepts/metrics/available_metrics/faithfulness.md)
2. [回答依据充分性](../../concepts/metrics/available_metrics/nvidia_metrics.md#response-groundedness)

### 构建 Ragas EvaluationDataset { #constructing-a-ragas-evaluationdataset }

创建 `EvaluationDataset` 供评估使用。


```python
import json

# This function extracts the search results for the trace of each query
def extract_retrieved_contexts(turn_object):
    results = []
    for step in turn_object.steps:
        if step.step_type == "tool_execution":
            tool_responses = step.tool_responses
            for response in tool_responses:
                content = response.content
                if content:
                    try:
                        parsed_result = json.loads(content)
                        results.append(parsed_result)
                    except json.JSONDecodeError:
                        print("Warning: Unable to parse tool response content as JSON.")
                        continue

    retrieved_context = []
    for result in results:
        top_content_list = [item["content"] for item in result["top_k"]]
        retrieved_context.extend(top_content_list)
    return retrieved_context
```


```python
from ragas.dataset_schema import EvaluationDataset

samples = []

references = [
    "Demis Hassabis completed his undergraduate degree in Computer Science.",
    "Ilya Sutskever earned his PhD from the University of Toronto.",
    "Sam Altman was born in Chicago, Illinois.",
]

for i, turn in enumerate(session_response.turns):
    samples.append(
        {
            "user_input": turn.input_messages[0].content,
            "response": turn.output_message.content,
            "reference": references[i],
            "retrieved_contexts": extract_retrieved_contexts(turn),
        }
    )

ragas_eval_dataset = EvaluationDataset.from_list(samples)
```


```python
ragas_eval_dataset.to_pandas()
```


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
      <th>reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>In which major did Demis Hassabis complete his...</td>
      <td>[Demis Hassabis holds a Bachelor's degree in C...</td>
      <td>Demis Hassabis completed his undergraduate deg...</td>
      <td>Demis Hassabis completed his undergraduate deg...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ilya Sutskever is one of the key figures in AI...</td>
      <td>[Jump to content Main menu Search Donate Creat...</td>
      <td>Ilya Sutskever earned his PhD in machine learn...</td>
      <td>Ilya Sutskever earned his PhD from the Univers...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sam Altman, widely known for his role at OpenA...</td>
      <td>[Sam Altman | Biography, OpenAI, Microsoft, &amp; ...</td>
      <td>Sam Altman was born in Chicago, Illinois, USA.</td>
      <td>Sam Altman was born in Chicago, Illinois.</td>
    </tr>
  </tbody>
</table>
</div>



### 设置 Ragas 指标 { #setting-the-ragas-metrics }


```python
from ragas.metrics import AnswerAccuracy, Faithfulness, ResponseGroundedness
from langchain_together import ChatTogether
from ragas.llms import LangchainLLMWrapper

llm = ChatTogether(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
evaluator_llm = LangchainLLMWrapper(llm)

ragas_metrics = [
    AnswerAccuracy(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    ResponseGroundedness(llm=evaluator_llm),
]
```

## 评估 { #evaluation }

最后运行评估。


```python
from ragas import evaluate

results = evaluate(dataset=ragas_eval_dataset, metrics=ragas_metrics)
results.to_pandas()
```
```
Evaluating: 100%|██████████| 9/9 [00:04<00:00,  2.03it/s]
```

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
      <th>reference</th>
      <th>nv_accuracy</th>
      <th>faithfulness</th>
      <th>nv_response_groundedness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>In which major did Demis Hassabis complete his...</td>
      <td>[Demis Hassabis holds a Bachelor's degree in C...</td>
      <td>Demis Hassabis completed his undergraduate deg...</td>
      <td>Demis Hassabis completed his undergraduate deg...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ilya Sutskever is one of the key figures in AI...</td>
      <td>[Jump to content Main menu Search Donate Creat...</td>
      <td>Ilya Sutskever earned his PhD in machine learn...</td>
      <td>Ilya Sutskever earned his PhD from the Univers...</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sam Altman, widely known for his role at OpenA...</td>
      <td>[Sam Altman | Biography, OpenAI, Microsoft, &amp; ...</td>
      <td>Sam Altman was born in Chicago, Illinois, USA.</td>
      <td>Sam Altman was born in Chicago, Illinois.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>


```python
kill_llama_stack_server()
```
