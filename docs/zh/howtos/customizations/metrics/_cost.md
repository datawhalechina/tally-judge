# 了解操作成本与用量 { #understand-cost-and-usage-of-operations }

使用 LLM 进行评估和测试集生成时，需要考虑成本。Ragas 提供了一些工具来统计用量与估算费用。

## 理解 `TokenUsageParser` { #understanding-tokenusageparser }

Ragas 默认不统计 `evaluate()` 的 token 用量，因为 LangChain 的不同 LLM 并不总以统一格式返回用量信息。要获取这些数据，需要实现 `TokenUsageParser`。

`TokenUsageParser` 是一个函数，将 LangChain 模型 `generate_prompt()` 返回的 `LLMResult` 或 `ChatResult` 解析成 Ragas 所需的 `TokenUsage`。

下面使用已定义的解析器，演示如何解析 OpenAI 用量。


```python
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"
```


```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompt_values import StringPromptValue

gpt4o = ChatOpenAI(model="gpt-4o")
p = StringPromptValue(text="hai there")
llm_result = gpt4o.generate_prompt([p])

# lets import a parser for OpenAI
from ragas.cost import get_token_usage_for_openai

get_token_usage_for_openai(llm_result)
```

    /opt/homebrew/Caskroom/miniforge/base/envs/ragas/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm





    TokenUsage(input_tokens=9, output_tokens=9, model='')



可以自行定义解析器，也可以导入现有解析器。如果希望建议支持某个 LLM 服务商，或贡献解析器，请查看[此 issue](https://github.com/vibrantlabsai/ragas/issues/1151)。

下面结合[入门评估示例](../../../getstarted/evals.md)使用解析器。


```python
from datasets import load_dataset
from ragas import EvaluationDataset
from ragas.metrics._aspect_critic import AspectCriticWithReference

dataset = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")


eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

metric = AspectCriticWithReference(
    name="answer_correctness",
    definition="is the response correct compared to reference",
)
```

    Repo card metadata block was not found. Setting CardData to empty.



```python
from ragas import evaluate
from ragas.cost import get_token_usage_for_openai

results = evaluate(
    eval_dataset[:5],
    metrics=[metric],
    llm=gpt4o,
    token_usage_parser=get_token_usage_for_openai,
)
```

    Evaluating: 100%|██████████| 5/5 [00:01<00:00,  2.81it/s]



```python
results.total_tokens()
```




    TokenUsage(input_tokens=5463, output_tokens=355, model='')



将每个 token 的价格传给 `Result.total_cost()`，即可计算每次运行的费用。

此示例按 GPT-4o 每 100 万输入 token 5 美元、每 100 万输出 token 15 美元估算。


```python
results.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```




    0.03264




```python

```
