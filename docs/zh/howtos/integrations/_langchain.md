# LangChain { #langchain }
## 评估 LangChain 问答链 { #evaluating-langchain-qa-chains }

LangChain 是用于开发语言模型应用的框架，也可构建 RAG 系统（LangChain 中称为问答系统）。详细构建方法见[文档](https://python.langchain.com/docs/use_cases/question_answering/)。

通过本集成，可以使用 Ragas 指标评估问答链。


```python
#!pip install ragas langchain_openai python-dotenv
```


```python
# attach to the existing event loop when using jupyter notebooks
import os

import nest_asyncio
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# IMPORTANT: Remember to create a .env variable containing: OPENAI_API_KEY=sk-xyz where xyz is your key

# Access the API key from the environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAI API client
openai.api_key = api_key

nest_asyncio.apply()
```

先加载[纽约市维基百科页面](https://en.wikipedia.org/wiki/New_York_City)的数据，创建 `VectorstoreIndex` 和 `RetrievalQA`，构建通用问答系统。


```python
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI

loader = TextLoader("./nyc_wikipedia/nyc_text.txt")
index = VectorstoreIndexCreator().from_loaders([loader])


llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True,
)
```

    /home/jjmachan/.pyenv/versions/ragas/lib/python3.10/site-packages/langchain/indexes/vectorstore.py:128: UserWarning: Using InMemoryVectorStore as the default vectorstore.This memory store won't persist data. You should explicitlyspecify a vectorstore when using VectorstoreIndexCreator
      warnings.warn(



    ---------------------------------------------------------------------------

    ValidationError                           Traceback (most recent call last)

    Cell In[2], line 7
          4 from langchain_openai import ChatOpenAI
          6 loader = TextLoader("./nyc_wikipedia/nyc_text.txt")
    ----> 7 index = VectorstoreIndexCreator().from_loaders([loader])
         10 llm = ChatOpenAI(temperature=0)
         11 qa_chain = RetrievalQA.from_chain_type(
         12     llm,
         13     retriever=index.vectorstore.as_retriever(),
         14     return_source_documents=True,
         15 )


    File ~/.pyenv/versions/ragas/lib/python3.10/site-packages/pydantic/main.py:212, in BaseModel.__init__(self, **data)
        210 # `__tracebackhide__` tells pytest and some other tools to omit this function from tracebacks
        211 __tracebackhide__ = True
    --> 212 validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
        213 if self is not validated_self:
        214     warnings.warn(
        215         'A custom validator is returning a value other than `self`.\n'
        216         "Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\n"
        217         'See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.',
        218         category=None,
        219     )


    ValidationError: 1 validation error for VectorstoreIndexCreator
    embedding
      Field required [type=missing, input_value={}, input_type=dict]
        For further information visit https://errors.pydantic.dev/2.9/v/missing



```python
# testing it out

question = "How did New York City get its name?"
result = qa_chain({"query": question})
result["result"]
```

为评估问答系统，准备一组相关问题。这里提供了一些示例，也可以自行添加。


```python
eval_questions = [
    "What is the population of New York City as of 2020?",
    "Which borough of New York City has the highest population?",
    "What is the economic significance of New York City?",
    "How did New York City get its name?",
    "What is the significance of the Statue of Liberty in New York City?",
]

eval_answers = [
    "8,804,190",
    "Brooklyn",
    "New York City's economic significance is vast, as it serves as the global financial capital, housing Wall Street and major financial institutions. Its diverse economy spans technology, media, healthcare, education, and more, making it resilient to economic fluctuations. NYC is a hub for international business, attracting global companies, and boasts a large, skilled labor force. Its real estate market, tourism, cultural industries, and educational institutions further fuel its economic prowess. The city's transportation network and global influence amplify its impact on the world stage, solidifying its status as a vital economic player and cultural epicenter.",
    "New York City got its name when it came under British control in 1664. King Charles II of England granted the lands to his brother, the Duke of York, who named the city New York in his own honor.",
    "The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th and early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an iconic landmark and a global symbol of cultural diversity and freedom.",
]

examples = [
    {"query": q, "ground_truth": [eval_answers[i]]}
    for i, q in enumerate(eval_questions)
]
```

## 认识 `RagasEvaluatorChain` { #introducing-ragasevaluatorchain }

`RagasEvaluatorChain` 封装了 [Ragas 指标](https://github.com/vibrantlabsai/ragas/blob/main/docs/concepts/metrics/index.md)，便于在 LangChain 和 LangSmith 中评估。

评估链提供以下 API。

- `__call__()`，直接对问答链结果调用 `RagasEvaluatorChain`
- `evaluate()`，评估示例输入查询与问答链预测输出的列表
- `evaluate_run()`，由 LangSmith 评估器调用，用于评估 LangSmith 数据集

下面分别说明。


```python
result = qa_chain({"query": eval_questions[1]})
result["result"]
```


```python
result = qa_chain(examples[4])
result["result"]
```


```python
from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# create evaluation chains
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_precision)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)
```

1. `__call__()`

直接将问答链结果传给评估链。`context_precision` 和 `faithfulness` 等指标需要结果包含 `source_documents`。


```python
# Recheck the result that we are going to validate.
result
```

**忠实度**


```python
eval_result = faithfulness_chain(result)
eval_result["faithfulness_score"]
```

较高的 faithfulness_score 表示答案与来源文档高度一致。

将答案或 source_documents 替换为不一致的内容，可以观察得分降低。


```python
fake_result = result.copy()
fake_result["result"] = "we are the champions"
eval_result = faithfulness_chain(fake_result)
eval_result["faithfulness_score"]
```

**上下文召回率**


```python
eval_result = context_recall_chain(result)
eval_result["context_recall_score"]
```

较高的 context_recall_score 表示来源文档包含标准答案所需的信息。

将 source_documents 替换为其他内容，可以观察召回率降低。


```python
from langchain.schema import Document

fake_result = result.copy()
fake_result["source_documents"] = [Document(page_content="I love christmas")]
eval_result = context_recall_chain(fake_result)
eval_result["context_recall_score"]
```

2. `evaluate()`

评估输入查询列表和问答链的预测输出列表。


```python
# run the queries as a batch for efficiency
predictions = qa_chain.batch(examples)

# evaluate
print("evaluating...")
r = faithfulness_chain.evaluate(examples, predictions)
r
```


```python
# evaluate context recall
print("evaluating...")
r = context_recall_chain.evaluate(examples, predictions)
r
```

## 结合 LangSmith 评估 { #evaluate-with-langsmith }

[LangSmith](https://docs.smith.langchain.com/) 支持调试、测试、评估和监控各类 LLM 框架构建的链与智能体，并与 LangChain 集成。

LangSmith 提供构建测试数据集与运行评估的工具，结合 `RagasEvaluatorChain` 即可使用 Ragas 指标。详情参阅[快速入门](https://docs.smith.langchain.com/evaluation/quickstart)。


使用 `eval_questions` 中的纽约市问题创建 LangSmith 数据集并上传。


```python
# dataset creation

from langsmith import Client
from langsmith.utils import LangSmithError

client = Client()
dataset_name = "NYC test"

try:
    # check if dataset exists
    dataset = client.read_dataset(dataset_name=dataset_name)
    print("using existing dataset: ", dataset.name)
except LangSmithError:
    # if not create a new one with the generated query examples
    dataset = client.create_dataset(
        dataset_name=dataset_name, description="NYC test dataset"
    )
    for e in examples:
        client.create_example(
            inputs={"query": e["query"]},
            outputs={"ground_truth": e["ground_truth"]},
            dataset_id=dataset.id,
        )

    print("Created a new dataset: ", dataset.name)
```

![](../../../_static/langsmith-dataset.png)

问题上传后，可以在该测试集上运行问答链，并在平台中比较结果。

调用 `run_on_dataset` 前，需要一个工厂函数，为每条测试样本创建新的问答链实例，避免内部状态被复用。


```python
# factory function that return a new qa chain
def create_qa_chain(return_context=True):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=index.vectorstore.as_retriever(),
        return_source_documents=return_context,
    )
    return qa_chain
```

接下来运行评估。


```python
from langchain.smith import RunEvalConfig, run_on_dataset

evaluation_config = RunEvalConfig(
    custom_evaluators=[
        faithfulness_chain,
        answer_rel_chain,
        context_rel_chain,
        context_recall_chain,
    ],
    prediction_key="result",
)

result = run_on_dataset(
    client,
    dataset_name,
    create_qa_chain,
    evaluation=evaluation_config,
    input_mapper=lambda x: x,
)
```

通过输出链接在 LangSmith 中打开本次结果，并查看每条样本得分。

![](../../../_static/langsmith-evaluation.png)

点击样本并打开 feedback 标签页，即可深入查看得分并分析改进方向。

![](../../../_static/langsmith-feedback.png)

还可以查看对应的 `RagasEvaluatorChain` 追踪，理解评分原因。

![](../../../_static/langsmith-ragas-chain-trace.png)
