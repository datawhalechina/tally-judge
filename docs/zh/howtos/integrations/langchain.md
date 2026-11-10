# LangChain 集成 { #langchain-integration }

本教程使用 Ragas 评估基于 LangChain 构建的 RAG 问答应用，并介绍如何通过 Ragas App 分析和改进应用表现。

### 构建简单的问答应用 { #building-a-simple-qa-application }

先创建小型数据集，将文本嵌入向量存入向量数据库并建立索引。


```python
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

content_list = [
    "Andrew Ng is the CEO of Landing AI and is known for his pioneering work in deep learning. He is also widely recognized for democratizing AI education through platforms like Coursera.",
    "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He is a strong advocate for creating safe and beneficial AI technologies.",
    "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems that can master complex games like AlphaGo.",
    "Sundar Pichai is the CEO of Google and Alphabet Inc., and he is praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences on a global scale.",
    "Arvind Krishna is the CEO of IBM and is recognized for transforming the company towards cloud computing and AI solutions. He focuses on providing cutting-edge technologies to address modern business challenges.",
]

langchain_documents = []

for content in content_list:
    langchain_documents.append(
        Document(
            page_content=content,
        )
    )
```


```python
from ragas.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import openai

openai_client = openai.OpenAI()
embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

_ = vector_store.add_documents(langchain_documents)
```

接下来，将检索器、LLM 与提示词组合成 Retrieval QA Chain。检索器从知识库获取相关文档，提示词引导 LLM 理解上下文，并根据文档生成相关、连贯的回答。

LangChain 中，可以通过向量存储的 `.as_retriever` 方法创建检索器。详情参阅[向量存储检索器文档](https://python.langchain.com/docs/how_to/vectorstore_retriever/)。


```python
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
```


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```

下面定义一条 Chain，将用户查询和相关文档组织为结构化提示词，传入模型，再将输出解析为最终回答字符串。


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


template = """Answer the question based only on the following context:
{context}

Question: {query}
"""
prompt = ChatPromptTemplate.from_template(template)

qa_chain = prompt | llm | StrOutputParser()
```


```python
def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)


query = "Who is the CEO of OpenAI?"

relevant_docs = retriever.invoke(query)
qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})
```
输出如下。
```
'The CEO of OpenAI is Sam Altman.'
```


### 评估 { #evaluate }


```python
sample_queries = [
    "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
    "Who is Sam Altman?",
    "Who is Demis Hassabis and how did he gained prominence?",
    "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
    "How did Arvind Krishna transformed IBM?",
]

expected_responses = [
    "Andrew Ng is the CEO of Landing AI and is widely recognized for democratizing AI education through platforms like Coursera.",
    "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He strongly advocates for creating safe and beneficial AI technologies.",
    "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems like AlphaGo that can master complex games.",
    "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences globally.",
    "Arvind Krishna is the CEO of IBM and has transformed the company towards cloud computing and AI solutions. He focuses on delivering cutting-edge technologies to address modern business challenges.",
]
```

将查询、预期回答及各指标所需的其他字段，组织为 [EvaluationDataset][ragas.dataset_schema.EvaluationDataset]。


```python
from ragas import EvaluationDataset


dataset = []

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = qa_chain.invoke({"context": format_docs(relevant_docs), "query": query})
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)
```

本例使用以下指标。


- `LLMContextRecall`，衡量检索上下文对参考答案陈述的覆盖程度，无须手动标注参考上下文。
- `Faithfulness`，判断生成答案中的全部陈述能否从给定上下文中推断出来。
- `FactualCorrectness`，通过陈述拆分和自然语言推断，比较生成回答与参考答案的事实准确性。

详情参阅 [Ragas 指标文档](../../concepts/metrics/available_metrics/index.md)。


```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)

result
```

输出如下。
```
{'context_recall': 1.0000, 'faithfulness': 0.9000, 'factual_correctness': 0.9260}
```
