# 评估简单的 RAG 系统 { #evaluate-a-simple-rag-system }

本指南演示如何使用 `ragas` 测试和评估 RAG 系统。你只需具备 RAG 系统构建与评估的基础知识。安装方法请参阅[安装说明](install.md)。

## 基础配置 { #basic-setup }

我们将通过 `langchain_openai` 配置 LLM 和嵌入模型，构建简单的 RAG 系统。你也可以选择其他模型，具体方法请参阅 [LangChain 模型集成文档](https://python.langchain.com/docs/integrations/chat/)。


```python
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai

llm = ChatOpenAI(model="gpt-4o")
openai_client = openai.OpenAI()
embeddings = OpenAIEmbeddings(client=openai_client)
```

!!! note "OpenAI 嵌入 API"
    `ragas.embeddings.OpenAIEmbeddings` 提供 `embed_text`（单条）与 `embed_texts`（批量）方法。部分 LangChain 封装提供的则是 `embed_query` 和 `embed_documents`。下面的示例使用 `embed_texts` 处理文档，使用 `embed_text` 处理查询。参见 [OpenAI 嵌入实现](https://docs.ragas.io/en/stable/references/embeddings/\#ragas.embeddings.OpenAIEmbeddings)。

### 构建简单的 RAG 系统 { #build-a-simple-rag-system }

构建一个简单的 RAG 系统，需要定义以下组件。

- 将文档转换为向量的方法
- 检索相关文档的方法
- 生成回答的方法

??? note "点击查看代码"

    ```python

    import numpy as np

    class RAG:
        def __init__(self, model="gpt-4o"):
            import openai
            self.llm = ChatOpenAI(model=model)
            openai_client = openai.OpenAI()
            self.embeddings = OpenAIEmbeddings(client=openai_client)
            self.doc_embeddings = None
            self.docs = None

        def load_documents(self, documents):
            """Load documents and compute their embeddings."""
            self.docs = documents
            self.doc_embeddings = self.embeddings.embed_texts(documents)

        def get_most_relevant_docs(self, query):
            """Find the most relevant document for a given query."""
            if not self.docs or not self.doc_embeddings:
                raise ValueError("Documents and their embeddings are not loaded.")

            query_embedding = self.embeddings.embed_text(query)
            similarities = [
                np.dot(query_embedding, doc_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
                for doc_emb in self.doc_embeddings
            ]
            most_relevant_doc_index = np.argmax(similarities)
            return [self.docs[most_relevant_doc_index]]

        def generate_answer(self, query, relevant_doc):
            """Generate an answer for a given query based on the most relevant document."""
            prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
            messages = [
                ("system", "You are a helpful assistant that answers questions based on given documents only."),
                ("human", prompt),
            ]
            ai_msg = self.llm.invoke(messages)
            return ai_msg.content
    ```

### 加载文档 { #load-documents }
接下来加载一些文档，并测试 RAG 系统。

```python
sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]
```

```python
# Initialize RAG instance
rag = RAG()

# Load documents
rag.load_documents(sample_docs)

# Query and retrieve the most relevant document
query = "Who introduced the theory of relativity?"
relevant_doc = rag.get_most_relevant_docs(query)

# Generate an answer
answer = rag.generate_answer(query, relevant_doc)

print(f"Query: {query}")
print(f"Relevant Document: {relevant_doc}")
print(f"Answer: {answer}")
```


输出如下。
```
Query: Who introduced the theory of relativity?
Relevant Document: ['Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.']
Answer: Albert Einstein introduced the theory of relativity.
```

## 收集评估数据 { #collect-evaluation-data }

首先准备一组查询，并通过 RAG 系统运行，收集每个查询对应的 `response` 和 `retrieved_contexts`。你也可以为每个查询准备标准答案，以评估系统表现。



```python


sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]
```

```python
dataset = []

for query,reference in zip(sample_queries,expected_responses):

    relevant_docs = rag.get_most_relevant_docs(query)
    response = rag.generate_answer(query, relevant_docs)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
    )
```

然后，将数据集加载为 `EvaluationDataset` 对象。

```python
from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)
```

## 评估 { #evaluate }

评估数据已经收集完成。现在可以在数据集上使用常见的 RAG 评估指标来衡量系统表现。评估时可以选择任意合适的[评估 LLM](../howtos/customizations/customize_models.md)。

```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper


evaluator_llm = LangchainLLMWrapper(llm)
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
result
```

输出如下。
```
{'context_recall': 1.0000, 'faithfulness': 0.8571, 'factual_correctness': 0.7280}
```

### 希望通过评估改进 AI 应用？ { #want-help-in-improving-your-ai-application-using-evals }

过去两年，我们通过评估帮助许多 AI 应用提升了质量。

我们正在将这些经验融入产品，让评估闭环取代凭感觉判断，使你能够专注于构建出色的 AI 应用。

如果你希望通过评估改进 AI 应用并扩大其应用规模，欢迎联系我们。


🔗 [预约交流](https://bit.ly/3EBYq4J)，或发送邮件至 [founders@vibrantlabs.com](mailto:founders@vibrantlabs.com)。

![](../../_static/ragas_app.gif)


## 下一步 { #up-next }

- [生成用于评估 RAG 的测试数据](rag_testset_generation.md)
