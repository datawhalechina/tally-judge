# Openlayer
## 结合 Openlayer 与 Ragas 评估 RAG 流水线 { #evaluating-rag-pipelines-with-openlayer-and-ragas }

[Openlayer](https://www.openlayer.com/) 可以融入开发与生产流水线，帮助评估和发布高质量模型。

本教程应配合[这篇博客](https://www.openlayer.com/blog/post/evaluating-rag-pipelines-with-ragas-and-openlayer)阅读。

## 前置条件 { #pre-requisites }


```bash
%%bash
git clone https://huggingface.co/datasets/vibrantlabsai/prompt-engineering-papers
```


```python
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
```

## 生成合成测试数据 { #synthetic-test-data-generation }


```python
from llama_index import SimpleDirectoryReader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# load documents
dir_path = "./prompt-engineering-papers"
reader = SimpleDirectoryReader(dir_path, num_files_limit=2)
documents = reader.load_data()

# generator with openai models
generator = TestsetGenerator.with_openai()

# set question type distribution
distribution = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}

# generate testset
testset = generator.generate_with_llamaindex_docs(
    documents, test_size=10, distributions=distribution
)
test_df = testset.to_pandas()
test_df.head()
```

## 构建 RAG { #building-rag }


```python
import nest_asyncio
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import OpenAIEmbedding


nest_asyncio.apply()


def build_query_engine(documents):
    vector_index = VectorStoreIndex.from_documents(
        documents,
        service_context=ServiceContext.from_defaults(chunk_size=512),
        embed_model=OpenAIEmbedding(),
    )

    query_engine = vector_index.as_query_engine(similarity_top_k=2)
    return query_engine
```


```python
query_engine = build_query_engine(documents)
```


```python
def generate_single_response(query_engine, question):
    response = query_engine.query(question)
    return {
        "answer": response.response,
        "contexts": [c.node.get_content() for c in response.source_nodes],
    }
```


```python
question = "What are some strategies proposed to enhance the in-context learning capability of language models?"
generate_single_response(query_engine, question)
```


```python
from datasets import Dataset


def generate_ragas_dataset(query_engine, test_df):
    test_questions = test_df["question"].values
    responses = [generate_single_response(query_engine, q) for q in test_questions]

    dataset_dict = {
        "question": test_questions,
        "answer": [response["answer"] for response in responses],
        "contexts": [response["contexts"] for response in responses],
        "ground_truth": test_df["ground_truth"].values.tolist(),
    }
    ds = Dataset.from_dict(dataset_dict)
    return ds
```


```python
ragas_dataset = generate_ragas_dataset(query_engine, test_df)
ragas_df = ragas_dataset.to_pandas()
```

## 提交到 Openlayer { #commit-to-openlayer }


```python
from openlayer.tasks import TaskType

client = openlayer.OpenlayerClient("YOUR_OPENLAYER_API_KEY_HERE")
```


```python
project = client.create_project(
    name="My-Rag-Project",
    task_type=TaskType.LLM,
    description="Evaluating an LLM used for product development.",
)
```


```python
validation_dataset_config = {
    "contextColumnName": "contexts",
    "questionColumnName": "question",
    "inputVariableNames": ["question"],
    "label": "validation",
    "outputColumnName": "answer",
    "groundTruthColumnName": "ground_truth",
}
project.add_dataframe(
    dataset_df=ragas_df,
    dataset_config=validation_dataset_config,
)
```


```python
model_config = {
    "inputVariableNames": ["question"],
    "modelType": "shell",
    "metadata": {"top_k": 2, "chunk_size": 512, "embeddings": "OpenAI"},
}
project.add_model(model_config=model_config)
```


```python
project.commit("Initial commit!")
project.push()
```


```python

```
