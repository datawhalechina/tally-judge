# Phoenix (Arize)

## 1. 简介 { #1-introduction }

建立 RAG 基线通常并不困难，但要达到生产要求、保证回答质量，还需持续优化。面对众多工具和参数，选型也不容易。本教程通过一个完整流程，帮助你作出选择并检查 RAG 质量。

我们结合以下开源库，评估、可视化并分析 RAG。

- [Ragas](https://docs.ragas.io/en/stable/)，生成合成测试数据并评估
- Arize AI 的 [Phoenix](https://docs.arize.com/phoenix)，进行追踪、可视化与聚类分析
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/)，构建 RAG 流水线

本例使用 arXiv 上关于提示词工程的论文构建 RAG。

ℹ️ 本教程需要 OpenAI API 密钥。

## 2. 安装依赖并导入库 { #2-install-dependencies-and-import-libraries }

先安装 Git LFS，用于下载数据集。


```python
!git lfs install
```

安装并导入 Python 依赖。


```python
!pip install "ragas<0.1.1" pypdf arize-phoenix "openinference-instrumentation-llama-index<1.0.0" "llama-index<0.10.0" pandas
```


```python
import pandas as pd

# Display the complete contents of DataFrame cells.
pd.set_option("display.max_colwidth", None)
```

## 3. 配置 OpenAI API 密钥 { #3-configure-your-openai-api-key }

如果尚未通过环境变量设置，请先配置 OpenAI API 密钥。


```python
import os
from getpass import getpass
import openai

if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
    openai_api_key = getpass("🔑 Enter your OpenAI API key: ")
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
```

## 4. 生成合成测试数据集 { #4-generate-your-synthetic-test-dataset }

手工整理标准评估数据集可能耗时、繁琐且成本高，尤其是在项目初期或数据源频繁变化时。可以先生成高质量合成样本，再由开发者复核，文中示例可将整理工作量减少约 90%。

下载 arXiv 提示词工程论文的 PDF 数据集，再使用 LlamaIndex 读取。


```python
!git clone https://huggingface.co/datasets/vibrantlabsai/prompt-engineering-papers
```


```python
from llama_index import SimpleDirectoryReader

dir_path = "./prompt-engineering-papers"
reader = SimpleDirectoryReader(dir_path, num_files_limit=2)
documents = reader.load_data()
```

理想测试集应兼顾质量与多样性，并接近生产环境的数据分布。Ragas 采用基于演化的方法生成多样问题。默认使用 OpenAI，也可以更换模型。这里生成 100 条数据。


```python
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

TEST_SIZE = 25

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

# generate testset
testset = generator.generate_with_llamaindex_docs(documents, test_size=TEST_SIZE)
test_df = testset.to_pandas()
test_df.head()
```

可以按需调整问题类型分布。数据集准备后，使用 LlamaIndex 构建简单 RAG 流水线。

## 5. 使用 LlamaIndex 构建 RAG { #5-build-your-rag-application-with-llamaindex }

LlamaIndex 是易用且灵活的 RAG 框架。为简化示例，使用默认 LLM（gpt-3.5-turbo）和嵌入模型（openai-ada-2）。

在后台启动 Phoenix，并为应用添加埋点，使其收集 OpenInference 的 span 与 trace。[OpenInference](https://github.com/Arize-ai/openinference/tree/main/spec) 是基于 OpenTelemetry 的开放标准，记录 LLM 执行及相关上下文，包括向量检索、外部工具和 API 调用。


```python
import phoenix as px
from llama_index import set_global_handler

session = px.launch_app()
set_global_handler("arize_phoenix")
```

构建查询引擎。


```python
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding


def build_query_engine(documents):
    vector_index = VectorStoreIndex.from_documents(
        documents,
        service_context=ServiceContext.from_defaults(chunk_size=512),
        embed_model=OpenAIEmbedding(),
    )
    query_engine = vector_index.as_query_engine(similarity_top_k=2)
    return query_engine


query_engine = build_query_engine(documents)
```

在 Phoenix 中可看到索引语料时产生的嵌入 span。将这些嵌入导出为 DataFrame，供后续可视化使用。


```python
from phoenix.trace.dsl import SpanQuery

client = px.Client()
corpus_df = px.Client().query_spans(
    SpanQuery().explode(
        "embedding.embeddings",
        text="embedding.text",
        vector="embedding.vector",
    )
)
corpus_df.head()
```

重新启动 Phoenix，清除已积累的追踪。


```python
px.close_app()
session = px.launch_app()
```

## 6. 评估 LLM 应用 { #6-evaluate-your-llm-application }

Ragas 提供全面的指标，支持 RAG 的组件级与端到端评估。

先将问题、生成答案、检索上下文和标准答案组织为评估数据集。


```python
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd


def generate_response(query_engine, question):
    response = query_engine.query(question)
    return {
        "answer": response.response,
        "contexts": [c.node.get_content() for c in response.source_nodes],
    }


def generate_ragas_dataset(query_engine, test_df):
    test_questions = test_df["question"].values
    responses = [generate_response(query_engine, q) for q in tqdm(test_questions)]

    dataset_dict = {
        "question": test_questions,
        "answer": [response["answer"] for response in responses],
        "contexts": [response["contexts"] for response in responses],
        "ground_truth": test_df["ground_truth"].values.tolist(),
    }
    ds = Dataset.from_dict(dataset_dict)
    return ds


ragas_eval_dataset = generate_ragas_dataset(query_engine, test_df)
ragas_evals_df = pd.DataFrame(ragas_eval_dataset)
ragas_evals_df.head()
```

在 Phoenix 中查看 LlamaIndex 应用追踪。


```python
print(session.url)
```

![Phoenix 中的 LlamaIndex 应用追踪](https://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/ragas/ragas_trace_slide_over.gif)

保存两份 DataFrame，一份包含稍后可视化的嵌入，另一份包含待 Ragas 评估的 trace 与 span。


```python
# dataset containing embeddings for visualization
query_embeddings_df = px.Client().query_spans(
    SpanQuery().explode(
        "embedding.embeddings", text="embedding.text", vector="embedding.vector"
    )
)
query_embeddings_df.head()
```


```python
from phoenix.session.evaluation import get_qa_with_reference

# dataset containing span data for evaluation with Ragas
spans_dataframe = get_qa_with_reference(client)
spans_dataframe.head()
```

本例 Ragas 使用 LangChain 评估应用数据。为 LangChain 添加 OpenInference 埋点，可以观察评估过程内部的调用。


```python
from openinference.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()
```

评估追踪数据，并以 DataFrame 查看得分。


```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
)

evaluation_result = evaluate(
    dataset=ragas_eval_dataset,
    metrics=[faithfulness, answer_correctness, context_recall, context_precision],
)
eval_scores_df = pd.DataFrame(evaluation_result.scores)
```

将评估结果发送到 Phoenix，使其作为 span 的标注显示。


```python
from phoenix.trace import SpanEvaluations

# Assign span ids to your ragas evaluation scores (needed so Phoenix knows where to attach the spans).
eval_data_df = pd.DataFrame(evaluation_result.dataset)
assert eval_data_df.question.to_list() == list(
    reversed(spans_dataframe.input.to_list())  # The spans are in reverse order.
), "Phoenix spans are in an unexpected order. Re-start the notebook and try again."
eval_scores_df.index = pd.Index(
    list(reversed(spans_dataframe.index.to_list())), name=spans_dataframe.index.name
)

# Log the evaluations to Phoenix.
for eval_name in eval_scores_df.columns:
    evals_df = eval_scores_df[[eval_name]].rename(columns={eval_name: "score"})
    evals = SpanEvaluations(eval_name, evals_df)
    px.Client().log_evaluations(evals)
```

在 Phoenix 中即可看到应用 span 上的 Ragas 评估标注。


```python
print(session.url)
```

![span 上的 Ragas 评估标注](https://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/ragas/ragas_evaluation_annotations.gif)

## 7. 可视化并分析嵌入 { #7-visualize-and-analyze-your-embeddings }

[嵌入向量](https://arize.com/blog-course/embeddings-meaning-examples-and-how-to-compute/)编码了文档与查询的语义，既是 RAG 的基础，也有助于理解和调试 LLM 应用。

Phoenix 对高维嵌入降维并聚类，形成具有语义意义的数据分组。你可以选择忠实度或答案正确性等指标，直观检查表现，发现问题簇。这种方式能够分析细粒度、有实际意义的子集，帮助判断应用在哪类查询上表现较差。

重新启动 Phoenix 作为嵌入可视化工具，查看测试集表现。


```python
query_embeddings_df = query_embeddings_df.iloc[::-1]
assert ragas_evals_df.question.tolist() == query_embeddings_df.text.tolist()
assert test_df.question.tolist() == ragas_evals_df.question.tolist()
query_df = pd.concat(
    [
        ragas_evals_df[["question", "answer", "ground_truth"]].reset_index(drop=True),
        query_embeddings_df[["vector"]].reset_index(drop=True),
        test_df[["evolution_type"]],
        eval_scores_df.reset_index(drop=True),
    ],
    axis=1,
)
query_df.head()
```


```python
query_schema = px.Schema(
    prompt_column_names=px.EmbeddingColumnNames(
        raw_data_column_name="question", vector_column_name="vector"
    ),
    response_column_names="answer",
)
corpus_schema = px.Schema(
    prompt_column_names=px.EmbeddingColumnNames(
        raw_data_column_name="text", vector_column_name="vector"
    )
)
# relaunch phoenix with a primary and corpus dataset to view embeddings
px.close_app()
session = px.launch_app(
    primary=px.Dataset(query_df, query_schema, "query"),
    corpus=px.Dataset(corpus_df.reset_index(drop=True), corpus_schema, "corpus"),
)
```

启动后，按以下步骤查看所选指标。

- 选择 `vector` 嵌入。
- 选择 `Color By > dimension`，再选择字段，例如忠实度或答案正确性，以颜色表示得分。
- 在 `metric` 下拉菜单中选择指标，查看各簇的聚合结果。

![检查嵌入簇、聚合指标并按得分着色](https://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/ragas/ragas_correctness_clusters.gif)

## 8. 回顾 { #8-recap }

你已经使用 Ragas 与 Phoenix 构建并评估了 LlamaIndex 查询引擎，主要完成以下工作。

- 通过 Ragas 生成测试集，并计算忠实度和答案正确性等指标。
- 通过 OpenInference 为查询引擎埋点，观察 LlamaIndex 与 Ragas 的内部过程。
- 通过 Phoenix 收集 span 与 trace，导入评估结果，并可视化查询和文档，定位表现较差的数据群。

更多功能参阅 [Ragas 文档](https://docs.ragas.io/en/stable/)与 [Phoenix 文档](https://docs.arize.com/phoenix/)。

如果喜欢本教程，欢迎在 GitHub 点亮 ⭐。

- [Ragas](https://github.com/vibrantlabsai/ragas)
- [Phoenix](https://github.com/Arize-ai/phoenix)
- [OpenInference](https://github.com/Arize-ai/openinference)
