# 使用已分块的数据生成测试集 { #using-pre-chunked-data-for-testset-generation }

如果已经有成熟的分块策略，可以直接向 Ragas 提供自己的片段，跳过内部文档切分，适用于以下场景。

- 已为特定领域优化分块策略
- 希望 RAG 流水线与评估过程使用一致的分块
- 已完成文档预处理，并附加自定义元数据
- 需要按特定业务逻辑或文档结构划分片段

## 概览 { #overview }

`TestsetGenerator` 的 `generate_with_chunks` 方法接收已分块数据，将每个片段直接视为 `NodeType.CHUNK`，跳过内部切分变换，完整保留提供的片段内容与元数据。

## 工作原理 { #how-it-works }

使用 `generate_with_chunks` 时，Ragas 会执行以下操作。

1. **接收片段**，支持 `Document` 对象或字符串
2. **应用提取器**，如 `SummaryExtractor`、`ThemesExtractor`、`NERExtractor` 和 `EmbeddingExtractor`，为片段补充属性
3. **建立片段间的关系**，使用 `CosineSimilarityBuilder` 与 `OverlapScoreBuilder`
4. **生成用户画像**，依据内容主题进行创建
5. **创建场景**，涵盖单跳、多跳等查询类型
6. **合成测试样本**，包含问题、上下文和参考答案

## 示例，使用已分块文档 { #example-using-pre-chunked-documents }

传入 LangChain `Document` 对象列表，可以保留片段元数据，便于追踪来源文档和其他自定义信息。

```python
import os
from langchain_core.documents import Document
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize generator with your preferred models
generator = TestsetGenerator(
    llm=llm_factory("gpt-4o-mini", client=client),
    embedding_model=OpenAIEmbeddings(client=client)
)

# Your pre-chunked documents
chunks = [
    Document(
        page_content="""The Eiffel Tower (Tour Eiffel) is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Locally nicknamed "La Dame de Fer" (French for "The Iron Lady"), it was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. Although initially criticized by some of France's leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognizable structures in the world.""",
        metadata={"source": "doc1", "chunk_id": 1}
    ),
    Document(
        page_content="""The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).""",
        metadata={"source": "doc1", "chunk_id": 2}
    )
]

# Generate testset
testset = generator.generate_with_chunks(
    chunks=chunks,
    testset_size=10
)

# Save to CSV
output_file = "testset.csv"
testset.to_csv(output_file)
print(f"Testset saved to {output_file}")
print(testset.to_pandas().head())
```

### 生成过程 { #generation-process }

生成期间会显示进度日志，展示不同的变换和合成阶段。

```
Applying SummaryExtractor: 100%|████████████████████████████████| 2/2 [00:07<00:00,  3.67s/it]
Applying CustomNodeFilter: 100%|█████████████████████████████| 2/2 [00:00<00:00, 2226.87it/s]
Applying EmbeddingExtractor: 100%|███████████████████████████| 2/2 [00:02<00:00,  1.19s/it]
Applying ThemesExtractor: 100%|██████████████████████████████| 2/2 [00:06<00:00,  3.07s/it]
Applying NERExtractor: 100%|█████████████████████████████████| 2/2 [00:06<00:00,  3.10s/it]
Applying CosineSimilarityBuilder: 100%|█████████████████████| 1/1 [00:00<00:00, 613.29it/s]
Applying OverlapScoreBuilder: 100%|████████████████████████| 1/1 [00:00<00:00, 1491.57it/s]
Generating personas: 100%|███████████████████████████████████| 2/2 [00:05<00:00,  2.77s/it]
Generating Scenarios: 100%|██████████████████████████████████| 2/2 [00:08<00:00,  4.19s/it]
Generating Samples: 100%|████████████████████████████████| 11/11 [00:45<00:00,  4.13s/it]
Testset saved to testset.csv
```


测试集包含不同类型的查询。
- **单跳查询**，仅靠单个片段即可回答
- **多跳查询**，需要多个片段的信息，前提是片段之间存在关联

## 示例，使用普通字符串 { #example-using-plain-strings }

如果无须保留元数据，也可以直接传入普通字符串。

```python
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Initialize models
client = OpenAI()
generator = TestsetGenerator(
    llm=llm_factory("gpt-4o-mini", client=client),
    embedding_model=OpenAIEmbeddings(client=client)
)

# Simple text chunks
text_chunks = [
    "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It involves machine learning, natural language processing, and computer vision.",
    "Machine Learning is a subset of AI that enables systems to learn from data without explicit programming. Popular algorithms include neural networks and decision trees.",
    "Deep Learning uses neural networks with multiple layers to process complex patterns in large datasets. It powers modern applications like image recognition and language translation."
]

# Generate testset
testset = generator.generate_with_chunks(
    chunks=text_chunks,
    testset_size=5
)

# Save to CSV
output_file = "testset.csv"
testset.to_csv(output_file)
print(f"Testset saved to {output_file}")
print(testset.to_pandas())
```

## 边界情况处理 { #handling-edge-cases }

- **空内容**。`page_content` 为空或仅含空白字符的片段会自动过滤。
- **空序列**。传入空片段序列时，会生成空测试集。
