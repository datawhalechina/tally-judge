# 使用 Ragas 为 RAG 问答生成合成测试集 { #generating-a-synthetic-test-set-for-rag-based-question-answering-with-ragas }

## 概览 { #overview }

本教程使用 **Ragas 测试集生成模块**，为 **RAG 问答机器人**创建**合成测试集**。示例是 Ragas 航空助手，需要回答以下主题的问题。

- 航班预订
- 航班变更与取消
- 行李政策
- 查看预订
- 航班延误
- 机上服务
- 特殊协助

为使合成数据尽可能**真实、多样**，我们创建**不同的客户画像**，体现各类旅客及其行为，构建**全面且具有代表性的测试集**，评估 RAG 模型的有效性与稳健性。

下面开始。

## 下载并加载文档 { #download-and-load-documents }

运行以下命令下载虚构的 Ragas 航空数据集，再通过 LangChain 加载。

```sh
! git clone https://huggingface.co/datasets/vibrantlabsai/ragas-airline-dataset
```

```python
from langchain_community.document_loaders import DirectoryLoader

path = "ragas-airline-dataset"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
```

## 设置 LLM 与嵌入模型 { #set-up-the-llm-and-embedding-model }


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import openai


generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
openai_client = openai.OpenAI()
generator_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
```

## 创建知识图谱 { #create-knowledge-graph }

使用文档创建基础知识图谱。


```python
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType


kg = KnowledgeGraph()

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

kg
```
输出如下。
```
KnowledgeGraph(nodes: 8, relationships: 0)
```


## 设置变换 { #setup-the-transforms }

本例仅使用节点构建知识图谱，生成单跳查询，并应用以下三种变换，丰富图谱信息。

- **标题提取**。使用语言模型从文档中提取明确的章节标题，例如从 *flight cancellations.md* 提取“航空公司发起的取消”，为问题生成提供聚焦的主题上下文。
- **按标题切分**。根据提取的标题划分文档，增加节点数量，使查询围绕更细粒度的具体上下文生成。
- **关键短语提取**。提取核心主题短语，例如座位信息，作为语义种子，增强查询的多样性与相关性。


```python
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor

headline_extractor = HeadlinesExtractor(llm=generator_llm, max_num=20)
headline_splitter = HeadlineSplitter(max_tokens=1500)
keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor
]

apply_transforms(kg, transforms=transforms)
```
```
Applying HeadlinesExtractor: 100%|██████████| 8/8 [00:00<?, ?it/s]
Applying HeadlineSplitter: 100%|██████████| 8/8 [00:00<?, ?it/s]
Applying KeyphrasesExtractor: 100%|██████████| 25/25 [00:00<?, ?it/s]
```

## 为查询生成配置用户画像 { #configuring-personas-for-query-generation }

画像为问题提供用户视角，让查询更自然、具体和多样，覆盖不同场景。

- **首次乘机旅客**，关注详细、逐步的指导，需要明确说明。
- **常旅客**，提问简洁，重视效率。
- **不满的商务舱旅客**，语气急切且挑剔，期望较高，希望立即解决问题。


```python
from ragas.testset.persona import Persona

persona_first_time_flier = Persona(
    name="First Time Flier",
    role_description="Is flying for the first time and may feel anxious. Needs clear guidance on flight procedures, safety protocols, and what to expect throughout the journey.",
)

persona_frequent_flier = Persona(
    name="Frequent Flier",
    role_description="Travels regularly and values efficiency and comfort. Interested in loyalty programs, express services, and a seamless travel experience.",
)

persona_angry_business_flier = Persona(
    name="Angry Business Class Flier",
    role_description="Demands top-tier service and is easily irritated by any delays or issues. Expects immediate resolutions and is quick to express frustration if standards are not met.",
)

personas = [persona_first_time_flier, persona_frequent_flier, persona_angry_business_flier]
```

## 使用合成器生成查询 { #query-generation-using-synthesizers }

合成器将补充了属性的节点与画像转化为查询。它选择节点属性（如 entities 或 keyphrases），与画像、风格和查询长度组合，再使用 LLM 根据节点内容生成问答对。

使用两个 `SingleHopSpecificQuerySynthesizer` 实例定义查询分布。

- **基于标题的合成器**，围绕提取的文档标题生成问题，关注具体章节。
- **基于关键短语的合成器**，围绕关键概念生成更广泛的主题问题。

两者权重均为 0.5，使具体与概念性问题均衡，提升测试集多样性。


```python
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

query_distibution = [
    (
        SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"),
        0.5,
    ),
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm, property_name="keyphrases"
        ),
        0.5,
    ),
]
```

## 测试集生成 { #testset-generation }


```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
    persona_list=personas,
)
```

现在可以生成测试集。


```python
testset = generator.generate(testset_size=10, query_distribution=query_distibution)
testset.to_pandas()
```
```
Generating Scenarios: 100%|██████████| 2/2 [00:00<?, ?it/s]
Generating Samples: 100%|██████████| 10/10 [00:00<?, ?it/s]
```
输出如下。



<div>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>reference_contexts</th>
      <th>reference</th>
      <th>synthesizer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wut do I do if my baggage is Delayed, Lost, or...</td>
      <td>[Baggage Policies\n\nThis section provides a d...</td>
      <td>If your baggage is delayed, lost, or damaged, ...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wht asistance is provided by the airline durin...</td>
      <td>[Flight Delays\n\nFlight delays can be caused ...</td>
      <td>Depending on the length of the delay, Ragas Ai...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What is Step 1: Check Fare Rules in the contex...</td>
      <td>[Flight Cancellations\n\nFlight cancellations ...</td>
      <td>Step 1: Check Fare Rules involves logging into...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How can I access my booking online with Ragas ...</td>
      <td>[Managing Reservations\n\nManaging your reserv...</td>
      <td>To access your booking online with Ragas Airli...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What assistance does Ragas Airlines provide fo...</td>
      <td>[Special Assistance\n\nRagas Airlines provides...</td>
      <td>Ragas Airlines provides special assistance ser...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>5</th>
      <td>What steps should I take if my baggage is dela...</td>
      <td>[Baggage Policies This section provides a deta...</td>
      <td>If your baggage is delayed, lost, or damaged w...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>6</th>
      <td>How can I resubmit the claim for my baggage is...</td>
      <td>[Potential Issues and Resolutions for Baggage ...</td>
      <td>To resubmit the claim for your baggage issue, ...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wut are the main causes of flight delays and h...</td>
      <td>[Flight Delays Flight delays can be caused by ...</td>
      <td>Flight delays can be caused by weather conditi...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>8</th>
      <td>How can I request reimbursement for additional...</td>
      <td>[2. Additional Expenses Incurred Due to Delay ...</td>
      <td>To request reimbursement for additional expens...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>9</th>
      <td>What are passenger-initiated cancelations?</td>
      <td>[Flight Cancellations Flight cancellations can...</td>
      <td>Passenger-initiated cancellations occur when a...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
  </tbody>
</table>
</div>



## 结语 { #final-thoughts }

本教程介绍了通过 Ragas 生成测试集，重点是单跳查询。后续教程将进一步介绍多跳查询，构建更丰富的测试场景。
