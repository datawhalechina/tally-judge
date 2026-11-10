# 根据文档创建自定义多跳查询 { #create-custom-multi-hop-queries-from-your-documents }

本教程介绍如何根据文档创建自定义多跳查询，生成标准查询类型难以覆盖的问题，使测试更贴合具体场景。

### 加载示例文档 { #load-sample-documents }
这里使用 [GitLab 员工手册示例](https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown)，可以运行以下命令下载。


```python
! git clone https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown
```


```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
```

### 创建知识图谱 { #create-kg }

使用文档创建基础知识图谱。


```python
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType


kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )
```

### 设置 LLM 与嵌入模型 { #set-up-the-llm-and-embedding-model }
你可以选择[任意合适的模型](../customize_models.md)，这里使用 OpenAI 模型。


```python
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

openai_client = OpenAI()
llm = llm_factory("gpt-4o-mini", client=openai_client)
embedding = OpenAIEmbeddings(client=openai_client)
```

### 设置提取器与关系构建器 { #setup-extractors-and-relationship-builders }

创建多跳查询前，需要确定可共同用于生成问题的文档。Ragas 根据文档或节点间的关系筛选节点。例如，节点 A 与节点 B 通过实体或关键短语重合建立联系后，即可围绕它们创建多跳查询。

这里配置两类提取器，并结合标题切分与重合关系构建。
- 标题提取器，从文档中提取标题
- 关键短语提取器，从文档中提取关键短语
- 标题切分器，根据标题将文档切分为节点
- OverlapScore 构建器，根据关键短语重合建立节点关系


```python
from ragas.testset.transforms import Parallel, apply_transforms
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    OverlapScoreBuilder,
)


headline_extractor = HeadlinesExtractor(llm=llm)
headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
keyphrase_extractor = KeyphrasesExtractor(
    llm=llm, property_name="keyphrases", max_num=10
)
relation_builder = OverlapScoreBuilder(
    property_name="keyphrases",
    new_property_name="overlap_score",
    threshold=0.01,
    distance_threshold=0.9,
)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
    relation_builder,
]

apply_transforms(kg, transforms=transforms)
```
输出如下。
```
Applying KeyphrasesExtractor:   6%|██████▏                                                                                                         | 2/36 [00:01<00:17,  1.94it/s]Property 'keyphrases' already exists in node 'a2f389'. Skipping!
Applying KeyphrasesExtractor:  17%|██████████████████▋                                                                                             | 6/36 [00:01<00:04,  6.37it/s]Property 'keyphrases' already exists in node '3068c0'. Skipping!
Applying KeyphrasesExtractor:  53%|██████████████████████████████████████████████████████████▌                                                    | 19/36 [00:02<00:01,  8.88it/s]Property 'keyphrases' already exists in node '854bf7'. Skipping!
Applying KeyphrasesExtractor:  78%|██████████████████████████████████████████████████████████████████████████████████████▎                        | 28/36 [00:03<00:00,  9.73it/s]Property 'keyphrases' already exists in node '2eeb07'. Skipping!
Property 'keyphrases' already exists in node 'd68f83'. Skipping!
Applying KeyphrasesExtractor:  83%|████████████████████████████████████████████████████████████████████████████████████████████▌                  | 30/36 [00:03<00:00,  9.35it/s]Property 'keyphrases' already exists in node '8fdbea'. Skipping!
Applying KeyphrasesExtractor:  89%|██████████████████████████████████████████████████████████████████████████████████████████████████▋            | 32/36 [00:04<00:00,  7.76it/s]Property 'keyphrases' already exists in node 'ef6ae0'. Skipping!
```

### 配置用户画像 { #configure-personas }

也可以使用[自动画像生成器](_persona_generator.md)完成。


```python
from ragas.testset.persona import Persona

person1 = Persona(
    name="gitlab employee",
    role_description="A junior gitlab employee curious on workings on gitlab",
)
persona2 = Persona(
    name="Hiring manager at gitlab",
    role_description="A hiring manager at gitlab trying to underestand hiring policies in gitlab",
)
persona_list = [person1, persona2]
```

### 创建多跳查询 { #create-multi-hop-query }

继承 `MultiHopQuerySynthesizer`，修改用于生成查询场景的函数。

**步骤如下。**
- 根据节点间的关系，找出符合条件的（nodeA、relationship、nodeB）组合
- 对每组符合条件的节点执行以下操作
    - 将关键短语匹配到一个或多个用户画像
    - 创建节点组、画像、查询风格与查询长度的所有可能组合
    - 从组合中采样所需数量的查询



```python
from dataclasses import dataclass
import typing as t
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)


@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):

    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph,
        persona_list,
        callbacks,
    ) -> t.List[MultiHopScenario]:

        # query and get (node_a, rel, node_b) to create multi-hop queries
        results = kg.find_two_nodes_single_rel(
            relationship_condition=lambda rel: (
                True if rel.type == "keyphrases_overlap" else False
            )
        )

        num_sample_per_triplet = max(1, n // len(results))

        scenarios = []
        for triplet in results:
            if len(scenarios) < n:
                node_a, node_b = triplet[0], triplet[-1]
                overlapped_keywords = triplet[1].properties["overlapped_items"]
                if overlapped_keywords:

                    # match the keyword with a persona for query creation
                    themes = list(dict(overlapped_keywords).keys())
                    prompt_input = ThemesPersonasInput(
                        themes=themes, personas=persona_list
                    )
                    persona_concepts = (
                        await self.theme_persona_matching_prompt.generate(
                            data=prompt_input, llm=self.llm, callbacks=callbacks
                        )
                    )

                    overlapped_keywords = [list(item) for item in overlapped_keywords]

                    # prepare and sample possible combinations
                    base_scenarios = self.prepare_combinations(
                        [node_a, node_b],
                        overlapped_keywords,
                        personas=persona_list,
                        persona_item_mapping=persona_concepts.mapping,
                        property_name="keyphrases",
                    )

                    # get number of required samples from this triplet
                    base_scenarios = self.sample_diverse_combinations(
                        base_scenarios, num_sample_per_triplet
                    )

                    scenarios.extend(base_scenarios)

        return scenarios

query = MyMultiHopQuery(llm=llm)
scenarios = await query.generate_scenarios(
    n=10, knowledge_graph=kg, persona_list=persona_list
)

scenarios[4]
```
输出如下。
```
MultiHopScenario(
nodes=2
combinations=['Diversity Inclusion & Belonging', 'Diversity, Inclusion & Belonging Goals']
style=Web search like queries
length=short
persona=name='Hiring manager at gitlab' role_description='A hiring manager at gitlab trying to underestand hiring policies in gitlab')
```

### 运行多跳查询生成 { #run-the-multi-hop-query }


```python
result = await query.generate_sample(scenario=scenarios[-1])
result.user_input
```

输出如下。
```
'How does GitLab ensure that its DIB roundtables are effective in promoting diversity and inclusion?'
```


现在已经创建了多跳查询。通过建立和探索文档间的关系，你可以进一步生成其他类型的问题。

##
