# 根据文档创建自定义单跳查询 { #create-custom-single-hop-queries-from-your-documents }

### 加载示例文档 { #load-sample-documents }
这里使用 [GitLab 员工手册示例](https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown)，可以运行以下命令下载。

```
! git clone https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown

```

```python
from langchain_community.document_loaders import DirectoryLoader


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

### 设置变换 { #setup-the-transforms }


这里配置文档标题提取、关键短语提取和基于标题的切分。
- 标题提取器，从文档中提取标题
- 关键短语提取器，从文档中提取关键短语
- 标题切分器，根据标题将文档切分为节点



```python
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
)


headline_extractor = HeadlinesExtractor(llm=llm)
headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
keyphrase_extractor = KeyphrasesExtractor(
    llm=llm, property_name="keyphrases", max_num=10
)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
]

apply_transforms(kg, transforms=transforms)
```

输出如下。
```
Applying KeyphrasesExtractor:   6%| | 2/36 [00:01<00:20,  1Property 'keyphrases' already exists in node '514fdc'. Skipping!
Applying KeyphrasesExtractor:  11%| | 4/36 [00:01<00:10,  2Property 'keyphrases' already exists in node '84a0f6'. Skipping!
Applying KeyphrasesExtractor:  64%|▋| 23/36 [00:03<00:01,  Property 'keyphrases' already exists in node '93f19d'. Skipping!
Applying KeyphrasesExtractor:  72%|▋| 26/36 [00:04<00:00, 1Property 'keyphrases' already exists in node 'a126bf'. Skipping!
Applying KeyphrasesExtractor:  81%|▊| 29/36 [00:04<00:00,  Property 'keyphrases' already exists in node 'c230df'. Skipping!
Applying KeyphrasesExtractor:  89%|▉| 32/36 [00:04<00:00, 1Property 'keyphrases' already exists in node '4f2765'. Skipping!
Property 'keyphrases' already exists in node '4a4777'. Skipping!
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

##

## 单跳查询 { #singlehop-query }

继承 `SingleHopQuerySynthesizer`，修改用于生成查询场景的函数。

**步骤如下。**
- 找出符合查询生成条件的节点，这里选择所有已提取关键短语的节点
- 对每组符合条件的节点执行以下操作
    - 将关键短语匹配到一个或多个用户画像
    - 创建节点、画像、查询风格与查询长度的所有可能组合
    - 从组合中采样所需数量的查询


```python
from ragas.testset.synthesizers.single_hop import (
    SingleHopQuerySynthesizer,
    SingleHopScenario,
)
from dataclasses import dataclass
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)


@dataclass
class MySingleHopScenario(SingleHopQuerySynthesizer):

    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):

        property_name = "keyphrases"
        nodes = []
        for node in knowledge_graph.nodes:
            if node.type.name == "CHUNK" and node.get_property(property_name):
                nodes.append(node)

        number_of_samples_per_node = max(1, n // len(nodes))

        scenarios = []
        for node in nodes:
            if len(scenarios) >= n:
                break
            themes = node.properties.get(property_name, [""])
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            base_scenarios = self.prepare_combinations(
                node,
                themes,
                personas=persona_list,
                persona_concepts=persona_concepts.mapping,
            )
            scenarios.extend(
                self.sample_combinations(base_scenarios, number_of_samples_per_node)
            )

        return scenarios

query = MySingleHopScenario(llm=llm)

scenarios = await query.generate_scenarios(
    n=5, knowledge_graph=kg, persona_list=persona_list
)

scenarios[0]
```
输出如下。
```
SingleHopScenario(
nodes=1
term=what is an ally
persona=name='Hiring manager at gitlab' role_description='A hiring manager at gitlab trying to underestand hiring policies in gitlab'
style=Web search like queries
length=long)
```



```python
result = await query.generate_sample(scenario=scenarios[-1])
```

### 修改提示词，定制查询风格 { #modify-prompt-to-customize-the-query-style }
此步骤可选。这里替换默认提示词，要求只生成是/否问题。


```python
instruction = """Generate a Yes/No query and answer based on the specified conditions (persona, term, style, length)
and the provided context. Ensure the answer is entirely faithful to the context, using only the information
directly from the provided context.

### Instructions:
1. **Generate a Yes/No Query**: Based on the context, persona, term, style, and length, create a question
that aligns with the persona's perspective, incorporates the term, and can be answered with 'Yes' or 'No'.
2. **Generate an Answer**: Using only the content from the provided context, provide a 'Yes' or 'No' answer
to the query. Do not add any information not included in or inferable from the context."""
```


```python
prompt = query.get_prompts()["generate_query_reference_prompt"]
prompt.instruction = instruction
query.set_prompts(**{"generate_query_reference_prompt": prompt})
result = await query.generate_sample(scenario=scenarios[-1])
```


```python
result.user_input
```
输出如下。
```
'Does the Diversity, Inclusion & Belonging (DIB) Team at GitLab have a structured approach to encourage collaborations among team members through various communication methods?'
```

```python
result.reference
```
输出如下。
```
'Yes'
```
