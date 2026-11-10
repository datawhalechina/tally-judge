## 从非英语语料生成合成测试数据 { #synthetic-test-generation-from-non-english-corpus }

本教程介绍如何将合成测试数据生成流程适配到非英语语料，示例使用西班牙语维基百科文章生成西班牙语查询。

### 下载并加载语料 { #download-and-load-corpus }


```python
! git clone https://huggingface.co/datasets/vibrantlabsai/Sample_non_english_corpus
```

    Cloning into 'Sample_non_english_corpus'...
    remote: Enumerating objects: 12, done.[K
    remote: Counting objects: 100% (8/8), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 12 (delta 0), reused 0 (delta 0), pack-reused 4 (from 1)[K
    Unpacking objects: 100% (12/12), 11.43 KiB | 780.00 KiB/s, done.



```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader


path = "Sample_non_english_corpus/"
loader = DirectoryLoader(path, glob="**/*.txt")
docs = loader.load()
```

    /opt/homebrew/Caskroom/miniforge/base/envs/ragas/lib/python3.9/site-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.20) or chardet (5.2.0)/charset_normalizer (None) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "



```python
len(docs)
```




    6



### 初始化所需模型 { #initialize-required-models }


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import openai

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
openai_client = openai.OpenAI()
generator_embeddings = OpenAIEmbeddings(client=openai_client)
```

    /opt/homebrew/Caskroom/miniforge/base/envs/ragas/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


### 设置用户画像与变换 { #setup-persona-and-transforms }
可以通过[此教程](_persona_generator.md)自动生成用户画像。为简化示例，这里使用预定义画像、两个基本变换和简单的查询分布。


```python
from ragas.testset.persona import Persona

personas = [
    Persona(
        name="curious student",
        role_description="A student who is curious about the world and wants to learn more about different cultures and languages",
    ),
]
```


```python
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter

transforms = [HeadlineSplitter(), NERExtractor()]
```

### 初始化测试集生成器 { #initialize-test-generator }


```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas
)
```

### 加载并适配查询 { #load-and-adapt-queries }

加载所需查询类型，并适配到目标语言。


```python
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
]

for query, _ in distribution:
    prompts = await query.adapt_prompts("spanish", llm=generator_llm)
    query.set_prompts(**prompts)
```

### 生成 { #generate }


```python
dataset = generator.generate_with_langchain_docs(
    docs[:],
    testset_size=5,
    transforms=transforms,
    query_distribution=distribution,
)
```

    Applying HeadlineSplitter:   0%|          | 0/6 [00:00<?, ?it/s]unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    Generating Scenarios: 100%|██████████| 1/1 [00:07<00:00,  7.75s/it]
    Generating Samples: 100%|██████████| 5/5 [00:03<00:00,  1.65it/s]



```python
eval_dataset = dataset.to_evaluation_dataset()
```


```python
print("Query:", eval_dataset[0].user_input)
print("Reference:", eval_dataset[0].reference)
```

    Query: Quelles sont les caractéristiques du Bronx en tant que borough de New York?
    Reference: Le Bronx est l'un des cinq arrondissements de New York, qui est la plus grande ville des États-Unis. Bien que le contexte ne fournisse pas de détails spécifiques sur le Bronx, il mentionne que New York est une ville cosmopolite avec de nombreux quartiers ethniques, ce qui pourrait inclure des caractéristiques culturelles variées présentes dans le Bronx.


至此完成。你可以根据需要定制测试集生成过程。


