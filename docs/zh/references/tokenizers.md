# 分词器 { #tokenizers }

Ragas 支持多种分词器，在知识图谱操作和测试数据生成时进行文本切分。

## 概览 { #overview }

从知识图谱节点提取属性时，需要按 token 数量限制切分文本。Ragas 默认使用 OpenAI 的 tiktoken，也可以使用 HuggingFace 分词器，更好地匹配开源模型。

## 可用分词器 { #available-tokenizers }

### TiktokenWrapper

OpenAI tiktoken 分词器的封装，也是默认分词器。

```python
from ragas import TiktokenWrapper

# Using default encoding (o200k_base)
tokenizer = TiktokenWrapper()

# Using a specific encoding
tokenizer = TiktokenWrapper(encoding_name="cl100k_base")

# Using encoding for a specific model
tokenizer = TiktokenWrapper(model_name="gpt-4")
```

### HuggingFaceTokenizer

HuggingFace transformers 分词器的封装，适用于开源模型。

```python
from ragas import HuggingFaceTokenizer

# Load tokenizer for a specific model
tokenizer = HuggingFaceTokenizer(model_name="meta-llama/Llama-2-7b-hf")

# Use a pre-initialized tokenizer
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer)
```

**注意**。HuggingFace 分词器需要 `transformers` 包，安装方法如下。
```sh
pip install transformers
# or
uv add transformers
```

### 工厂函数 { #factory-function }

使用 `get_tokenizer()` 可以便捷地创建分词器。

```python
from ragas import get_tokenizer

# Default tiktoken tokenizer
tokenizer = get_tokenizer()

# Tiktoken for a specific model
tokenizer = get_tokenizer("tiktoken", model_name="gpt-4")

# HuggingFace tokenizer
tokenizer = get_tokenizer("huggingface", model_name="meta-llama/Llama-2-7b-hf")
```

## 使用自定义分词器 { #using-custom-tokenizers }

### 结合基于 LLM 的提取器 { #with-llm-based-extractors }

所有基于 LLM 的提取器都接受 `tokenizer` 参数。

```python
from ragas import HuggingFaceTokenizer
from ragas.testset.transforms import (
    SummaryExtractor,
    KeyphrasesExtractor,
    HeadlinesExtractor,
)

# Create a HuggingFace tokenizer for your model
tokenizer = HuggingFaceTokenizer(model_name="meta-llama/Llama-2-7b-hf")

# Use it with extractors
summary_extractor = SummaryExtractor(llm=your_llm, tokenizer=tokenizer)
keyphrase_extractor = KeyphrasesExtractor(llm=your_llm, tokenizer=tokenizer)
headlines_extractor = HeadlinesExtractor(llm=your_llm, tokenizer=tokenizer)
```

### 实现自定义分词器 { #custom-tokenizer-implementation }

可以继承 `BaseTokenizer`，实现自己的分词器。

```python
from ragas.tokenizers import BaseTokenizer

class MyCustomTokenizer(BaseTokenizer):
    def __init__(self, ...):
        # Initialize your tokenizer
        pass

    def encode(self, text: str) -> list[int]:
        # Return token IDs
        pass

    def decode(self, tokens: list[int]) -> str:
        # Return decoded text
        pass
```

## API 参考 { #api-reference }

::: ragas.tokenizers
