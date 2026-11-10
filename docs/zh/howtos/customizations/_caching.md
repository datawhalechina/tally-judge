# Ragas 缓存 { #caching-in-ragas }

缓存可以避免重复计算，加快评估与测试集生成。Ragas 使用精确匹配缓存，存储 LLM 和嵌入模型的响应。

[DiskCacheBackend][ragas.cache.DiskCacheBackend] 将响应缓存在本地磁盘。你也可以实现 [CacheInterface][ragas.cache.CacheInterface]，创建自定义缓存后端。

## 为新版 LLM 与嵌入接口启用缓存 { #using-caching-with-modern-llms-and-embeddings }

新版 collections 指标与实验通过简单的接口支持缓存。

### 快速开始 { #quick-start }

```python
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from openai import OpenAI

# Create cache once
cache = DiskCacheBackend()

# Use with LLM factory
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# All LLM calls are now cached!
from pydantic import BaseModel

class Response(BaseModel):
    answer: str

response = llm.generate("Evaluate this...", Response)
```

### 在 llm_factory 中使用缓存 { #caching-with-llm_factory }

```python
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from openai import OpenAI

# Create cache instance
cache = DiskCacheBackend()

# Create LLM with caching
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# First call - makes API request and caches result
response1 = llm.generate("Evaluate this text", Response)

# Second call - returns cached result instantly
response2 = llm.generate("Evaluate this text", Response)

# Result: Same output, 60x faster, $0 cost
```

### 在 embedding_factory 中使用缓存 { #caching-with-embedding_factory }

```python
from ragas.cache import DiskCacheBackend
from ragas.embeddings import embedding_factory
from openai import OpenAI

cache = DiskCacheBackend()
client = OpenAI(api_key="...")

embeddings = embedding_factory("openai", client=client, cache=cache)

# First call - makes API request
vector1 = embeddings.embed_text("Some text to embed")

# Second call - instant cache hit
vector2 = embeddings.embed_text("Some text to embed")

assert vector1 == vector2  # Identical results
```

### 在实验中使用缓存 { #caching-in-experiments }

重复运行相同评估的实验尤其适合使用缓存。

```python
from ragas import experiment, Dataset
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

# Setup cached LLM once
cache = DiskCacheBackend()
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# Use in metric
metric = FactualCorrectness(llm=llm)

@experiment()
async def evaluate_model(row):
    score = metric.score(
        response=row["response"],
        reference=row["reference"]
    )
    return {
        **row,
        "factual_correctness": score.value,
        "reason": score.reason
    }

# Load your dataset
dataset = Dataset.from_list([
    {"response": "Paris is the capital of France", "reference": "Paris"},
    {"response": "London is the capital of UK", "reference": "London"},
])

# First run - makes API calls and caches results
print("First run (populating cache)...")
results1 = await evaluate_model.arun(dataset)
# Takes ~2 seconds for 2 samples

# Second run - uses cache, nearly instant!
print("Second run (using cache)...")
results2 = await evaluate_model.arun(dataset)
# Takes ~0.1 seconds for 2 samples

# Results are identical, but 20x faster!
```

### 缓存管理 { #cache-management }

#### 清空缓存 { #clearing-the-cache }

```python
# Clear all cached data
cache = DiskCacheBackend()
cache.cache.clear()
```

#### 设置容量上限 { #setting-size-limits }

```python
# Limit cache to 1GB
cache = DiskCacheBackend()
cache.cache.reset('size_limit', 1e9)  # 1GB
cache.cache.reset('cull_limit', 10)   # Remove 10% when full
```

#### 缓存位置 { #cache-location }

默认缓存存放于 `.cache/` 目录，也可以更改。

```python
cache = DiskCacheBackend(cache_dir="my_custom_cache")
```

### 缓存的收益 { #benefits-of-caching }

1. **节约成本**。相同输入可避免重复 API 调用，示例场景可节省 50～60%。
2. **提升速度**。命中缓存的调用几乎立即返回，示例场景可提速 60 倍以上。
3. **便于开发**。减少等待 API 返回的时间，加快迭代。
4. **可复现**。相同输入返回相同结果。

以下条件同时满足时会命中缓存。

- ✅ 提示词或文本完全相同
- ✅ 模型参数相同，例如 temperature、max_tokens 等
- ✅ LLM 的响应模型或结构相同

以下情况会导致缓存未命中。

- ❌ 提示词或文本不同
- ❌ 参数不同
- ❌ 响应模型不同

### 不适合缓存的情况 { #anti-patterns-when-not-to-cache }

- ❌ **含不确定内容的提示词**，例如随机元素或时间戳
- ❌ **高温度设置**，例如 temperature > 0.7，预期回答变化较大
- ❌ **流式响应**，缓存不支持流式输出
- ❌ **实时数据**，响应需要反映当前状态

### 不同运行环境的说明 { #environment-specific-notes }

**Notebook**。缓存在单元格执行之间、内核重启之后仍会保留。

**Web 应用**。跨请求共享缓存可以改善性能。

**无服务器函数**。使用 `/tmp` 目录。

```python
cache = DiskCacheBackend(cache_dir="/tmp/.cache")
```

**分布式工作进程**。缓存支持多进程安全访问，高吞吐系统可以考虑通过 `CacheInterface` 实现 Redis 后端。

### 性能示例 { #performance-expectations }

| 场景 | 耗时 | 成本 |
|----------|------|------|
| 首次运行（100 个样本） | 约 2 分钟 | 0.50 美元 |
| 再次运行（命中缓存） | 约 2 秒 | 0.00 美元 |
| **提升** | **提速 60 倍** | **节省 100% 成本** |

---

## 旧版缓存（已弃用） { #legacy-caching-deprecated }

!!! warning "已弃用"
    使用 `LangchainLLMWrapper` 的方式已弃用，将在 v1.0 移除。请使用上文结合 `llm_factory()` 与 `embedding_factory()` 的新版方式。

### 为 LangchainLLMWrapper 使用旧版缓存 { #using-legacy-caching-with-langchainllmwrapper }

下面介绍如何将 [DiskCacheBackend][ragas.cache.DiskCacheBackend] 用于旧版 LLM 和嵌入模型。

```python
from ragas.cache import DiskCacheBackend

cacher = DiskCacheBackend()

# check if the cache is empty and clear it
print(len(cacher.cache))
cacher.cache.clear()
print(len(cacher.cache))
```

创建启用缓存的 LLM 和嵌入模型。这里以 [langchain-openai](https://github.com/langchain-ai/langchain-openai) 的 `ChatOpenAI` 为例。

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

cached_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"), cache=cacher)
```

```python
# if you want to see the cache in action, set the logging level to debug
import logging
from ragas.utils import set_logging_level

set_logging_level("ragas.cache", logging.DEBUG)
```

接下来运行一次简单评估。

```python
from ragas import evaluate
from ragas import EvaluationDataset

from ragas.metrics import FactualCorrectness, AspectCritic
from datasets import load_dataset

# Define Answer Correctness with AspectCritic
answer_correctness = AspectCritic(
    name="answer_correctness",
    definition="Is the answer correct? Does it match the reference answer?",
    llm=cached_llm,
)

metrics = [answer_correctness, FactualCorrectness(llm=cached_llm)]

# load the dataset
dataset = load_dataset(
    "vibrantlabsai/amnesty_qa", "english_v3", trust_remote_code=True
)
eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

# evaluate the dataset
results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
)

results
```

在本地机器上，这次运行耗时接近 2 分钟。再次执行，看看缓存效果。

```python
results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
)

results
```

结果几乎立即返回。

同样可以为测试集生成启用缓存，只需将 `generator_llm` 替换为启用缓存的版本。详情参阅[测试集生成](../../getstarted/rag_testset_generation.md)。
