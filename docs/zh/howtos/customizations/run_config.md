# 自定义超时与速率限制 { #customize-timeouts-and-rate-limits }

使用 collections API 和 `llm_factory` 时，直接在 LLM 客户端上配置超时与重试。

## OpenAI 客户端配置 { #openai-client-configuration }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Configure timeout and retries on the client
client = AsyncOpenAI(
    timeout=60.0,        # 60 second timeout
    max_retries=5,       # Retry up to 5 times on failures
)

llm = llm_factory("gpt-4o-mini", client=client)

# Use with metrics
scorer = Faithfulness(llm=llm)
result = scorer.score(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)
```

### 可用选项 { #available-options }

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `timeout` | 600.0 | 请求超时时间，单位为秒 |
| `max_retries` | 2 | 请求失败后的重试次数 |

### 细粒度超时控制 { #fine-grained-timeout-control }

如需分别控制不同类型的超时，可以使用以下配置。

```python
import httpx
from openai import AsyncOpenAI

client = AsyncOpenAI(
    timeout=httpx.Timeout(
        60.0,           # Total timeout
        connect=5.0,    # Connection timeout
        read=30.0,      # Read timeout
        write=10.0,     # Write timeout
    ),
    max_retries=3,
)
```

!!! tip "服务商文档"
    各 LLM 服务商的客户端配置选项有所不同，请参阅对应 SDK 文档。

    - [OpenAI Python SDK](https://github.com/openai/openai-python)
    - [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)


## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API 和 `RunConfig`。新项目建议使用上文 collections API 的客户端级配置方式。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到基于 collections 的 API。

### RunConfig 参数 { #runconfig-parameters }

```python
from ragas.run_config import RunConfig

run_config = RunConfig(
    timeout=180,        # Max seconds per operation (default: 180)
    max_retries=10,     # Retry attempts (default: 10)
    max_wait=60,        # Max seconds between retries (default: 60)
    max_workers=16,     # Concurrent workers (default: 16)
    log_tenacity=False, # Log retry attempts (default: False)
    seed=42,            # Random seed (default: 42)
)
```

### 与 evaluate 一起使用 { #usage-with-evaluate }

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig

# Legacy LLM setup
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# Configure run settings
run_config = RunConfig(max_workers=64, timeout=60)

# Use with evaluate
results = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(llm=llm)],
    run_config=run_config,
)
```
