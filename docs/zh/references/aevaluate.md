# 异步评估 { #async-evaluation }

## aevaluate()

::: ragas.evaluation.aevaluate

## 异步用法 { #async-usage }

Ragas 提供同步与异步评估 API，适用于不同场景。

### 使用 aevaluate()（推荐用于生产环境） { #using-aevaluate-recommended-for-production }

在生产环境的异步应用中，使用 `aevaluate()` 避免事件循环冲突。

```python
import asyncio
from ragas import aevaluate

async def evaluate_app():
    result = await aevaluate(dataset, metrics)
    return result

# In your async application
result = await evaluate_app()
```

### 控制 evaluate() 的异步行为 { #using-evaluate-with-async-control }

为兼容旧版用法和 Jupyter notebook，`evaluate()` 提供对 `nest_asyncio` 的可选控制。

```python
# Default behavior (Jupyter-compatible)
result = evaluate(dataset, metrics)  # allow_nest_asyncio=True

# Production-safe (avoids event loop patching)
result = evaluate(dataset, metrics, allow_nest_asyncio=False)
```

### 解决 nest_asyncio 问题 { #migration-from-nest_asyncio-issues }

如果在生产环境遇到 `nest_asyncio` 相关问题，可以按以下方式迁移。

**修改前（存在问题）**
```python
# This may cause event loop conflicts
result = evaluate(dataset, metrics)
```

**修改后（已修正）**
```python
# Option 1: Use async API
result = await aevaluate(dataset, metrics)

# Option 2: Disable nest_asyncio
result = evaluate(dataset, metrics, allow_nest_asyncio=False)
```
