# 使用可观测性工具跟踪和记录评估 { #tracing-and-logging-evaluations-with-observability-tools }

日志和追踪是语言模型应用的重要组成部分。本教程介绍如何跟踪 Ragas 评估。Ragas 的 `callbacks` 功能便于接入 LangSmith、wandb、Opik 等追踪器。这里以 LangSmith 为例。

首先设置 LangSmith 所需的环境变量，更多详情参阅其[文档](https://docs.smith.langchain.com/)。

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

然后从 LangChain 导入追踪器，这里使用 `LangChainTracer`，也可以使用其他受支持的追踪器，例如 [WandbTracer](https://python.langchain.com/docs/integrations/providers/wandb_tracing) 或 [OpikTracer](https://comet.com/docs/opik/tracing/integrations/ragas?utm_source=ragas&utm_medium=docs&utm_campaign=opik&utm_content=tracing_how_to)。

```python
# LangSmith
from langchain.callbacks.tracers import LangChainTracer

tracer = LangChainTracer(project_name="callback-experiments")
```

调用 `evaluate` 时，将追踪器传入 `callbacks` 参数。

```python
from ragas import EvaluationDataset
from datasets import load_dataset
from ragas.metrics import LLMContextRecall

dataset = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")

dataset = EvaluationDataset.load_from_hf(dataset["eval"])
evaluate(dataset, metrics=[LLMContextRecall()],callbacks=[tracer])
```

```text
{'context_precision': 1.0000}
```
<figure markdown="span">
  ![使用 LangSmith 追踪](../../../../_static/imgs/trace-langsmith.png)
  <figcaption>使用 LangSmith 追踪</figcaption>
</figure>


也可以基于 LangChain 的 `BaseCallbackHandler` 编写自定义回调，详情参见[相关说明](https://www.notion.so/Docs-logging-and-tracing-6f21cde9b3cb4d499526f48fd615585d?pvs=21)。
