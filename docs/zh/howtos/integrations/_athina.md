# Athina AI
## 在生产日志上使用 Ragas 指标 { #ragas-metrics-on-your-production-logs }

[Athina](https://athina.ai) 是生产监控与评估平台，可以体验[沙盒](https://demo.athina.ai/observe?filters=dateSpan%3D30)。

可以通过 [Athina 的 Ragas 指标集成](http://localhost:3001/evals/preset_evals/ragas_evals)评估生产日志，获取细粒度模型表现。

![Athina 性能指标](https://docs.athina.ai/performance-metrics.png)

例如，直观查看以下信息。
- 客户 `nike-usa` 的退款类查询，其 `AnswerRelevancy` 得分是多少？
- 使用 `catalog_answerer/v3` 提示词与 `gpt-3.5-turbo` 的产品目录查询，其 `Faithfulness` 得分是多少？

### ▷ 通过代码运行 Athina { #running-athina-programmatically }

程序化运行 Ragas 评估后，可以在 Athina 界面查看结果。👇

![在 Athina 中查看 Ragas 指标](https://docs.athina.ai/ragas-develop-view.png)

1. 安装 Athina Python SDK。

```
pip install athina
```

2. 在 [app.athina.ai](https://app.athina.ai) 注册账户，获取 API 密钥。

可以跟随[示例 notebook](https://github.com/athina-ai/athina-evals/blob/main/examples/ragas.ipynb)操作。

3. 运行代码。


```python
import os
from athina.evals import (
    RagasAnswerCorrectness,
    RagasAnswerRelevancy,
    RagasContextRelevancy,
    RagasFaithfulness,
)
from athina.loaders import RagasLoader
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.runner.run import EvalRunner
import pandas as pd

# Set your API keys
OpenAiApiKey.set_key(os.getenv("OPENAI_API_KEY"))
AthinaApiKey.set_key(os.getenv("ATHINA_API_KEY"))

# Load your dataset from a dictionary, json, or csv: https://docs.athina.ai/evals/loading_data
dataset = RagasLoader().load_json("raw_data.json")

# Configure the eval suite
eval_model = "gpt-3.5-turbo"
eval_suite = [
    RagasAnswerCorrectness(),
    RagasFaithfulness(),
    RagasContextRelevancy(),
    RagasAnswerRelevancy(),
]

# Run the evaluation suite
batch_eval_result = EvalRunner.run_suite(
    evals=eval_suite,
    data=dataset,
    max_parallel_evals=1,  # If you increase this, you may run into rate limits
)

pd.DataFrame(batch_eval_result)
```

### ▷ 配置自动评估生产日志 { #configure-ragas-to-run-automatically-on-your-production-logs }

如果已将[生产推理日志发送到 Athina](https://docs.athina.ai/logging/log_via_api)，可以配置自动运行 Ragas 指标。

1. 打开 [Athina 控制台](https://app.athina.ai/evals/config)。

2. 进入 Evals 页面，即左侧闪电图标。
3. 点击右上角 New Eval。
4. 选择 Ragas 标签页。
5. 选择需要配置的评估。

![在 Athina 界面配置 Ragas](https://docs.athina.ai/ragas-modal-bg.png)

#### 进一步了解 Athina { #learn-more-about-athina }
- [官网](https://athina.ai)
- [文档](https://docs.athina.ai)
- [GitHub 仓库](https://github.com/athina-ai/athina-evals)
- [沙盒](https://demo.athina.ai/observe?filters=dateSpan%3D30)
