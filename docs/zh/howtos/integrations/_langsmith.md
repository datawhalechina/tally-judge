# LangSmith { #langsmith }
## 数据集与追踪可视化 { #dataset-and-tracing-visualisation }

[LangSmith](https://docs.smith.langchain.com/) 是 LangChain 团队提供的 LLM 应用平台，支持追踪、调试与评估。

LangSmith 与 Ragas 集成提供以下功能。
1. 查看 Ragas `evaluator` 的追踪
2. 在 LangChain 评估中使用 Ragas 指标（即将支持）


## 追踪 Ragas 指标 { #tracing-ragas-metrics }

本例 Ragas 底层使用 LangChain，因此只需配置 LangSmith 即可记录追踪。

设置以下环境变量，详情参阅 [LangSmith 文档](https://docs.smith.langchain.com/#quick-start)。

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

配置后，按通常方式运行评估。


```python
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

fiqa_eval = load_dataset("vibrantlabsai/fiqa", "ragas_eval")

result = evaluate(
    fiqa_eval["baseline"].select(range(3)),
    metrics=[context_precision, faithfulness, answer_relevancy],
)

result
```

    Found cached dataset fiqa (/home/jjmachan/.cache/huggingface/datasets/vibrantlabs___fiqa/ragas_eval/1.0.0/3dc7b639f5b4b16509a3299a2ceb78bf5fe98ee6b5fee25e7d5e4d290c88efb8)



      0%|          | 0/1 [00:00<?, ?it/s]


    evaluating with [context_precision]


    100%|█████████████████████████████████████████████████████████████| 1/1 [00:23<00:00, 23.21s/it]


    evaluating with [faithfulness]


    100%|█████████████████████████████████████████████████████████████| 1/1 [00:36<00:00, 36.94s/it]


    evaluating with [answer_relevancy]


    100%|█████████████████████████████████████████████████████████████| 1/1 [00:10<00:00, 10.58s/it]





    {'context_precision': 0.5976, 'faithfulness': 0.8889, 'answer_relevancy': 0.9300}



然后进入项目即可查看追踪。
