# 噪声敏感度 { #noise-sensitivity }

`NoiseSensitivity` 衡量系统利用相关或无关的检索文档时，生成错误回答的频率。得分为 0～1，越低越好。计算需要 `user_input`、`reference`、`response` 和 `retrieved_contexts`。

计算时，逐一检查生成回答中的陈述，依据标准答案判断其是否正确，再判断它是否有相关或无关的检索上下文作为依据。理想情况下，回答中的全部陈述都应得到相关检索上下文的支持。


$$
\text{noise sensitivity (relevant)} = {|\text{Total number of incorrect claims in response}| \over |\text{Total number of claims in the response}|}
$$


### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import NoiseSensitivity

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = NoiseSensitivity(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
    ]
)
print(f"Noise Sensitivity Score: {result.value}")
```

输出如下。

```
Noise Sensitivity Score: 0.3333333333333333
```

要计算无关上下文的噪声敏感度，可将 `mode` 参数设为 `irrelevant`。

```python
scorer = NoiseSensitivity(llm=llm, mode="irrelevant")
result = await scorer.ascore(
    user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
    ]
)
print(f"Noise Sensitivity (Irrelevant) Score: {result.value}")
```

输出如下。

```
Noise Sensitivity (Irrelevant) Score: 0.0
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        user_input="What is the Life Insurance Corporation of India (LIC) known for?",
        response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India...",
        reference="The Life Insurance Corporation of India (LIC) is the largest insurance company...",
        retrieved_contexts=[...]
    )
    ```

## 计算过程 { #how-its-calculated }

!!! example
    问题：印度人寿保险公司（LIC）以什么闻名？

    标准答案：印度人寿保险公司（LIC）是印度最大的保险公司，于 1956 年保险业国有化时成立，以管理庞大的投资组合闻名。

    相关检索结果如下。
        - 印度人寿保险公司（LIC）成立于 1956 年，当时印度保险业实行国有化。
        - LIC 是印度最大的保险公司，拥有庞大的投保人网络，在金融业中发挥重要作用。
        - 作为印度最大的机构投资者，LIC 管理着巨额人寿保险基金，为国家金融稳定作出贡献。

    无关检索结果如下。
        - 得益于金融、科技和制造业等行业的发展，印度经济是全球增长最快的主要经济体之一。

下面看看如何计算相关上下文的噪声敏感度。

- **第一步**，找出能够支持标准答案的相关上下文。

    - 标准答案如下。
    印度人寿保险公司（LIC）是印度最大的保险公司，于 1956 年保险业国有化时成立，以管理庞大的投资组合闻名。

    - 上下文如下。
        - 上下文 1：印度人寿保险公司（LIC）成立于 1956 年，当时印度保险业实行国有化。
        - 上下文 2：LIC 是印度最大的保险公司，拥有庞大的投保人网络，在金融业中发挥重要作用。
        - 上下文 3：作为印度最大的机构投资者，LIC 管理着巨额资金，为国家金融稳定作出贡献。

- **第二步**，检查生成答案中的陈述能否从相关上下文中推断出来。

    - 答案如下。
    印度人寿保险公司（LIC）是印度最大的保险公司，以庞大的投资组合闻名。LIC 为国家金融稳定作出贡献。

    - 上下文如下。
        - 上下文 1：印度人寿保险公司（LIC）成立于 1956 年，当时印度保险业实行国有化。
        - 上下文 2：LIC 是印度最大的保险公司，拥有庞大的投保人网络，在金融业中发挥重要作用。
        - 上下文 3：作为印度最大的机构投资者，LIC 管理着巨额资金，为国家金融稳定作出贡献。


- **第三步**，找出答案中的错误陈述，即标准答案未支持的陈述。

    - 标准答案如下。
    印度人寿保险公司（LIC）是印度最大的保险公司，于 1956 年保险业国有化时成立，以管理庞大的投资组合闻名。

    - 答案如下。
    印度人寿保险公司（LIC）是印度最大的保险公司，以庞大的投资组合闻名。LIC 为国家金融稳定作出贡献。

    解释：标准答案未提及 LIC 对国家金融稳定的贡献，因此按本例的评估依据，这条陈述被判为错误。

    错误陈述数为 1
    陈述总数为 3

- **第四步**，使用以下公式计算噪声敏感度。

    $$
    \text{noise sensitivity} = { \text{1} \over \text{3} } = 0.333
    $$

最终噪声敏感度为 0.333，表示答案中的三条陈述有一条错误。


## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NoiseSensitivity

sample = SingleTurnSample(
    user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
    ]
)

scorer = NoiseSensitivity(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

输出如下。

```
0.3333333333333333
```

要计算无关上下文的噪声敏感度，可将 `mode` 参数设为 `irrelevant`。

```python
scorer = NoiseSensitivity(mode="irrelevant")
await scorer.single_turn_ascore(sample)
```

来源：噪声敏感度指标由 [RAGChecker](https://github.com/amazon-science/RAGChecker/tree/main/ragchecker) 提出。
