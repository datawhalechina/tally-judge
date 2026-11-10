## 上下文实体召回率 { #context-entities-recall }

`ContextEntityRecall` 衡量参考答案中的实体有多大比例被检索上下文覆盖。具体计算 `reference` 与 `retrieved_contexts` 中共有实体的数量，占 `reference` 中实体总数的比例。它适用于旅游咨询、历史问答等依赖事实的场景。通过比较参考答案与检索上下文中的实体，可以衡量检索机制对关键实体的覆盖能力。

计算时使用两个集合。

- **$RE$**，参考答案中的实体集合
- **$RCE$**，检索上下文中的实体集合

计算两个集合共有实体的数量（$RCE \cap RE$），再除以参考答案中的实体总数（$RE$），公式如下。

$$
\text{Context Entity Recall} = \frac{\text{Number of common entities between $RCE$ and $RE$}}{\text{Total number of entities in $RE$}}
$$


### 示例 { #example }

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextEntityRecall

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextEntityRecall(llm=llm)

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."]
)
print(f"Context Entity Recall Score: {result.value}")
```

输出如下。
```
Context Entity Recall Score: 0.999999995
```

!!! note "同步用法"
    如果采用同步代码，可以使用 `.score()` 代替 `.ascore()`。

    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["The Eiffel Tower is located in Paris."]
    )
    ```

### 计算过程 { #how-its-calculated }



!!! example
    **参考答案**：泰姬陵位于印度阿格拉市亚穆纳河右岸，是一座象牙白色的大理石陵墓。莫卧儿皇帝沙贾汗于 1631 年下令修建，用于安葬他最宠爱的妻子穆姆塔兹·玛哈尔。
    **实体召回率高的上下文**：泰姬陵位于印度阿格拉，是爱情的象征与建筑奇迹。莫卧儿皇帝沙贾汗为纪念爱妻穆姆塔兹·玛哈尔而修建了它。这座建筑以精细的大理石工艺和周围美丽的花园闻名。
    **实体召回率低的上下文**：泰姬陵是印度的标志性建筑，被联合国教科文组织列为世界遗产，每年吸引数百万游客。精细的雕刻和壮观的建筑使它成为必游之地。

以上述参考答案与检索上下文为例。

- **第一步**，提取参考答案中的实体。
    - 参考答案中的实体（RE）为 ['Taj Mahal', 'Yamuna', 'Agra', '1631', 'Shah Jahan', 'Mumtaz Mahal']
- **第二步**，提取检索上下文中的实体。
    - 第一个上下文中的实体（RCE1）为 ['Taj Mahal', 'Agra', 'Shah Jahan', 'Mumtaz Mahal', 'India']
    - 第二个上下文中的实体（RCE2）为 ['Taj Mahal', 'UNESCO', 'India']
- **第三步**，使用上面的公式计算实体召回率。

    $$
    \text{context entity recall 1} = \frac{| RCE1 \cap RE |}{| RE |}
                                 = 4/6
                                 = 0.666
    $$

    $$
    \text{context entity recall 2} = \frac{| RCE2 \cap RE |}{| RE |}
                                 = 1/6
    $$

    第一个上下文对参考答案中的实体覆盖更充分，因此实体召回率更高。如果这两段上下文来自同一文档集上的不同检索机制，那么在重视实体的场景中，第一个检索机制表现更好。

## 旧版指标 API { #legacy-metrics-api }

以下示例使用旧版指标 API。新项目建议使用上文基于 collections 的 API。

!!! warning "弃用时间表"
    此 API 将在 0.4 版本中弃用，在 1.0 版本中移除。请迁移到上文基于 collections 的 API。

### 使用 SingleTurnSample 的示例 { #example-with-singleturnsample }

```python
from ragas import SingleTurnSample
from ragas.metrics import ContextEntityRecall

sample = SingleTurnSample(
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)

scorer = ContextEntityRecall(llm=evaluator_llm)

await scorer.single_turn_ascore(sample)
```
输出如下。
```
0.999999995
```
