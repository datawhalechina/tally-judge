# TallyJudge

<p align="center">
  <img src="assets/banner.png" alt="TallyJudge" width="860">
</p>

TallyJudge 是一个评估检索增强生成（RAG）系统的 Python 库。它可以衡量检索结果与参考段落的匹配程度、回答是否有依据、回答是否切题。你也可以给裁判模型一套自己的评分标准，得到逐条分数、标签和理由。

> [!NOTE]
> TallyJudge 源自 Ragas，由 Datawhale 社区独立维护，沿用 Apache License 2.0。[Ragas 项目](https://github.com/vibrantlabsai/ragas) · [许可证](LICENSE)

## 安装

在使用 Python 3.12 或更高版本的 uv 项目中，从 PyPI 添加 TallyJudge。

```bash
uv add tallyjudge
```

默认安装可以运行精确匹配、字符串包含、文档 ID 检索精度与召回，以及自定义函数指标。按文本距离匹配时，运行 `uv add "tallyjudge[text-distance]"` 安装 RapidFuzz。使用 OpenAI Python SDK 接入裁判时，运行 `uv add "tallyjudge[openai]"`。同时使用两者时，运行 `uv add "tallyjudge[openai,text-distance]"`。

## 用参考答案给一批回答评分

下面的例子比较图书馆问答系统的两条回答。`response` 是系统给出的回答，`reference` 是参考答案。两个文本指标都在本地计算，无需 API 密钥。

```python
from tallyjudge import Dataset, Evaluator, create_metric

dataset = Dataset.from_records([
    {"case_id": "闭馆日", "response": "周一", "reference": "周一"},
    {"case_id": "开馆时间", "response": "上午九点开放。", "reference": "九点"},
])
metrics = [
    create_metric("metric.text.exact_match"),
    create_metric("metric.text.string_presence"),
]
result = Evaluator(metrics).run(dataset)

for name, stats in result.summary.items():
    print(name, stats["mean"])
```

```text
metric.text.exact_match 0.5
metric.text.string_presence 1.0
```

两条回答都包含参考答案，只有第一条完全相同。`mean` 是有效得分的平均值，`coverage` 是取得有效得分的样本比例。用 `result.row_for("开馆时间").metrics` 可以查看第二条回答的逐项得分。

## 检查回答是否有检索依据

假设图书馆公告只写了工作日的开放时间，而系统回答“周末也开放”。忠实度指标会把回答拆成陈述，请裁判模型逐条判断资料是否支持。

下例使用支持 JSON 输出的 Chat Completions 服务。先运行 `uv add "tallyjudge[openai]"`，再设置环境变量 `OPENAI_API_KEY`（服务密钥）和 `JUDGE_MODEL`（裁判模型名）。使用其他兼容服务时，可通过 `OPENAI_BASE_URL` 指定地址。

```python
import asyncio
import os

from openai import AsyncOpenAI
from tallyjudge import create_language_model, create_metric


async def main():
    async with AsyncOpenAI() as client:
        judge = create_language_model(
            "model.openai_compatible.chat_completions",
            client=client,
            model=os.environ["JUDGE_MODEL"],
            temperature=0.0,
        )
        faithfulness = create_metric("metric.rag.faithfulness", llm=judge)
        score = await faithfulness.ascore_inputs({
            "user_input": "图书馆什么时候开放？",
            "response": "图书馆工作日上午九点开放，周末也开放。",
            "retrieved_contexts": ["图书馆公告：工作日上午九点开放。"],
        })
        print(score.value)


asyncio.run(main())
```

如果裁判把“工作日上午九点开放”和“周末也开放”识别为两条陈述，只判定第一条有依据，这条回答的忠实度就是 `0.5`。

`temperature` 需要显式指定。不传时插件不发送该参数，由服务端默认值决定（常见为 1.0），裁判的判断会更不稳定。客户端在哪个事件循环里创建，就只能在那个循环里使用：在一个 `asyncio.run(main())` 内创建客户端、连续 `await Evaluator(...).arun(dataset)`，不要用同一个模型反复调用同步的 `run()`。

## 从文档切片生成评测集

已有知识库切片但缺少参考答案时，可以让生成模型依据每个切片出一道题。每条成功样本包含问题、参考答案、证据原文，以及对应的切片和文档 ID。下面的函数接收前例构造的模型。

```python
import json
from pathlib import Path

from tallyjudge import Dataset, create_synthesizer
from tallyjudge.interfaces.models import LanguageModel


async def make_testset(judge: LanguageModel, path: Path) -> Dataset:
    chunks = [
        {"chunk_id": "hours", "document_id": "library", "text": "图书馆周一闭馆。"},
        {"chunk_id": "loans", "document_id": "library", "text": "图书借期为14天。"},
    ]
    generator = create_synthesizer("synthesis.testset.single_hop", llm=judge, seed=7)
    result = await generator.arun(chunks, testset_size=2)
    with path.open("w", encoding="utf-8") as stream:
        for row in result.to_records():
            print(json.dumps(row, ensure_ascii=False), file=stream)
    return result.to_dataset()
```

每个切片最多尝试一次。用 `result.generated_count` 和 `result.attempted_count` 查看成功数与尝试数，失败原因和来源保存在 `result.candidates`。在 `@experiment` 里，把 `row["user_input"]` 传给被测应用，返回实际的 `response`、`retrieved_contexts` 和 `retrieved_context_ids`。出题证据保存在 `reference_contexts` 和 `reference_context_ids`，可按 ID 回到原文复核。需要用裁判抽查参考答案时，给 `faithfulness` 传入列映射 `{"response": "reference", "retrieved_contexts": "reference_contexts"}`。

### 从知识库构建单轮 RAG 评测集

只有知识库文件时，运行 `uv add "tallyjudge[synthesis-graph]"` 安装建图依赖，并准备一个生成模型和一个嵌入模型。下面的函数读取本地 UTF-8 文本，生成参考问题、答案和证据 ID。评测检索器时，让它只看问题，用它实际召回的切片和生成的回答评分。

```python
from pathlib import Path

from tallyjudge import Dataset, create_synthesizer
from tallyjudge.interfaces.models import EmbeddingModel, LanguageModel
from tallyjudge.interfaces.synthesis import SourceDocument
from tallyjudge.plugins.synthesis.transforms import build_document_graph


async def knowledge_base_testset(
    directory: Path, judge: LanguageModel, embeddings: EmbeddingModel
) -> Dataset:
    documents = (
        SourceDocument(path.relative_to(directory).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(directory.rglob("*.txt"))
    )
    graph = await build_document_graph(documents, llm=judge, embeddings=embeddings)
    generator = create_synthesizer("synthesis.testset.rag", llm=judge, seed=7)
    result = await generator.arun(graph, testset_size=12)
    return result.to_dataset()
```

如果检索系统已经有切片，直接把含 `chunk_id`、`document_id` 和 `text` 的记录传给 `build_chunk_graph(chunks, llm=judge, embeddings=embeddings)`。这能保留检索器使用的 ID，也适合每份都很短的资料。原始文档按 token 长度选择转换流程；离线使用默认分词器时需预先缓存 `cl100k_base`，超长的抽取窗口还会用到 `o200k_base`。

生成器从摘要生成读者画像，按图中可用的三种策略出题。单跳具体题用一个节点的实体，多跳具体题用实体重叠，多跳抽象题用摘要相似的节点及其主题。每条记录的 `synthesizer_name` 标出题型。各策略的份额独立向上取整，三种策略都可用时，请求 4 题的配额为 2、2、2。用 `result.generated_count` 查看实际题数。比较两版检索器时，复用同一份已生成的数据集。[本地检索评测示例](tests/tallyjudge/integration/test_rag_testset.py)

### 仅按实体关系生成跨段题

需要按固定大小切分、只按实体关系出题时，可以用 `split_documents` 和 `build_entity_graph`。下例中，若模型从三份文件都抽出「馆藏指南」，并从前两份抽出 Kit，高频的「馆藏指南」会被排除，Kit 将前两份连起来。

```python
from tallyjudge import create_synthesizer
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.synthesis.documents import SourceDocument, split_documents
from tallyjudge.plugins.synthesis.graph import build_entity_graph


async def make_multihop_set(judge: LanguageModel):
    docs = [
        SourceDocument("design", "馆藏指南。Ada 设计了 Kit。"),
        SourceDocument("location", "馆藏指南。Kit 收藏在罗马。"),
        SourceDocument("rye", "馆藏指南。黑麦售价五枚硬币。"),
    ]
    graph = await build_entity_graph(split_documents(docs), llm=judge)
    generator = create_synthesizer("synthesis.testset.multi_hop", llm=judge)
    return await generator.arun(graph, testset_size=1)
```

已有逐块实体标签时，`build_graph(chunks)` 在本地连边。使用 `build_entity_graph` 时，单个未标注切片也会提取实体，已标注切片复用已有结果，用量见 `graph.generation_usage`。长切片分窗用到 tiktoken 的 `o200k_base`，离线处理长文时需预先缓存词表。也可以用 `build_similarity_graph` 按正文嵌入的余弦相似度连边。

### 用同一张图生成单段题和跨段题

把上例建好的 `graph` 保存一次，下次出题时从文件恢复原文、切片 ID 和证据关系。下面的例子请求 10 道题，证据充足时，5 道只需一个切片，另 5 道需要两个相关切片。

```python
import json
from pathlib import Path

from tallyjudge import create_synthesizer
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.synthesis.graph import EvidenceGraph


def save_evidence(graph: EvidenceGraph, path: Path) -> None:
    path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False), encoding="utf-8")


def load_evidence(path: Path) -> EvidenceGraph:
    return EvidenceGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))


async def make_mixed_set(graph: EvidenceGraph, judge: LanguageModel):
    generator = create_synthesizer(
        "synthesis.testset.mixed", llm=judge, seed=7, multi_hop_fraction=0.5
    )
    return await generator.arun(graph, testset_size=10)
```

结果中的 `synthesizer_id` 标出每题来自 `single_hop` 还是 `multi_hop`。跨段证据不足时，生成器会用剩余的单段候选补足配额；比例设为 `1.0` 就只出跨段题。两类候选用尽或出题失败时，实际题数可能少于请求数。用 `result.to_dataset()` 将成功样本交给实验，检索器只接收 `user_input`。随后可按题型对照实际召回 ID 与参考 ID。

迁移已有 Ragas 实验时，可以直接复用原题集。若保存了原图和出题场景，可用 `SingleHopScenario.from_ragas` 或 `MultiHopScenario.from_ragas` 恢复后交给 `generate_sample`。单跳和多跳使用归档出题模板，模型返回问题与答案，参考证据直接取自所选切片。仅给原始文件重新采样时，题目可能不同。[场景回放用例](https://github.com/datawhalechina/tally-judge/blob/main/tests/tallyjudge/migration/test_prompt_parity.py#L147)

## 可用指标

| 指标 | 所需字段 | 评分方式 |
|---|---|---|
| `metric.text.exact_match` | `response`、`reference` | 两段文本完全相同得 1 分，否则得 0 分。 |
| `metric.text.string_presence` | `response`、`reference` | 回答中包含参考文本得 1 分，否则得 0 分。 |
| `metric.rag.faithfulness` | `user_input`、`response`、`retrieved_contexts` | 用裁判模型计算有检索依据的陈述占比。检索资料按段落传入字符串列表。 |
| `metric.rag.answer_relevance` | `user_input`、`response` | 裁判模型根据回答反推问题，再用嵌入模型衡量这些问题与原问题的相似度。 |
| `metric.rag.context_precision_with_reference` | `user_input`、`reference`、`retrieved_contexts` | 裁判逐段判断检索结果对参考答案是否有用，按原排名计算平均精度。 |
| `metric.rag.context_precision_without_reference` | `user_input`、`response`、`retrieved_contexts` | 按系统回答判断每段是否有用。两种平均精度在所有段落都未命中时均无定义。 |
| `metric.rag.context_recall` | `user_input`、`retrieved_contexts`、`reference` | 裁判一次读取全部检索段落，得分是参考答案的陈述中能由这些段落支持的比例。 |
| `metric.rag.context_entity_recall` | `reference`、`retrieved_contexts` | 从参考答案与检索段落分别抽取实体，按原字符串去重，计算检索命中的参考实体比例。参考实体为空时无定义。 |
| `metric.rag.factual_correctness` | `response`、`reference` | `precision` 核对回答中的陈述，`recall` 核对参考答案中的陈述。默认的 `f1` 核对两个方向，`beta` 调整召回的权重。 |
| `metric.rag.noise_sensitivity` | `user_input`、`response`、`reference`、`retrieved_contexts` | 衡量全部回答陈述中，错误且被相关段落（`relevant`）或仅被无关段落（`irrelevant`）支持的比例，越低越好。 |
| `metric.rag.answer_accuracy` | `user_input`、`response`、`reference` | 第二次判分交换回答与参考的呈现顺序，按 0、2、4 三档判分，返回归一化均值。 |
| `metric.rag.context_relevance` | `user_input`、`retrieved_contexts` | 两份提示词分别按 0、1、2 三档评估检索段落对问题的相关性，返回归一化均值。 |
| `metric.rag.response_groundedness` | `response`、`retrieved_contexts` | 两份提示词分别评估整段回答是否有检索依据，返回归一化均值。 |
| `metric.text.semantic_similarity` | `reference`、`response` | 嵌入余弦相似度，范围 −1 至 1。配置 `threshold` 后，达到阈值得 1 分，低于阈值得 0 分。 |
| `metric.rag.answer_correctness` | `user_input`、`response`、`reference` | 按事实陈述分类和嵌入相似度加权评分，默认权重为 `[0.75, 0.25]`。设置 `weights=[1.0, 0.0]` 只需裁判模型。 |
| `metric.text.string_similarity` | `response`、`reference` | 归一化字符串相似度，默认使用 Levenshtein，范围 0 至 1。 |
| `metric.retrieval.id_context_precision` | `retrieved_context_ids`、`reference_context_ids` | 命中的不同文档数占检索文档数的比例，检索 ID 为空时无定义。无需模型。 |
| `metric.retrieval.id_context_recall` | 同上 | 命中的不同文档数占参考文档数的比例，参考 ID 为空时无定义。整数和同值字符串 ID 视作同一份文档。 |
| `metric.retrieval.text_context_precision` | `retrieved_contexts`、`reference_contexts` | 按返回顺序匹配段落，在命中位置计算平均精度。 |
| `metric.retrieval.text_context_recall` | 同上 | 参考段落被命中的比例。参考列表为空时，结果标记为未定义。 |
| `metric.custom.numeric` | `response` | 按 `criteria` 输出指定范围的数值，或从给定整数等级中投票。 |
| `metric.custom.categorical` | `response` | 按 `criteria` 从 `labels` 中选择标签，汇总每个标签的次数。 |
| `metric.custom.rubric_without_reference` | 通常传 `response`，可带问题和上下文 | 使用默认无参考量表或自定义规则，保留等级分和反馈。 |
| `metric.custom.rubric_with_reference` | 通常传 `response`、`reference` | 使用默认有参考量表或自定义规则，保留等级分和反馈。 |
| `metric.workflow.tool_sequence` | `trace`、`expected_tools` | 按开始顺序比较实际成功的工具调用，保留失败及重试记录。 |
| `metric.custom.instance_rubric` | `rubrics`，可带 `response`、`reference` 等 | 每条样本提供自己的评分说明。 |

答案准确度、上下文相关度和回答依据使用两个裁判提示。每路默认最多尝试 5 次取得有效等级，可在 `create_metric` 时设置 `max_retries=1`。只有一路取得有效等级时按该路给分，两路都耗尽等级尝试时记为 `undefined`。模型拒答、传输或结构解析失败会返回错误状态。

### 本地评检索结果

把检索器实际返回的段落按原顺序放入 `retrieved_contexts`，把标准证据放入 `reference_contexts`。这个例子找到了两段标准证据中的一段。

```python
from tallyjudge import Dataset, Evaluator, create_metric

recall = create_metric("metric.retrieval.text_context_recall", threshold=1.0)
data = Dataset.from_records([{
    "retrieved_contexts": ["周一闭馆。"],
    "reference_contexts": ["周一闭馆。", "法定节假日另行通知。"],
}])
print(Evaluator([recall]).run(data).rows[0].metrics[recall.spec.id].value)  # 0.5
```

### 按业务标准判分

在前面示例的 `async with AsyncOpenAI() as client` 内，复用已创建的 `judge`。三票使用相同标准独立判分，得票最多的标签胜出。得票数相同时取最早出现的标签。

```python
grade = create_metric(
    "metric.custom.categorical",
    llm=judge,
    criteria="回答同时说明退货时限和凭证要求才算完整。",
    labels=["完整", "部分", "缺失"],
    strictness=3,
)
score = await grade.ascore_inputs({"response": "签收后七天内可申请退货。"})
print(score.value, score.reason)
```

数值裁判用 `levels=[0, 1]` 可以做二元投票，连续数值用 `value_range=(0, 10)` 单次评分。固定量表默认提供五档说明，也可以传入自己的规则。逐行量表从每行的 `rubrics` 读取非空字符串字典，裁判按字典的插入顺序看到规则。

## 用自己的函数评分

带类型注解的函数可以直接变成指标。插件 ID 和评分版本 `revision` 由你声明，修改评分逻辑时提升 `revision`，以免和旧结果混在一起比较。

```python
from tallyjudge import Dataset, Evaluator, numeric_metric


@numeric_metric("metric.library.length_ratio", revision="1", value_range=(0.0, None))
def length_ratio(response: str, reference: str) -> float:
    return len(response) / max(len(reference), 1)


result = Evaluator([length_ratio()]).run(
    Dataset.from_records([{"case_id": "闭馆日", "response": "周一", "reference": "周一"}])
)
print(result.summary["metric.library.length_ratio"]["mean"])  # 1.0
```

`categorical_metric(..., labels=[...])` 用于固定标签的分类结果，`ranking_metric(..., length=n)` 用于固定长度的排序结果。

## 评测自己的应用

`@experiment` 对每行调用一次你的函数。调用 RAG 系统时传入 `row["user_input"]`，返回实际回答和召回结果。评分时，问题、参考答案和参考证据取自题集，业务函数回传的同名字段只保存在原始输出里。自定义列映射也遵循这个规则。函数返回 `None` 时这一行记为跳过，抛错、超时和取消都保留在结果中。

```python
from tallyjudge import Dataset, create_metric, experiment


@experiment
def answer(row):
    return {"response": "周一"}  # 替换为对你的系统的调用


result = answer.run(
    Dataset.from_records([{"case_id": "闭馆日", "user_input": "哪天闭馆？", "reference": "周一"}]),
    metrics=[create_metric("metric.text.exact_match")],
)
print(result.row_for("闭馆日").task.output, result.summary["metric.text.exact_match"]["mean"])
```

需要等待网络或磁盘的函数请写成 `async def`，并用 `await answer.arun(...)` 运行。

## 保存、读取与对照

结果可以保存到本地目录，在另一个进程中按运行 ID 读回，再按样本 ID 逐行对照两次运行。读取不需要重新创建模型。

```python
from tallyjudge import compare_runs, create_storage_backend

store = create_storage_backend("storage.local.jsonl", root="runs")
result.save(backend=store)  # include_inputs=False 时不保存原始样本
restored = store.load_run(result.run_id)

for case in compare_runs(restored, result).rows:
    print(case.case_id, {key: score.delta for key, score in case.metrics.items()})
```

`compare_runs` 只在输入和评分口径一致、两边都评分成功时给出差值，其余情况在 `changes` 中说明原因。给 `Evaluator` 传入 `backend=store` 可在运行结束时自动保存；`cache=create_cache_backend("cache.local.sqlite", path=..., namespace=...)` 可复用已经算过的分数。

## 比较与优化业务提示词

给每次实验标记提示词版本和模型，然后对相同 `case_id` 的回答做配对比较。返回的行包含双方业务输出与原始评分，汇总中可见各自的覆盖率、成对样本数、胜平负与分差。

```python
from tallyjudge import Dataset, compare_experiments, create_metric, experiment

data = Dataset.from_records([
    {"case_id": "loans", "user_input": "借期多久？", "reference": "14天"},
])
metric = create_metric("metric.text.exact_match")
v1 = experiment(lambda row: {"response": "7天"}, metadata={"prompt_version": "v1"})
v2 = experiment(lambda row: {"response": "14天"}, metadata={"prompt_version": "v2"})
paired = compare_experiments(v1.run(data, metrics=[metric]), v2.run(data, metrics=[metric]))
print(paired.metrics[metric.spec.id].wins)  # 1
```

有两份带标注且 `case_id` 不重叠的数据时，可用 `create_optimizer("optimization.prompt.genetic", llm=judge, max_candidates=6, max_requests=100)` 创建优化器。调用 `optimizer.run(instruction, train=train, holdout=holdout, metric=metric)`，从 `result.instruction` 取入选指令，从 `result.holdout` 读取留出集报告。使用 MIPROv2 时，运行 `uv add "tallyjudge[dspy]"` 并选择 `optimization.prompt.dspy_mipro`。

| 可选接入 | 用法 |
| --- | --- |
| LangChain、LlamaIndex 文档 | `plugins.adapters.documents` 中的 `langchain_documents`、`llamaindex_documents` 保留来源，可接 `split_documents`。 |
| 工作流 | `plugins.adapters.workflow.langchain_trace` 接收 v2 事件。传 `include_content=True` 时保留工具参数、结果原文。 |
| 精确切片长度 | 把 `tiktoken_counter(encoding)` 或 `huggingface_counter(tokenizer_file)` 作为 `split_documents(..., measure=...)` 的长度函数。相应安装组为 `tiktoken`、`hf-tokenizers`。 |
| Git 与外部追踪 | `plugins.versioning.git.git_revision(repo, revision)` 返回可写入实验元数据的 commit。运行 `uv add "tallyjudge[mlflow]"` 后，可把 `create_event_sink("event.mlflow.tracking", tracking_uri=uri, experiment_id=exp_id)` 传给 `Evaluator`。 |

## 命令行

安装后可以用命令行运行本地指标并查看保存的结果。输入文件每行是一个 JSON 样本。

```bash
uv run tallyjudge metrics list
uv run tallyjudge run --input samples.jsonl --metric metric.text.exact_match --store runs
uv run tallyjudge runs show RUN_ID --store runs
```

运行完成时退出码为 0（`record` 模式下个别样本失败仍为 0），运行被停止或读写失败时为 1，参数错误为 2。需要注入模型客户端的指标请使用 Python 接口。

## 参与贡献

欢迎通过 [Issues](https://github.com/datawhalechina/tally-judge/issues) 提交问题和建议，通过 Pull Request 提交改进。开发环境和提交要求见[贡献指南](CONTRIBUTING.md)。原 Ragas 实现保存在 [`legacy-ragas`](https://github.com/datawhalechina/tally-judge/tree/legacy-ragas) 分支，来源与版权说明见 [NOTICE](NOTICE)。
