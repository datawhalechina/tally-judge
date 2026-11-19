# TallyJudge

<p align="center">
  <img src="./banner.png" alt="TallyJudge" width="860">
</p>

<p align="center">面向 RAG 与 AI 应用的评测框架，由 Datawhale 社区维护。</p>

> [!NOTE]
> TallyJudge 基于 [Ragas](https://github.com/vibrantlabsai/ragas) 的开源代码继续维护，遵循 [Apache License 2.0](LICENSE)，感谢原作者和社区贡献者。
>
> 根目录 `tallyjudge/` 提供本地文本评测和首批单轮 RAG 指标。现可评估回答中的陈述是否有检索依据，以及回答与原问题是否相关。原实现保留在 `src/ragas/` 供迁移对照。

[迁移设计](plan/plugin-architecture.zh-CN.md) · [迭代路线](plan/roadmap.zh-CN.md) · [第一阶段设计](plan/phase-1.zh-CN.md) · [RAG 评测设计](plan/phase-2.zh-CN.md) · [问题反馈](https://github.com/datawhalechina/tally-judge/issues)

## 我们要做什么

TallyJudge 希望让评测组件可以独立维护、按需组合，并让评测结果能够追溯到样本、指标和运行配置。

- **先做好 RAG 评测**：整理已有指标与数据处理能力，验证评分行为，逐步迁入新的实现。
- **构建包内插件架构**：模型、嵌入、指标、存储和框架接入通过统一接口组合，插件代码随主包维护与发布。
- **让运行结果完整可查**：明确样本身份、并发、失败、取消和用量记录，避免失败样本消失后产生误导性的平均分。
- **逐步扩展评测范围**：在 RAG 基础上完善 Prompt、AI 工作流和 Agent 的评测能力。

可视化工具已选定 Streamlit，将在独立仓库维护，开发暂不排期。界面将直接调用 TallyJudge 的 Python 接口，供业务用户选择评测组件、调整提示词和比较结果。

## 当前进度

| 状态 | 内容 |
|---|---|
| 已实现并通过本地验证 | Python 3.12 环境；标准样本与结果；字段绑定；插件注册；两个本地指标；有限并发、超时和取消；仓库外 wheel 安装 |
| 已在原实现整理 | 部分上游独立修复，包括评分边界、数据转换、异步嵌入和测试集生成问题 |
| 已实现并通过本地验证 | 异步生成和嵌入插件、Faithfulness 与 AnswerRelevance、逐行和按绑定的请求用量 |
| 下一阶段 | 上下文精度与召回、文档 ID 精度与召回，随后迁移参考答案相似度与正确性 |
| 已实现并通过本地验证 | 逐行实验、函数指标、结果存取与对照、可选请求事件、SQLite 评分缓存和本地 CLI |
| 后续迭代 | 其他 RAG 指标、框架接入及扩展评测能力 |
| 暂不排期 | 独立可视化界面 |

本地文本评测可以直接运行。RAG 评测由调用者提供模型客户端。Windows / Python 3.12 已验收，真实 SDK 在本地模拟 HTTP 上通过测试。跨平台 CI 和真实服务端点的结果待验证。

## 本地开发

需要 Python **3.12 或以上**和 uv。以下命令在仓库根目录执行：

```bash
git clone https://github.com/datawhalechina/tally-judge.git
cd tally-judge
uv sync --python 3.12
uv run python -c "import tallyjudge; print(tallyjudge.__file__)"
uv run pytest
```

`.python-version` 固定本地默认版本为 3.12。`uv sync` 安装新包及开发工具，当前默认运行依赖仅为 Pydantic。OpenAI SDK 与 HTTP 测试依赖由 `pyproject.toml` 的 `integration` 组管理。

项目当前以源码开发为主；正式分发名称和版本发布安排另行确认。这里不提供未经发布验证的 `pip install tallyjudge` 命令。

开发检查：

```bash
uv run ruff check tallyjudge tests/tallyjudge
uv run ruff format --check tallyjudge tests/tallyjudge
uv run pyright
uv run pytest
```

旧测试需要对应的原实现环境。默认测试只收集 `tests/tallyjudge/`，避免旧测试依赖进入新环境。

## 本地文本评测

安装开发环境后，可以直接运行：

```python
from tallyjudge import Dataset, Evaluator, MetricBinding, create_metric

dataset = Dataset.from_records([
    {"case_id": "a", "answer": "Paris", "reference": "Paris"},
    {"case_id": "b", "answer": "The capital is Paris.", "reference": "Paris"},
])
evaluator = Evaluator(metrics=[
    MetricBinding(
        create_metric("metric.text.exact_match"),
        key="exact",
        input_map={"response": "answer"},
    ),
    MetricBinding(
        create_metric("metric.text.string_presence"),
        key="contains",
        input_map={"response": "answer"},
    ),
])
result = evaluator.run(dataset)
print(result.summary["exact"]["mean"])     # 0.5
print(result.summary["contains"]["mean"])  # 1.0
print(result.to_json(include_inputs=False))
```

在 Notebook 或已有事件循环中使用 `result = await evaluator.arun(dataset)`。`ExactMatchMetric` 比较文本是否完全相等，`StringPresenceMetric` 判断参考文本是否出现在回答中；两者保留大小写和空白，空参考文本的包含分数为 1.0。

结果按输入顺序保留每行及各指标绑定的状态。缺字段、输入类型错误、超时和无效分数不会被填成零分；汇总同时提供有效数和覆盖率。`include_inputs=False` 不导出原始行，保留样本 ID、内容摘要和配置快照；自定义指标仍需自行避免在 reason 或配置快照中写入敏感信息。

超时和取消的清理宽限可通过 `ExecutionConfig(timeout=30.0, cleanup_timeout=5.0)` 设置。宽限期后仍未退出的任务会记录在 `result.cleanup_warnings`，并停止本次运行的后续调度。通过 `result.run_status` 和 `result.stop_reason` 可以直接识别运行是否中途停止；尚未开始的任务记为 `not_started`。校验错误提供 `field` 和 `expected`，默认只记录一行 WARNING；其 traceback 仅在 DEBUG 输出。指标运行异常保留异常类名及 ERROR traceback。空数据集的覆盖率和均值均为 null。

异步入口停止等待后，吞掉取消的协程可能仍在调用者的循环内运行；外层 `asyncio.run()` 的关闭行为不由 TallyJudge 控制。同步入口使用有界的循环清理。需要强制终止不可信或阻塞的插件时，仍需进程级隔离；具体约定见[阶段 1 执行说明](plan/phase-1.zh-CN.md#执行约定)。

## 保存和比较两次评测

在上面的文本评测结束后，可以按运行 ID 找回每条样本的分数。后续调整指标或回答后，继续使用相同的样本 ID 和指标绑定名，就能逐行查看变化。

```python
from tallyjudge import compare_runs, create_storage_backend

store = create_storage_backend("storage.local.jsonl", root="runs")
result.save(backend=store)
previous = store.load_run(result.run_id)
print(previous.row_for("a").metrics["exact"].value)

current = evaluator.run(dataset)
print(compare_runs(previous, current).rows[0].metrics["exact"].delta)
```

已有 JSONL 样本也可以从命令行评分。第一条命令列出指标和所需的模型依赖，第二条命令输出本次运行 ID，第三条命令用它读回结果。

```bash
uv run tallyjudge metrics list
uv run tallyjudge run --input samples.jsonl --metric metric.text.exact_match --store runs
uv run tallyjudge runs show RUN_ID --store runs
```

要先生成回答再评分，可用 `@experiment` 定义逐行函数。业务输出与原始输入分别保存。数值、类别与排序函数指标均可参与同一轮评测。请求事件与磁盘评分缓存由调用者显式启用。[实验、事件与缓存的使用约定](plan/phase-3.zh-CN.md#使用约定)

## 单轮 RAG 评测

同一份样本可以同时评估回答的依据和相关性。两项指标共用裁判模型，只有回答相关性需要嵌入。下面的脚本使用 OpenAI 异步客户端。在本地开发环境运行 `uv sync --group integration` 可安装本项目验证过的 SDK 版本。设置 `OPENAI_API_KEY`，并将示例中的 `JUDGE_MODEL` 和 `EMBEDDING_MODEL` 环境变量指向实际部署名。

```python
import asyncio
import os

from openai import AsyncOpenAI

from tallyjudge import (
    Dataset, Evaluator, create_embedding_model, create_language_model, create_metric,
)


async def main():
    dataset = Dataset.from_records([{
        "case_id": "fr-1",
        "user_input": "法国的首都是哪里？",
        "response": "巴黎。",
        "retrieved_contexts": ["法国的首都是巴黎。"],
    }])
    async with AsyncOpenAI() as client:
        judge = create_language_model(
            "model.openai_compatible.chat_completions",
            client=client, model=os.environ["JUDGE_MODEL"], temperature=0.01,
        )
        embeddings = create_embedding_model(
            "embedding.openai_compatible.embeddings",
            client=client, model=os.environ["EMBEDDING_MODEL"],
        )
        metrics = [
            create_metric("metric.rag.faithfulness", llm=judge),
            create_metric("metric.rag.answer_relevance", llm=judge, embeddings=embeddings),
        ]
        result = await Evaluator(metrics).arun(dataset)
    for metric_id, score in result.rows[0].metrics.items():
        print(metric_id, score.status, score.value, score.error)


if __name__ == "__main__":
    asyncio.run(main())
```

`faithfulness` 先拆分陈述，再计算收到的判定中有依据的比例。`answer_relevance` 对应旧版 `AnswerRelevancy`，默认从回答反推三个问题，用带符号的余弦相似度评分，负分有效。没有陈述、没有判定或遇到零向量时，行结果记录 `undefined` 和原因码。所有反推问题都为空时，相关性保留旧版的有效分数 `0.0`。

每个绑定的请求数和已知 token 见 `result.summary[metric_id]["usage"]`。服务端遗漏的 token 字段在结果中保留为空。单次 HTTP 失败直接记为 `model_error`，只有 JSON 或输出字段校验失败会按 `structured_retries` 重问，默认最多一次。[模型配置、用量字段与迁移差异](plan/phase-2.zh-CN.md)

## 架构方向

```text
tallyjudge/
├── api/          # Python 公共入口
├── core/         # 样本、执行、结果与运行事件
├── interfaces/   # 指标、模型、存储等插件契约
├── registry/     # 插件描述、注册与按需加载
└── plugins/      # 指标、模型、嵌入、接入、存储等实现
```

内核负责样本、调度、结果及用量。模型和嵌入插件持有客户端，RAG 指标只调用模型协议。存储、请求事件和评分缓存是包内插件。框架接入沿用相同的扩展路线。

“包内插件”指插件源码随同一个包发布，不要求维护一组独立插件包；第三方 SDK 仍可按所选能力安装。接口版本、指标评分版本和插件 ID 分开管理。

## 迭代顺序

1. **本地评测闭环（已实现）**：标准样本、完整结果、两个无模型指标和批量运行；跨平台验收由 CI 补齐。
2. **单轮 RAG 评测**。首版回答质量已实现。接下来按检索质量、参考答案正确性的顺序补齐。
3. **实验与持久化（已实现）**。实验、失败记录和模型用量随运行结果保存，可按样本查询和比较。
4. **能力补齐与扩展**：迁移其他诊断指标、框架接入、合成与优化能力，再完善 Prompt、工作流和 Agent 评测。

每一阶段先验证输入、分数和失败行为，再扩大范围。不以目录搬迁或接口数量作为完成标准。具体交付物与验收条件见[迭代路线](plan/roadmap.zh-CN.md)。

## 仓库与文档

- `tallyjudge/`：新实现。
- `src/ragas/`：原实现与迁移对照，暂时原位保留。
- `tests/tallyjudge/`：新包测试；其余旧测试保留作参考。
- `plan/`：迁移设计、阶段计划和验收标准。
- `docs/`：项目文档，其中仍有基于原实现的内容，尚不能全部用于新包。
- `examples/`：尚未迁移的旧示例，不属于新包默认工作区。

文档站使用 MkDocs，依赖仍统一维护在根 `pyproject.toml`。预览需要 Node.js/npm：

```bash
uv venv --python 3.12 .venv-docs
uv pip install --python .venv-docs --group docs
npm run docs:dev
```

## 参与贡献

新增指标、修改接口或评审实现时，请参阅总纲中的模块职责、函数拆分和注释示例。开发环境与检查命令见贡献者指引。[架构与编码总纲](plan/implementation-standards.zh-CN.md) · [贡献者指引](AGENTS.md)

欢迎通过 [Issues](https://github.com/datawhalechina/tally-judge/issues) 提交可复现问题、迁移遗漏和设计建议，通过 [Pull Requests](https://github.com/datawhalechina/tally-judge/pulls) 提交改进。

迁移贡献请注明原实现位置、目标名称、行为差异和测试结果。优先处理有明确边界的修复，不把评分口径变化混入重命名或文件迁移。新实现遵循[全局命名规范](plan/plugin-architecture.zh-CN.md)。

Datawhale 是一个关注数据科学与人工智能的开源学习社区。我们希望通过共同学习、协作开发和持续维护，让项目逐步成为可理解、可复现、可扩展的评测工具。

## 来源与许可证

TallyJudge 在 Ragas 开源代码基础上独立维护。新项目名称用于区分后续品牌与发布；PyPI 名称可用性将在正式发布前核验。本项目继续遵循 [Apache License 2.0](LICENSE)，保留源代码中的原有版权和贡献记录。
