# 阶段 4 · 指标能力扩展

状态为设计稿。阶段 3 的本地提交 `cb8ad6d` 提供了实验、运行存储、逐行对照和评分缓存。当前注册表中的四个评分插件是精确匹配、字符串包含、回答忠实度和回答相关性。阶段 4 的第一个使用场景是给只有参考段落、没有文档 ID 的 RAG 数据评检索质量，第二个是让用户用自己的判分标准评回答。[当前注册表](../tallyjudge/registry/builtins.py) · [阶段 3 复核](phase-3.zh-CN.md#提交复核) · [总路线](roadmap.zh-CN.md)

## 交付顺序

产品的下一版仍优先交付阶段 2b 的检索质量，先做文档 ID 精度、召回，再做两种模型上下文精度和模型上下文召回。随后补语义相似度和答案正确性，与已有的两项回答质量指标组成首个九指标 RAG 闭环。文本距离检索和自定义判分可以各自验收，后续专用指标按用户场景分批排期。[单轮 RAG 的三个功能版本](roadmap.zh-CN.md#单轮-rag-的三个功能版本)

| 顺序 | 使用场景及交付 | 依赖与完成条件 |
|---|---|---|
| 4A · 文本距离检索 | 先交 `StringSimilarityMetric`，再交 `TextContextPrecisionMetric`、`TextContextRecallMetric` | 只需文本和可选 RapidFuzz。有序检索列表、重复段落和空集合的预期见下节。 |
| 4B · 自定义裁判 | 先交单轮数值与类别判分、独立投票，再交固定和逐行 rubric。模型排序插件另验有序输出。 | 复用阶段 2 的文本模型协议和阶段 3 的三种输出。假模型可复验票数、分数、缓存及用量。 |
| 4C · 文本、SQL 与表格 | 分别交 BLEU、ROUGE、chrF，引用片段对齐，摘要，SQL 语义等价和表格比较 | 按插件安装依赖。固定文本可复验统计得分。SQL 用模型比较两段 SQL 及表结构。表格用列、行均不全的输入验收。 |
| 4D · 多轮与 Agent | 随第一个工具调用指标交付中立消息格式，再迁移工具 F1、有参考／无参考的目标达成和话题遵循 | 用手写 JSON 对话即可评分。从消息重建调用顺序、参数和重复调用，与上游逐例核对部分得分。 |
| 4E · 专用模型 | 按需迁移 HHEM 忠实度、多模态忠实度及相关性 | HHEM 需要陈述生成模型和本地分类器。多模态需要可接收图文输入的模型适配。分别有可复验的模型、权重版本和测试样本后排期。 |

框架接入、合成测试集、Prompt 优化与工作流过程评测沿用阶段 5。Streamlit 界面在独立仓库直接调用 Python 接口。首个 RAG 版本的验收条件见阶段 2b，表中 4C 至 4E 可以独立交付。[指标能力与目标符号](plugin-architecture.zh-CN.md#指标改名清单)

## 4A · 文本相似度与检索

| 目标插件 | 输入与计分 |
|---|---|
| `metric.text.string_similarity` | `reference: str` 与 `response: str`。计算 `1 - normalized_distance`，默认 Levenshtein，另支持 Hamming、Jaro、Jaro–Winkler。将 `DistanceMeasure` 的值写入绑定快照。 |
| `metric.retrieval.text_context_precision` | `retrieved_contexts: list[str]`、`reference_contexts: list[str]`。按检索原序，为每个段落取与所有参考段落的最高相似度。达到阈值算命中，对命中位置的前缀精度取平均，零次命中为 `0.0`。 |
| `metric.retrieval.text_context_recall` | 同样两列。从每个参考段落找最相似的检索段落，返回命中参考段落数除以参考段落数。参考列表为空时记 `undefined`，原因码拟为 `no_reference_contexts`。 |

默认阈值为 `0.5`，相似度等于阈值时命中。检索保留次序和重复项。首版选择已有的四种距离算法，两个检索插件复用同一段文本相似度计算。公开调用沿用 `create_metric`、`MetricBinding` 和 `Evaluator`。实现放在 `plugins/metrics/text/` 与 `plugins/metrics/retrieval/`，将距离算法与阈值记入配置快照。[上游字符串算法](../src/ragas/metrics/collections/_string.py) · [上游精度](../src/ragas/metrics/_context_precision.py) · [上游召回](../src/ragas/metrics/_context_recall.py)

用以下输入验收拟定的评分 revision。

- 参考为 `["abc","xyz"]`，检索为 `["abc","qqq","abc"]`，阈值为 `1.0`。相关序列为 `[1,0,1]`，精度为 `5/6`，召回为 `1/2`。
- Levenshtein 下 `"ab"` 与 `"ac"` 的相似度为 `0.5`，用默认阈值时命中。
- 空检索、有参考时两项均为 `0.0`。空参考时精度为 `0.0`、召回为 `undefined`。阈值设为 `0` 时仍须存在候选段落才能命中。最后一项修正了 `ec00339` 把空候选的占位零当成命中的行为，空分母则从 `NaN` 映射为已定义的结果状态。

## 4B · 模型裁判、投票与评分量表

阶段 3 的 `@numeric_metric`、`@categorical_metric` 和 `@ranking_metric` 运行调用者提供的 Python 函数。阶段 4 的 `metric.custom.numeric`、`metric.custom.categorical` 和 `metric.custom.ranking` 则接收评分标准与注入的 `LanguageModel`。模型通过 `StructuredPrompt` 给出结构化结果，指标声明值域、标签或长度，执行器负责统一校验、封存用量和汇总。[函数指标](../tallyjudge/api/metrics.py) · [模型协议](../tallyjudge/interfaces/models.py) · [提示词](../tallyjudge/core/prompts.py)

- **数值和类别。** `AspectCritic` 的二元裁决按数值 `0.0/1.0` 迁移，保留均值。`SimpleCriteriaScore` 按离散整数等级判分后转 `float`。类别结果保留声明的标签。首批投票用于二元、离散等级及类别，模型排序先支持单次判分，使用声明长度验收。
- **票数与用量。** `strictness` 默认为 1，正偶数向上取奇数。每票独立调用模型。三票 `[0,1,1]` 得 `1.0`，若其中一票因结构错误重问一次，应有三个逻辑调用、四次请求。多类别或多等级出现并列众数时取最早的一票。一次投票失败沿用原错误和已观测用量。整项评分命中缓存后新增请求数为零。旧单轮实现只向聚合函数传入一票，真实多票是显式行为变更。[旧二元裁判](../src/ragas/metrics/_aspect_critic.py) · [旧等级裁判](../src/ragas/metrics/_simple_criteria.py)
- **固定与逐行量表。** `metric.custom.rubric_without_reference` 与 `metric.custom.rubric_with_reference` 从配置取等级和说明，后者在绑定中显式要求 `reference`。`metric.custom.instance_rubric` 从当前行取 `rubrics: dict[str, str]`。返回等级分及反馈 `reason`。用同一回答换量表或参考答案，核对实际提示词、得分和缓存是否区分。反馈随评分写入结果，可能含样本文字。[固定量表](../src/ragas/metrics/collections/domain_specific_rubrics/metric.py) · [逐行量表](../src/ragas/metrics/collections/instance_specific_rubrics/metric.py)
- **复验条件。** 评分 revision、提示词版本、模型部署标识、有效票数和决定判分的规则都要能区分缓存与跨运行比较。规则正文需要保密时，用同一规范化内容生成摘要。逐行量表走绑定输入。类别与排序既要通过执行时的检查，也要通过存档读回的契约测试。[阶段 3 复核](phase-3.zh-CN.md#提交复核)

## 后续批次的接口边界

- **4C。** 引用片段评分用 `response` 和 `retrieved_contexts`，先固定无引文、部分匹配和全部匹配三类样本。摘要及 SQL 复用文本模型。表格比较的 DataComPy 和数据处理库仅随该插件安装，用缺行、缺列的样本核对分母和覆盖范围。[引用片段](../src/ragas/metrics/collections/quoted_spans/metric.py) · [摘要](../src/ragas/metrics/collections/summary_score/metric.py) · [SQL](../src/ragas/metrics/_sql_semantic_equivalence.py) · [表格](../src/ragas/metrics/collections/datacompy_score/metric.py)
- **4D。** 把用户、助手和工具消息以及工具名、参数、返回内容表示成可保存的 JSON。先用一条含两次工具调用的对话贯通 `Dataset → MetricBinding → RunResult → load_run`，再加入调用缺失、参数部分相同、重复和顺序改变的用例。消息格式由首个消费它的指标确定，外部框架的消息转换放在接入插件。[旧消息类型](../src/ragas/messages.py) · [工具调用准确率](../src/ragas/metrics/_tool_call_accuracy.py)
- **4E。** HHEM 复用裁判从回答提取陈述，分类器按批次判定每条陈述能否由上下文支持。加载权重由选用该插件的调用者触发。多模态插件先定义一个图文样本如何保存和重建、一个模型适配器如何接收它，再扩展现有纯文本 `PromptInput`。默认安装和插件枚举保持离线。[HHEM](../src/ragas/metrics/_faithfulness.py) · [多模态忠实度](../src/ragas/metrics/collections/multi_modal_faithfulness/metric.py)

## 每批怎么验收

1. 从 `ec00339` 记录同一组输入的旧行为，把分数、空值和有意差异直接写入 Python 测试。模型指标固定返回或固定权重，逐一核对请求次数、用量和失败状态。
2. 从公开工厂创建指标，经绑定运行。覆盖不同字段映射、同一插件的两份配置和存档读回。共用的输入校验、取消和缓存语义留在已有边界，指标只实现自己的算法。
3. 依赖集中放在根 `pyproject.toml` 的可选安装组，由静态 `PluginSpec` 声明所需分发包。分别验证默认安装下可列出插件、运行已安装能力，以及装上对应依赖后能真正评分。新实现沿用 Python 3.12+ 与根目录 `tallyjudge/`。[注册协议](../tallyjudge/registry/specs.py) · [编码总纲](implementation-standards.zh-CN.md)

阶段 4 的首个代码提交只需带上字符串相似度、注册项、可选依赖声明和行为测试。第二个提交补两个文本检索指标及上述边界样本。第三个提交用假模型交付单次判分和多票计数，随后分别交付 rubric 与排序。各批的实际支持范围随验收结果更新。[阶段 2b 安排](phase-2.zh-CN.md#11-分批实施与退出标准)
