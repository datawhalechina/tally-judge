# TallyJudge 架构与编码总纲

**适用范围**　根目录 `tallyjudge/`、包内插件及其测试。新增功能先确定调用者、结果和负责模块，再决定函数怎么写。阶段计划决定交付哪些能力。本总纲规定模块边界和公开行为，给出函数拆分与注释的默认写法。多种设计均能满足需求时，优先选择调用路径可读、影响模块少、默认依赖轻且能独立验证的方案。

| 遇到的问题 | 从这里开始 |
|---|---|
| 新指标、新供应商或新结果字段应改哪里？ | [模块职责与依赖](#模块职责与依赖)、[变更应落在哪里](#变更应落在哪里) |
| 这个判断、默认值或异常捕获该留吗？ | [日常设计判断](#日常设计判断) |
| 函数要不要拆？要写几行注释？ | [函数与类的拆分](#函数与类的拆分)、[注释与文档字符串](#注释与文档字符串) |
| 一个需求同时涉及数据库、API 和页面怎么办？ | [跨层功能示例](#跨层功能示例) |

## 一、架构与设计判断

### 模块职责与依赖

调用者把一批样本交给选定的指标，需要知道每行得了几分、哪些项没跑完。围绕这个用例拆分模块，评分公式可以单独与旧基线对照，故障可以定位到行和绑定，本地指标安装核心包就能运行。

```text
tallyjudge/
├── __init__.py     公开导入面，例如 Dataset、Evaluator、create_metric
├── api/            接受用户的选择，组装并发起一次评测
├── core/           样本身份、执行策略、评分状态、汇总与序列化
├── interfaces/     指标作者和调用者共同遵守的类型与协议
├── registry/       插件 ID、构造配置、实现的加载位置
└── plugins/        按能力组织的实现
    ├── metrics/text/                      已有本地指标
    ├── metrics/rag/、models/、embeddings/   阶段 2 设计
    └── storage/、adapters/ 等              后续按交付需要增加
```

**构造路径。** 调用者既可以传入自己的指标，也可以通过 `create_metric(id)` 让注册表按 ID 加载选中的工厂。`MetricBinding` 保存一次绑定的名字和字段映射。`Evaluator` 接受指标或绑定，统一成一组绑定，并检查本次运行的绑定名是否唯一。指标查询与构造到这里就结束了。[公共入口](../tallyjudge/api/evaluator.py) · [工厂](../tallyjudge/api/factories.py) · [注册表](../tallyjudge/registry/registry.py)

**执行路径。** `Evaluator.run` 将同步调用交给同一条 `arun` 路径。`TaskRunner` 将样本行交给 `MetricBinding`，绑定映射参数并通过指标的 `ascore_inputs` 执行评分，接收 `MetricResult`，将每个结果写入相应行的 `ScoreResult`。`RunResult.summary` 从这些行计算有效数、覆盖率和均值。尽管运行时执行了具体指标，执行器的源码只需要知道指标契约。[执行器](../tallyjudge/core/execution.py) · [指标契约](../tallyjudge/interfaces/metrics.py) · [结果模型](../tallyjudge/core/results.py)

**导入方向。** `api` 可组合 `core`、`interfaces`、`registry`。执行器可导入指标契约，契约可引用 `core.results`、`core.errors` 等中立类型。插件依赖契约和中立值类型。注册清单记录实现路径，解析选中的插件时才加载其代码。新增 SDK 的导入和响应转换留在对应插件，安装选项集中在根 `pyproject.toml`。例如阶段 2 的 RAG 指标接收注入的模型，执行器无需知道哪个供应商回答了问题。[插件清单](../tallyjudge/registry/builtins.py) · [目标目录与迁移账本](plugin-architecture.zh-CN.md) · [阶段 2 设计](phase-2.zh-CN.md)

### 变更应落在哪里

用改动范围检验架构。评审时沿着一个具体变化走一遍，就能发现多出来的耦合。

| 变化 | 负责位置 | 怎样判断边界合适 |
|---|---|---|
| 增加同一契约下的文本指标 | `plugins/metrics/text/` 放公式、`MetricSpec` 和评分版本，内置项在 `registry/builtins.py` 登记 | 原有 `Evaluator` 可运行它，自定义指标也可直接传入。执行循环里出现具体指标名，说明评分逻辑放错了位置 |
| 修改超时或取消语义 | `core/config.py` 定义策略，`core/execution.py` 实施，结果模型表达可观察状态 | 现有指标无需逐个修改。通过公开入口区分未启动、已取消和因清理超时中止的运行 |
| 为已有模型协议接入另一家供应商（阶段 2 以后） | 对应模型或嵌入插件负责 SDK 调用、响应转换和配置，登记其构造方式 | 同一指标可以换入模型替身。装核心包可继续运行本地评测。协议只随消费者需要的新能力演进 |
| 改动结果字段或评分口径 | 数据含义由结果模型或指标定义，导出与消费端随契约调整 | 重命名保留算法基线。分母或空输入语义改变时更新指标 `revision` 和对照样本。存档格式另看 `schema_version` |

一个能力可以先是一个文件。只有多种实现确实要被同一个消费者替换，或项目已经承诺了用户插件接口，才为它定义协议。协议只包含调用者要用的操作。构造时选择实现，异步算法按步骤调用。一次模型接入无需再为每个调用点添一层工厂。LangChain、LangGraph 等框架如需接入，在各自的适配插件里转换对象。旧 `src/ragas/` 保留作迁移对照。

### 日常设计判断

**哪一种受支持的输入或故障会走到这个分支？这一层掌握了哪些信息，能决定如何处理？** 新增校验、默认值或异常捕获时，用一个具体调用回答这两个问题。

| 情形 | 约定与例子 |
|---|---|
| 接收外部数据 | `Dataset.from_records` 确定行 ID，绑定输入入口依据 `MetricSpec` 做严格类型校验，调度器接收插件输出时检查浮点数和值域。每个事实在同一条路径上由一处负责 |
| 内部已经知道事实 | 已经验证为 `list[str]` 的上下文可以直接拼接。必需字段直接访问。`.get` 适用于契约允许缺省且默认值有业务含义的字段。内部反复出现 `getattr`、`or {}` 或 `assert x is not None` 时，先查构造后的类型 |
| 计算得到有效零分或无法评分 | 文本不相等可以成功得 `0.0`。运行异常记录失败，空数据集的均值和覆盖率为 `None`。汇总只计入成功分数 |
| 需要转换失败 | 导入或构造插件时补上插件 ID，用 `raise … from exc` 保留原因。行输入无效时，默认只记一条可定位的 WARNING，结果提供 `field`、`expected`。指标异常由转换为行结果的一层记一次 ERROR traceback，结果只保留异常类型等允许的字段。输入校验的详细 traceback 放在 DEBUG |
| 涉及并发、取消或所有权 | 调度器决定终态，清理共用有界宽限，迟到结果不能覆盖已记录的超时。`RunCollector.snapshot()` 向调用者交付独立副本，内部只读传参可直接复用。由谁创建事件循环或客户端，就由谁负责其生命周期 |

当前层无法恢复、转换错误或清理资源时，让异常传播。新配置项应对应调用者能够做出的选择。例如阶段 2 的结构化输出重问由模型帮助类负责，请求用量在真实请求边界记录，指标和调度器无需各写一份重试或计数逻辑。

## 二、具体编码要求

### 接口与数据类型

- 公共入口从调用代码读起。`Evaluator(metrics=[...], config=...).run(dataset)` 分开了可复用配置和单次输入。新增影响行为的可选参数宜用具名参数，交代默认值、失败行为和所有权。同步入口委派异步实现，使用 `run/arun`、`score/ascore` 的既有命名。
- 对外类型和方法写完整注解。固定结构用具名类型，动态行或插件配置可以是映射。`None` 必须有可解释的含义。公共配置允许省略的值，在构造时归一化，内部成员的注解应表达归一化后的事实。只在真实的动态边界使用 `Any`、反射或窄范围类型断言。
- 模块、函数和参数用 `snake_case`，类型用 `PascalCase`，常量用 `UPPER_SNAKE_CASE`。指标名如 `ExactMatchMetric`，插件 ID 如 `metric.text.exact_match`。`metric_key` 区分一次运行中的多个绑定，插件 ID 标识实现，`api_version` 标识插件接口，`revision` 标识评分口径。增删公开符号时一并检查包导出和调用示例。[全局命名与改名清单](plugin-architecture.zh-CN.md#全局命名规范)
- Python 下限为 3.12。缩进、引号、导入排序与行宽使用根配置。公开结果只携带可序列化的中立值，客户端、密钥、原始 SDK 对象和诊断 traceback 留在相应运行边界。[工具与依赖配置](../pyproject.toml)

### 函数与类的拆分

| 代码是什么形态 | 采用的写法 | 本项目的例子 |
|---|---|---|
| 一眼可读的公式，只在这里使用 | 就地表达，名字通过参数和结果给出 | `ExactMatchMetric.ascore` 一句 `MetricResult(float(reference == response))` 已经说明算法。再拆一次 `_compare_strings` 只会多一次跳转 |
| 有独立输入、输出和业务含义的计算 | 用一个纯函数，留在所属模块。共享需求出现后再提高可见性 | `aggregate` 根据行结果和绑定键汇总。阶段 2 的余弦计算也适合单独测试零向量、负分和精度 |
| 一项操作需要模型、配置或资源 | 用持有这些依赖的对象，构造时传入。主方法按调用顺序写 | Faithfulness 的规划是提取陈述、用上下文判定、按返回的判定计分。提示词和公式归指标，模型请求归适配插件 |
| 几步共享同一运行的队列或截止时间 | 只有一个调用点且状态局限于本次运行时可用局部函数。独立职责可以单独命名和测试 | `TaskRunner.arun` 同时等待执行和清理的截止时间。`fill` 补充任务，`cancel` 把已封存的任务移入清理集合，`_Job.finish` 处理正常或异常的完成结果 |

拆完以后，读者应能靠函数名和参数读出业务顺序。提取函数时，参数列明它读取的值，返回值或一个明确的写入点说明它改变了什么。若同一函数既控制请求重试、又解释供应商响应、还决定评分分母，应按各自变化的原因拆开。类适合保存有生命周期的状态或实现真正的可替换契约。单纯计算从函数起步。[本地指标](../tallyjudge/plugins/metrics/text/exact_match.py) · [汇总函数](../tallyjudge/core/aggregation.py)

### 注释与文档字符串

源码、标识符和 docstring 默认用英文，设计与迁移说明可用中文。公共操作先用一句话说清调用效果。读者从签名难以得知的信息才展开，例如超时的单位是秒，外层取消后还可能等待一个清理宽限期。

| 放在哪里 | 写什么 | 什么时候可以省 |
|---|---|---|
| 模块开头 | 模块负责的能力。迁移算法保留原作者、许可证与来源，必要时注明基线和行为差异 | 常规 `__init__.py` 仅导出具名符号时无需复述文件列表 |
| 公共类、协议及其方法 | 类说明职责和所有权，构造参数的约定集中写在类或构造函数的一处。方法首句说明调用效果，按需要补顺序、单位、缺省语义、副作用、取消和调用者可处理的异常。协议写一次通用契约 | 实现沿用协议时只补算法或适配器特有语义。同一段参数说明无需复制到每个实现 |
| 私有函数或局部变量 | 名字表达不了的不变量、算法选择、数据形状等。只有一个易懂表达式时，让表达式直接呈现 | `_compare_strings` 的逐行解释、变量赋值的复述、方法名的中文翻译均可省 |
| 特殊分支的上一行 | 说明改动这一步会破坏的行为，紧邻受约束代码。如调度器为何不能给仍占并发名额的超时任务补位 | 能通过命名或拆分讲清的先改代码。用例能充分表达的普通分支无需再加旁白 |

多行 docstring 先写一句可单独阅读的摘要，空一行后补契约。参数的角色或单位不明显时用 Google 风格 `Args:`，返回结构需要解释时用 `Returns:`，调用者可处理的异常用 `Raises:`。类型已经在签名中表达。无需给每个方法机械补齐三段。下面是给现有 `RunCollector.snapshot` 写 docstring 的示例，方法行为来自当前实现。[结果收集器](../tallyjudge/core/results.py)

```python
def snapshot(self) -> RunResult:
    """Return a detached copy of the current or most recent run.

    Raises:
        RuntimeError: If this collector has never accepted a run.
    """
    if self._result is None:
        raise RuntimeError("No run has been accepted by this collector")
    return copy.deepcopy(self._result)
```

`# Compare the strings.` 逐字复述了相等运算，可以删掉。Faithfulness 的分母则值得解释。按阶段 2 的设计，若生成三条陈述但只收到两个判定 `[1, 0]`，分数应为 `0.5`。空判定列表在进入这段计算前已单独处理。

```python
# Revision 1 uses returned verdicts as the denominator when the judge omits a statement.
score = sum(verdicts) / len(verdicts)
```

修改行为时同步改动紧邻的注释、公开说明和覆盖该行为的测试。临时代码若需要 `TODO`，写明什么条件满足后可以移除。已有议题时附编号。更长期的迁移决策记入阶段文档。

## 跨层功能示例

以“按运行 ID 查看评测为什么中止”为例。当前 Python 调用者已经能读取 `RunResult.run_status` 和 `stop_reason`。下面推演以后引入持久化和界面时，这个需求如何贯穿各层。存储属于后续阶段，HTTP 和界面尚未排期。

| 边界 | 这个需求如何实现 | 如何验收 |
|---|---|---|
| 核心与 Python API | 调度器记录一次运行为什么停止，调用者从 `RunResult` 获取同一事实。`cleanup_timeout` 表示清理宽限耗尽 | 任务清理超时后运行呈现 `stopped`，未开始的项仍为 `not_started` |
| 存储及 SQL（如果选用） | 存储插件映射结果。以 `run_id` 查运行，以 `input_index` 恢复行序。若逐条存分数，以 `(run_id, case_id, metric_key)` 区分同指标的多次绑定。事务由负责保存此次结果的操作持有 | 对既有表新增可空停止原因时，以版本化迁移交代旧行语义。用含旧行的库升级、写入新结果、读回对照。写入中断时验证事务回滚，重放相同结果不增加评分条数。承诺支持降级时再验证降级 |
| HTTP（若提供服务） | 例如 `GET /runs/{run_id}` 委派一次查询，边界负责参数、找不到运行的响应和输出结构。状态值与 Python 结果保持同义 | 已有运行即使状态为 `stopped` 也返回 `200` 和原状态，未知 ID 返回 `404` |
| 前端（若启动界面项目） | 数据访问模块持有调用和响应类型，页面负责选择运行、加载态和展示文案。如有共享接口描述，从该描述派生客户端类型。页面直接展示记录的停止原因 | 没有历史原因时显示“未记录”，得分 `0.0` 正常显示为零，失败行显示状态。页面无需从覆盖率或警告列表推断原因 |

新增一列、一个 API 字段、一段展示文案是三种不同的改动。只有确实需要保存的新事实才涉及数据迁移。仅调整显示文字时，在消费端完成。表结构版本与对外结果 `schema_version` 各自按其读者的兼容要求演进。

## 参考实现与取舍

下面对照 Dify 标签功能的两个发布版本。Dify 在后端架构文档中把传输、用例和适配器分别交给不同模块。TallyJudge 选择其中适用于进程内评测库的职责划分。[Dify 1.6.0](https://github.com/langgenius/dify/releases/tag/1.6.0) · [Dify 1.20.0](https://github.com/langgenius/dify/releases/tag/1.20.0) · [Dify 后端架构说明](https://github.com/langgenius/dify/blob/1.20.0/docs/BACKEND_ARCHITECTURE.md)

- **前端接口。** 1.6 的 `createTag` 封装请求路径、请求体以及 `Tag` 返回类型。1.20 的标签绑定 hook 通过共享 `consoleClient` 提交变更，通过 `consoleQuery` 找到应刷新的查询。TallyJudge 将来增加界面时，展示和数据访问也应有各自的归属。[1.6 标签请求](https://github.com/langgenius/dify/blob/1.6.0/web/service/tag.ts) · [1.20 标签变更 hook](https://github.com/langgenius/dify/blob/1.20.0/web/features/tag-management/hooks/use-tag-mutations.ts)
- **API。** 1.6 的标签路由把创建工作交给 `TagService.save_tags`。1.20 的路由声明请求和响应模型，组装 `CreateTagInput` 后调用 `application_services().tags.create_tag`。可借鉴的是让传输协议在入口结束。TallyJudge 目前只有进程内 Python 入口，一个 `Evaluator` 足以发起评测。[1.6 标签路由](https://github.com/langgenius/dify/blob/1.6.0/api/controllers/console/tag/tags.py) · [1.20 标签创建入口](https://github.com/langgenius/dify/blob/1.20.0/api/controllers/console/tag/tags.py#L139-L156)
- **数据库。** 1.6 源码里的迁移将标签和绑定表、查询索引及降级操作放在同一个版本中。1.20 源码里的知识空间标签迁移把租户、空间、标签三列组合定义为唯一约束。对 TallyJudge 的启发是把持久化身份落实到约束和升级测试里。[标签与绑定表迁移](https://github.com/langgenius/dify/blob/1.6.0/api/migrations/versions/3c7cac9521c6_add_tags_and_binding_table.py) · [空间标签绑定迁移](https://github.com/langgenius/dify/blob/1.20.0/api/migrations/versions/2026_08_13_1200-9d4e6f8a1b2c_add_knowledge_fs_space_tag_bindings.py)
- **算法可读性。** Torch-RecHub 的 `torch_rechub/models/ranking/deepfm.py` 在构造时装配线性项、FM 和 MLP，`forward` 里能直接看到三部分如何合成结果。`torch_rechub/basic/layers.py` 中的 `PredictionLayer` 在构造时确定任务类型。评测插件也可以在构造时确定组件，让读者从评分方法里看到公式。

## 评审与验证

| 审查的问题 | 在本仓库怎样证明 |
|---|---|
| 公式或输入语义变了吗？ | 用固定样本验证大小写、空输入、分母和值域等实际差异。迁移复用旧环境生成的基线，注明来源及评分版本。[迁移对照测试](../tests/tallyjudge/migration/test_string_baselines.py) |
| 新插件会让所有人安装新依赖吗？ | 从公共入口或注册清单列出插件，验证仅选择具体实现时才加载它。SDK 的接口适配可用真实客户端配离线传输验证。[注册契约测试](../tests/tallyjudge/contracts/test_registry.py) |
| 改动校验或状态后，两条公开路径还一致吗？ | 分别从批量评测与承诺严格校验的直接入口调用，同时覆盖内置与自定义指标。取消、快照和迟到结果用可控制的任务验证。[运行回归测试](../tests/tallyjudge/unit/test_execution_regressions.py) · [快照与状态测试](../tests/tallyjudge/unit/test_result_status.py) |

**阶段 1 的具体约定。** `MetricBinding` 构造完成就有确定的 `key` 和列映射。每条评分路径只校验一次，并把校验所得的参数交给指标。具名指标在 `MetricBase` 的检查入口委派给 `ascore`，直接实现映射协议的插件由绑定做入参检查。复验时同时覆盖这两种插件、独立快照和取消后的终态。

提交前的判断标准是改动能否用一个真实调用例说明、变更能否局部完成、失败后调用者能否获知发生了什么。检查命令和各阶段的交付范围分别维护在贡献者指引与路线中。[贡献者执行指引](../AGENTS.md) · [迭代路线](roadmap.zh-CN.md)
