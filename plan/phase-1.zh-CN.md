# 阶段 1：纯本地评测闭环

更新：2026-11-19。状态：已实现，Windows / Python 3.12 本地验收通过；跨平台 CI 待运行。Python >=3.12。范围见[路线](roadmap.zh-CN.md)，命名与边界见[总设计](plugin-architecture.zh-CN.md)。下文阶段 1 接口已实现，后续能力仍以总设计中的阶段边界为准。

## 目标

给定已有回答与参考答案，不调用模型、不安装外部框架，完成批量评分并返回可追踪的完整结果。

本阶段同时固定 RunResult schema v1 中后续阶段沿用的字段：指标绑定 key、配置快照、输入 manifest 和行状态。这些字段在阶段 1 就要定准，后续只做增量扩展。

## 最小文件范围

```text
tallyjudge/
├── __init__.py                      # 导出 Dataset、Evaluator、MetricBinding、RunCollector、create_metric
├── api/{evaluator,factories}.py     # factories 本阶段只实现 create_metric
├── core/{dataset,results,aggregation,execution,config,errors}.py
├── interfaces/metrics.py            # Metric 协议、MetricSpec、MetricBase 帮助类
├── registry/{registry,specs,builtins}.py
└── plugins/metrics/text/{exact_match,string_presence}.py
tests/tallyjudge/
├── installation/test_installation.py  # 由现有 tests/tallyjudge/test_installation.py 移入
├── unit/{test_dataset,test_metrics,test_aggregation,test_evaluator}.py
├── contracts/test_registry.py         # 含内置指标满足 Metric 协议的静态检查
└── migration/test_string_baselines.py  # 两种文本指标的公开入口回归
tools/migration/generate_baselines.py  # 只在旧基线环境运行，不进入默认 pytest/ruff/pyright 范围
```

迁移来源：原 collections/_string.py 的两个算法、metrics/result.py 的结果概念。executor.py 只参考按序号回填结果的做法；它一次性创建全部 Task、用 threading.Event 取消，这两点不沿用。新实现禁止 import ragas，原文件不搬走、不删除。

## 旧实现基线

- 基线提交为 ec00339。用 `git worktree add <目录> ec00339` 取出，在该目录按它自己的 pyproject 建立旧环境。根目录的新环境不安装 ragas。
- `tools/migration/generate_baselines.py` 可在独立旧环境生成临时报告，记录来源、运行环境及旧版分数或异常。
- 日常回归直接用 Python 参数化用例核对分数、输出类型、缺字段和错误类型。旧版与新包的校验差异见下表。

## 两个指标的行为约定

| 情况 | 原实现（ec00339） | 新实现 | 处理 |
|---|---|---|---|
| 两个 str 比较 | ExactMatch 为 `float(reference == response)`；StringPresence 为 `float(reference in response)` | 相同 | 保留 |
| 返回值类型 | float 0.0/1.0 | float 0.0/1.0，JSON 写 0.0/1.0 | 保留 |
| 大小写、首尾空白 | 不处理 | 不处理 | 保留，不新增归一化 |
| 双方均为空串 | ExactMatch 得 1.0 | 相同 | 保留 |
| reference 为空串 | StringPresence 恒得 1.0（`"" in s`） | 相同 | 保留并在指标文档注明；是否改判为 invalid 留给以后的 revision |
| 非 str 输入 | ExactMatch 不检查，如 `1` 与 `"1"` 得 0.0；StringPresence 用 assert 检查，`python -O` 下失效 | 按 MetricSpec 校验，status=error，类别 validation_error，不做类型转换 | 有意差异，登记于迁移记录 |
| 缺少字段 | 调用时抛 TypeError | status=error，类别 validation_error | 有意差异，登记于迁移记录 |

两个指标在新包中的初始 revision 均为 "1"。revision 只在新包内递增，不沿用旧版本号。上述差异属于输入契约，str 输入下的公式不变。

## 输入、指标绑定与结果

- Dataset.from_records(records, id_field="case_id") 深复制输入。冻结后，调用者再修改原数据不会影响 Dataset。
- case_id 接受 str 和 int，int 规范化为 str，其他类型报错。规范化后出现重复（包括 1 与 "1"）时报错。
- ID 必须全部提供或全部缺省，混合时报错，避免生成的 ID 与显式 ID 冲突。全部缺省时生成 `row-{input_index}`，manifest 记 id_source="generated"。生成的 ID 只在该 Dataset 内稳定；跨运行比较应提供或复用持久化 ID，后续的比较工具遇到 generated ID 必须警告。
- 第一阶段只支持字典型文本样本，不实现多轮、媒体和框架对象转换。
- MetricSpec 声明 id、revision、inputs（参数名 → 类型注解，支持 str、list[str] 等泛型；两个内置指标仍只用 str）、输出类型、值域与方向。MetricSpec 只在插件类上定义一次。
- MetricBinding(metric, key=None, input_map=None) 把指标接入一次运行：
  - key 默认取 metric.spec.id，同一次运行内必须唯一。同一指标用不同参数或不同字段运行多次时，靠 key 区分。
  - input_map 为 {指标参数: 数据集字段}，缺省时按同名字段取值。
  - 直接传入 Metric 等价于使用默认绑定。
  - key 重复，或 input_map 中的参数不在 spec 里，属于全局配置错误，在运行开始前抛出。
- 单行缺字段或类型不符时，在该行、该绑定的校验环节形成 error，不影响其他行和其他绑定。
- Metric.config_snapshot() 返回构造参数的 JSON 兼容字典，不含模型对象和凭据。两个内置指标返回 {}。
- MetricResult 保留 value/reason。
- RunResult 保存：schema_version、run_id、输入 manifest、metrics（key → 插件 ID、revision、input_map、config_snapshot）、执行配置快照，以及全部 RowResult。
- 输入 manifest 逐行记录 case_id、input_index、id_source，以及行内容的 sha256（基于规范 JSON 计算）。
  - to_dict(include_inputs=True) 是默认行为，同时写出原始行。
  - include_inputs=False 时只保留 manifest，适用于含敏感数据的场景。
- RowResult 保存 case_id、input_index，以及按绑定 key 索引的 status/value/reason/error。
  - 本阶段任务 status：not_started、running、success、error、invalid_output、timeout、cancelled。not_started 表示尚未进入指标任务；running 只用于运行中快照。
  - error 保存 category、exception_type、field、expected。校验错误只填 field（指标参数名）和 expected（MetricSpec 的类型注解），exception_type 留空；真正的指标异常记录异常类名。结果不保存原始异常消息或原始输入值。校验失败默认仅一行 WARNING，链式 traceback 仅在 DEBUG；指标异常仍记录 ERROR traceback。
- NaN 和无穷值归为 invalid_output，不参与汇总。
- 汇总按绑定 key 输出：总数、各状态计数、有效数、覆盖率、有效样本均值。零有效样本时均值为 null；空数据集的 coverage 也为 null，有样本但无有效值时 coverage 为 0.0。
- to_dict/to_json 输出标准 JSON。失败值写 null，不以 0 代替，也不写非标准的 NaN。磁盘保存与后端协议留到阶段 3。

## 已实现 API

```python
# 以下示例可在 Notebook 直接运行；脚本中需放进 async 函数并用 asyncio.run 调用。
from tallyjudge import Dataset, Evaluator, MetricBinding, create_metric

dataset = Dataset.from_records([
    {"case_id": "a", "answer": "Paris", "reference": "Paris"},
    {"case_id": "b", "answer": "Lyon", "reference": "Paris"},
])
evaluator = Evaluator(metrics=[
    MetricBinding(create_metric("metric.text.exact_match"), input_map={"response": "answer"}),
    MetricBinding(
        create_metric("metric.text.string_presence"),
        key="presence",
        input_map={"response": "answer"},
    ),
])
result = await evaluator.arun(dataset)
report = result.to_dict()
```

- `TaskRunner` 把独立的样本行交给 `MetricBinding.ascore_row`。绑定完成字段映射，再调用指标的 `ascore_inputs`。具名指标继承 `MetricBase` 实现 `ascore`，直接调用时用 `score(**inputs)` 或 `await ascore_inputs(inputs)` 取得相同的严格校验。直接实现 `Metric` 协议的插件也可以接收映射。
- create_metric 位于 api/factories.py，由包根导出。
- 批量入口 run/arun 共用同一执行器。本阶段不实现指标级的 score_batch/ascore_batch。

## 执行约定

- ExecutionConfig 包含四项：
  - max_concurrency：默认 8，必须为正整数。
  - timeout：单任务秒数，默认 None，即不限时。
  - failure_policy：默认 "record"，可选 "raise"。
  - cleanup_timeout：取消后等待任务退出的宽限秒数，默认 5.0，必须为有限正数。重复取消不会重新计时。

  以上默认值先用于离线用例，不能当作供应商限流配置。
- 不设 preserve_order。结果总是按输入序号和绑定序号重建；完成顺序只影响事件，而阶段 1 没有事件。
- 调度单位为"样本 × 绑定"。只创建受上限约束的在途任务，不一次创建全量 asyncio Task。单指标调用不再额外套一层批量执行器。
- record 模式：记录失败并继续；若超时任务在清理宽限期后仍未退出，则停止本次运行的后续调度，已启动的其余未完成项为 cancelled，未启动项为 not_started，避免残留任务与新任务叠加突破并发上限。
- raise 模式：遇到 error 或 timeout 时停止新调度，取消在途任务，并最多等待 cleanup_timeout，然后抛出 EvaluationAborted。它的 partial_result 是 RunResult，已启动未完成的任务记为 cancelled，未启动任务记为 not_started。invalid_output 在两种模式下都只记录，不触发停止。
- 外部取消：原样重新抛出 CancelledError，不替换、不包装、不子类化，以免破坏 asyncio.timeout、TaskGroup 的取消计数语义。需要部分结果的调用者传入 collector：`arun(dataset, collector=RunCollector())`；取消后，`collector.snapshot()` 返回 RunResult，其中已启动任务具有终态，未启动任务保留 not_started。
- 同步 run/score 在已有事件循环中抛错，提示改用 await；不使用 nest_asyncio，Notebook 示例一律写 `await arun()`。默认不在工作线程中另起事件循环：用户的异步客户端通常绑定原事件循环，换循环会在阶段 2 引入难排查的问题。
- 调度器同时观察每项工作的执行截止时间和正在清理任务的退出期限。一个任务进入清理后仍占用并发名额，其他已启动的任务到期时照常收到取消。已写入的超时或取消状态在清理期间封存。
- 清理超时的任务记录在 RunResult.cleanup_warnings，每项含 case_id、input_index、metric_key、category="cleanup_timeout"。结果终态由调度器写入，残留任务的迟到成功或异常不会修改快照；迟到异常仍写本地日志。
- arun 不拥有调用者的事件循环，停止等待不代表强杀协程；持续吞取消的任务可能仍在运行。调用者外层 asyncio.run 在关闭循环时仍可能等待这些任务。同步 run 使用同线程自有循环及有界关闭，避免标准 asyncio.run 的无限清理等待。阻塞事件循环的代码、SDK 工作线程和外部进程无法靠协程取消强停；需要硬隔离时应由进程级边界处理，不在本阶段引入隐式线程或强杀机制。
- 不提供自动重试，不用线程强停，不自动保存。重试与用量归属在阶段 2 定案。

## 注册约定

- 静态清单 builtins 为两个内置插件登记 PluginSpec：id、kind、api_version、factory 加载路径、required_distributions。MetricSpec 不在清单中重复。
- list 只读静态清单，不导入任何插件模块。
- describe 可以导入所选插件模块来读取 MetricSpec，但插件模块的顶层禁止导入第三方 SDK，由安装隔离测试和契约测试检查。
- create 先校验 api_version 和配置，再解析工厂。创建后校验 PluginSpec.id 与 MetricSpec.id 一致；内置插件的一致性由契约测试覆盖。加载、配置和构造异常包含插件 ID，并使用 raise ... from exc 保留原始异常链。
- 重复 ID 默认报错，只有显式 replace=True 才允许覆盖。用户可以显式注册自定义工厂。不扫描文件系统，不动态安装依赖。

## 工程前置（阶段 1 内完成）

- 在 pyproject 的 pytest addopts 中增加 `--import-mode=importlib`。测试按 unit/contracts/migration/installation 分目录，同名测试文件不会冲突。
- 增加 CI 步骤：构建 wheel，在仓库外的临时目录新建虚拟环境安装，然后导入。现有安装测试在仓库根目录运行，flat layout 会从当前目录找到源码，不能证明打包完整。
- tools/migration/ 不加入 ruff、pyright、pytest 的默认范围，也不进入 wheel。

## 验收清单

- 从公开入口执行两种文本指标的固定用例。评分口径和有意收紧的输入校验都有 Python 断言，干净检出即可收集全部测试。
- 混合成功、失败、延迟三行时：结果长度不缩水，ID 不变，顺序稳定；同一行其他绑定的成功结果保留。
- 同一指标以不同 key 绑定两次时结果互不覆盖；key 重复在开始前报错；input_map 生效。
- 峰值并发受配置限制。外部取消原样传播 `CancelledError`，已配合取消的任务完成清理。宽限耗尽的任务写入 `cleanup_warnings`，并由所属事件循环保活至退出或循环被释放。`RunCollector` 和 `EvaluationAborted.partial_result` 均保留各行的可解释状态。
- Dataset 不受调用者后续修改影响。重复 ID、混合提供 ID 时明确报错。空数据集返回空结果和 null 均值。
- 列出插件时不导入插件模块；描述插件时不导入 SDK。未知插件、重复插件、ID 不一致和无效配置都得到清晰错误。
- pyright 通过，两个内置指标可以赋值给 Metric 协议类型。
- 标准 JSON 往返（include_inputs 为 True 和 False 两种情况）不丢失状态、绑定 key、指标版本、manifest 和 ID。
- 新环境不安装、不导入 ragas、LangChain、OpenAI、Instructor、Pillow、图算法库即可运行。
- Python 3.12 在 Windows 和 Linux 上测试通过。在仓库外导入安装好的 wheel，确认不是从当前目录碰巧找到了源码。

## 不进入阶段 1

模型与嵌入、费用汇总、运行事件、实验装饰器、指标级批量方法、自定义指标装饰器、CSV/JSONL 存储、CLI、LangChain/LangGraph 接入、图生成、可视化。只在确有调用者时增加对应接口，避免为未来能力预建空框架。

## 本次实现与验收记录（2026-11-15）

- 运行链路：`Evaluator.run/arun` → `TaskRunner.arun` → 按 `MetricBinding.input_map` 取字段 → `MetricSpec` 校验 → `Metric.ascore_inputs` → `ScoreResult` → `RunResult.summary/to_json`。
- 迁移：`collections/_string.py::ExactMatch` → `plugins/metrics/text/exact_match.py::ExactMatchMetric`；`StringPresence` → `plugins/metrics/text/string_presence.py::StringPresenceMetric`。两者均复制公式、改名并重构输入契约，原文件保留。`MetricResult` 保留 value/reason 概念；其余阶段 1 契约、注册和执行实现为新增。
- 两个指标的评分 revision 均为 `"1"`。17 组输入在 `ec00339` 的独立旧环境分别运行两个指标，形成 34 组行为对照。本地提交 `2a1b99a` 收录阶段 1 的实现，`50ccb2d` 收录自包含的迁移断言和安装使用的 `py.typed`。[文本指标回归测试](../tests/tallyjudge/migration/test_string_baselines.py)
- 明确拒绝 bool 类型的 case_id，避免 Python 的 bool/int 继承关系导致 ID 混淆。数据以规范 JSON 冻结，读出时返回副本；非 JSON 对象不进入文本 Dataset。
- `MetricSpec` 阶段 1 只接受 float 输出，提供 `value_range` 和 `direction`；类别和排序输出按后续阶段扩展。`PluginSpec.config_type` 使用 Pydantic 配置模型，默认拒绝任何参数；自定义工厂如需 describe，可在工厂的 spec 属性上引用指标类的同一个 MetricSpec。
- `RunCollector.snapshot()` 返回独立副本。运行中快照区分 running 与 not_started；最终取消或失败停止后保留全部行，不把未启动项标成 cancelled。停止期间再次收到取消不会延长清理宽限期；未退出的任务单独记录 cleanup_warnings，内核不强停阻塞代码。
- 导出字段为 `schema_version/run_id/run_status/stop_reason/manifest/metrics/execution_config/rows/inputs/summary/cleanup_warnings`。每行的 metrics 以绑定 key 索引，summary 含 total/status_counts/valid_count/coverage/mean。`include_inputs=False` 只省略 inputs，不对自定义 reason 或 config_snapshot 做自动脱敏。
- 首轮本地 Windows / CPython 3.12.7：86 项测试通过；Ruff 检查和格式检查通过；Pyright 零错误、零警告。测试覆盖真实基线、无 SDK 导入、受限 Task 数、失败停止、重复取消、asyncio.timeout/TaskGroup、完整结果和 JSON 导出。
- 已构建 sdist 和 wheel；在仓库外新建仅安装 wheel 及运行依赖的虚拟环境，验证从 site-packages 导入、py.typed 存在、两个指标可以评分。wheel 不包含 src/ragas、旧示例或迁移工具。
- CI 已在原矩阵（Linux 3.12/3.13、Windows 3.12、macOS 3.12）增加仓库外 wheel 验证。本轮未触发远程 CI，因此不把 Linux/macOS 通过写成已验证事实。
- 发布版本号/tag 仍属于阶段 0 遗留项；本轮构建仅用于本地验证，没有创建 tag、发布或提交构建制品。

复验新包（仓库根目录）：

```bash
uv run pytest
uv run ruff check tallyjudge tests/tallyjudge
uv run ruff format --check tallyjudge tests/tallyjudge
uv run pyright
uv build --out-dir <临时制品目录>
uv run python tests/tallyjudge/installation/test_installation.py --wheel-dir <临时制品目录>
```

如需重新核对上游行为，在 `ec00339` 的独立旧环境运行下列工具。`--source-root` 指向对应提交的 checkout 或 Git 归档，`--output` 指向临时目录。归档安装可设置 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RAGAS=0.4.4`。

```bash
<旧环境的Python> <当前仓库>/tools/migration/generate_baselines.py --source-root <ec00339源码目录> --output <临时目录>/string-report.json
```

### 阶段 1 审查修复

- 输入类型改用预构建的 Pydantic TypeAdapter，并在 validate_python 时指定 strict=True。支持 list[str]、dict[str, list[int]] 等注解，嵌套元素类型不符时明确报错，不把单个字符串当文本列表。
- float 输出接受子类，成功后统一转为内建 float；bool、int、NaN、无穷值和越界分数仍为 invalid_output。真实 numpy.float64 用例在临时 NumPy 环境单独验证，主包依赖不变。
- ErrorInfo 的 message 已替换为 exception_type，之后又加入 field/expected；仍处于未发布的 schema v1 开发阶段。exception_type 只记录实际抛出的指标异常，不为超时或非法返回值虚构异常类名。日志名称为 tallyjudge.core.execution；校验 WARNING 不含原始数据，DEBUG 诊断与指标异常 ERROR traceback 可能含敏感信息，由调用者配置输出和访问权限。
- 默认环境回归：109 项通过，1 项 NumPy 可选测试跳过；该 NumPy 测试在临时依赖环境中单独通过。测试含吞取消、外层 asyncio.timeout、单任务超时、失败停止、重复取消不延长期限、迟到结果不覆盖、同步入口子进程 5 秒看门狗、异常链和严格泛型校验。Ruff 与 Pyright 通过。

### schema v1 冻结前的状态与诊断修订

| run_status | stop_reason | 含义 |
|---|---|---|
| running | null | 运行中的快照 |
| completed | null | 所有任务已处理；record 模式下可以包含失败或超时，不等于全部评分成功 |
| aborted | failure_policy | raise 策略触发中止，EvaluationAborted.partial_result 具有该状态 |
| aborted | internal_error | 调度器自身异常导致中止，异常继续向上传播 |
| cancelled | external_cancel | 外部取消，原 CancelledError 继续向上传播；快照保存状态 |
| stopped | cleanup_timeout | record 模式放弃清理后停止调度，正常返回但明确标识未完成 |

任务初始状态为 not_started，实际进入任务时变为 running。取消只作用于已经开始的工作；未调度或尚未进入任务的项目保持 not_started。summary.status_counts 同时统计上述状态，以便快照和最终报告都可按 ID 解释。

校验诊断的 field 是指标参数名，不是 input_map 映射后的数据列名；expected 来自 MetricSpec，例如 str 或 list[str]。一行有多个问题时按 spec 声明顺序报告第一个；缺字段和类型不符均能定位参数。未声明的多余参数只报告 validation_error，不把用户提供的键或值放入诊断。

对于一项有效输入，框架在从字段映射到插件调用的路径上只执行一次严格校验，并将 Pydantic 返回的值交给指标。例如声明 `float` 的参数接受整数 `1` 时，指标拿到 `1.0`。未通过严格校验的值仍产生 `validation_error`。两个文本指标的具名 `ascore` 直接执行公式。

外层 asyncio.timeout 到期后，异常传播还可能延后至多一个 cleanup_timeout（默认 5 秒），用于清理；该时间界限以事件循环未被同步代码阻塞为前提。同步 run 放弃残留任务并关闭循环时，asyncio 可能打印 “Task was destroyed but it is pending!”；这表示任务没有完成退出，与 cleanup_warnings 一致，不表示已强制清理成功。

11 月 15 日回归共 116 项通过、1 项可选 NumPy 测试跳过；真实 NumPy 用例另在临时依赖环境验证。新增用例覆盖五行数据中途停止、运行中快照、各运行终态、50 行不合规数据仅产生 50 条无 traceback 的 WARNING、字段映射与泛型类型诊断、空数据集覆盖率。Ruff 与 Pyright 通过。

### 2026-11-19 执行与输入回归

调度器在作业异常退出时先封存该行的 `error / internal_error`，记录异常类名，再向调用者传播原异常。运行终态在其余作业处理完后发布。清理宽限耗尽后，残留工作只持有自己的行和绑定，调用者释放已关闭的外部循环后，循环和本次运行均可回收。

新增用例覆盖并发指标错峰到期、工作任务异常后的行终态、关闭外部循环后的回收、严格校验返回值在两种插件入口的一致性，以及插件缺少评分方法时的构造错误。Windows / Python 3.12 在将警告视为错误的配置下通过 122 项测试，另有 1 项可选 NumPy 测试在带 NumPy 的环境中单独通过。Ruff 检查、格式检查和 Pyright 均通过。
