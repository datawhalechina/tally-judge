---
title: 内置指标速查
description: 按任务查找内置指标的插件 ID、样本输入、模型依赖、分数范围和方向。
---

每个小节的标题都是可传给 `create_metric` 的完整插件 ID。数值指标默认越高越好，噪声敏感度越低越好。类别指标返回标签。结果为 `undefined` 表示该样本无法按指标定义计算分数。

`llm` 指裁判语言模型，`embeddings` 指嵌入模型。标有 `text-distance` 的本地指标需要先安装 `uv add "tallyjudge[text-distance]"`。准备评测数据时，可先按手头材料选指标，再核对列名。[RAG 指标选择](/evaluate/rag/metrics) · [样本字段约定](/evaluate/reference/fields)

## 检索命中与排序

### metric.retrieval.id_context_precision

去重后，命中的 ID 数除以召回的 ID 数。整数和同值字符串视作同一 ID，次序不影响得分。召回 ID 为空时为 `undefined`。

**输入**　`retrieved_context_ids`、`reference_context_ids`。

**计算**　本地，无模型。**输出**　0～1。

### metric.retrieval.id_context_recall

去重后，命中的 ID 数除以参考 ID 数。例如召回 `["B", "A", "A"]`，参考 `["A", "C", "D"]`，精度为 1/2，召回率为 1/3。参考 ID 为空时为 `undefined`。

**输入**　`retrieved_context_ids`、`reference_context_ids`。

**计算**　本地，无模型。**输出**　0～1。

### metric.retrieval.text_context_precision

用字符串相似度匹配标准证据。某召回块与任一参考块的相似度达到 `threshold`（默认 0.5）即算命中。按召回顺序计算命中位置的平均精度，无命中得 0。

**输入**　`retrieved_contexts`、`reference_contexts`。

**计算**　本地 `text-distance`。**输出**　0～1。

### metric.retrieval.text_context_recall

逐个参考块查找召回块中相似度达到 `threshold`（默认 0.5）的候选，得分为找到匹配的参考块数除以参考块数。参考列表为空时为 `undefined`。

**输入**　`retrieved_contexts`、`reference_contexts`。

**计算**　本地 `text-distance`。**输出**　0～1。

### metric.rag.context_precision_with_reference

以问题和参考答案为准，裁判逐块判断哪些召回块有用。按原排名对所有命中位置的精度取平均。例如判定 `[1, 0, 1]` 得 5/6。所有块都未命中时为 `undefined`。

**输入**　`user_input`、`reference`、`retrieved_contexts`。

**计算**　`llm`。**输出**　0～1。

### metric.rag.context_precision_without_reference

以被测系统的实际回答为准，裁判逐块判断哪些召回块有助于给出这份回答，再按原排名计算平均精度。所有块都未命中时为 `undefined`。

**输入**　`user_input`、`response`、`retrieved_contexts`。

**计算**　`llm`。**输出**　0～1。

### metric.rag.context_recall

裁判同时读取所有召回块。参考答案的陈述中，可以归因于召回块的陈述占多大比例，就是这道题的召回得分。

**输入**　`user_input`、`reference`、`retrieved_contexts`。

**计算**　`llm`。**输出**　0～1。

### metric.rag.context_entity_recall

裁判分别从参考答案和召回块抽取实体，再按原字符串去重和匹配。分子是双方共有的参考实体数，分母为参考实体数加上 `1e-8`。没有参考实体时为 `undefined`。

**输入**　`reference`、`retrieved_contexts`。

**计算**　`llm`。**输出**　0～1。

### metric.rag.context_relevance

两份提示词分别判断整组召回块对问题的相关性，各按 0、1、2 三档给分，再对有效判定归一化取均值。

**输入**　`user_input`、`retrieved_contexts`。

**计算**　`llm`。**输出**　0～1。

## 回答质量与证据

### metric.rag.faithfulness

裁判把回答拆成可核查的陈述，逐条检查召回块是否提供依据。支持的陈述数除以陈述总数，得分明细可定位到具体陈述。

**输入**　`user_input`、`response`、`retrieved_contexts`。

**计算**　`llm`。**输出**　0～1。

### metric.rag.response_groundedness

两份提示词分别通读回答和召回块，按整段回答有多少依据给等级分，再对有效判定归一化取均值。适合整体判分，逐条定位可用 Faithfulness。

**输入**　`response`、`retrieved_contexts`。

**计算**　`llm`。**输出**　0～1。

### metric.rag.answer_relevance

裁判根据回答反推问题，嵌入模型计算这些问题与原问题的余弦相似度，再取平均。`strictness` 默认 3。没有生成问题或生成的问题全部被标为含糊作答时得 0。

**输入**　`user_input`、`response`。

**计算**　`llm`、`embeddings`。**输出**　−1～1。

### metric.rag.answer_accuracy

根据问题和参考答案给回答判分。两次判分交换回答与参考的展示顺序，各按 0、2、4 三档给分，再对有效判定归一化取均值。

**输入**　`user_input`、`response`、`reference`。

**计算**　`llm`。**输出**　0～1。

### metric.rag.factual_correctness

`mode="precision"` 检查回答里的陈述能否由参考答案支持，`mode="recall"` 检查参考答案里的陈述是否被回答覆盖。默认 `mode="f1"` 合并两个方向，`beta` 控制召回的权重。

**输入**　`response`、`reference`。

**计算**　`llm`。**输出**　0～1。

### metric.rag.answer_correctness

用事实判定和语义相似度的加权平均比较回答与参考答案。`weights` 默认 `[0.75, 0.25]`，权重按总和归一化。下界是归一化语义权重的负值。设为 `[1.0, 0.0]` 时只用裁判，范围为 0～1。

**输入**　`user_input`、`response`、`reference`。

**计算**　`llm`，语义权重大于 0 时还需 `embeddings`。**输出**　默认 −0.25～1。

### metric.rag.noise_sensitivity

在全部回答陈述中，计算参考答案不支持、却能归因于某类召回块的陈述占比。默认 `mode="relevant"` 统计被相关块支持的错误陈述，`mode="irrelevant"` 只统计由无关块独自支持的错误陈述。能支持任一参考陈述的块算相关块。

**输入**　`user_input`、`response`、`reference`、`retrieved_contexts`。

**计算**　`llm`。**输出**　0～1，越低越好。

## 文本匹配与相似度

### metric.text.exact_match

回答与参考文本逐字符相等得 1。大小写、标点和空白都参与比较。

**输入**　`reference`、`response`。

**计算**　本地，无模型。**输出**　0 或 1。

### metric.text.string_presence

参考文本作为连续子串出现在回答中得 1。比较区分大小写。

**输入**　`reference`、`response`。

**计算**　本地，无模型。**输出**　0 或 1。

### metric.text.string_similarity

以 1 减去归一化字符串距离，完全匹配得 1。`distance_measure` 支持 `levenshtein`（默认）、`hamming`、`jaro` 和 `jaro_winkler`。

**输入**　`reference`、`response`。

**计算**　本地 `text-distance`。**输出**　0～1。

### metric.text.semantic_similarity

计算参考答案与回答两个嵌入向量的余弦相似度。`threshold=None` 返回原始值。设置阈值后大于或等于该值即得 1，包括 `threshold=0.0`。零向量的余弦无定义。

**输入**　`reference`、`response`。

**计算**　`embeddings`。**输出**　−1～1，设置阈值时为 0 或 1。

## 自定义裁判与量表

### metric.custom.numeric

在 `criteria` 中写明判分标准，用 `value_range` 指定连续分值范围，或用 `levels` 列出允许的整数等级。离散等级可用 `strictness` 配置多次判定后投票。

**输入**　`response`。可选 `user_input`、`reference`、`retrieved_contexts`。

**计算**　`llm`。**输出**　默认 0～1。

### metric.custom.categorical

用 `criteria` 说明各类别的区分方式，用 `labels` 给出允许的标签，比如 `["已解决", "待跟进"]`。运行汇总统计各标签出现次数。

**输入**　`response`。可选 `user_input`、`reference`、`retrieved_contexts`。

**计算**　`llm`。**输出**　`labels` 中的一个字符串，无数值方向。

### metric.custom.rubric_without_reference

用同一份无参考量表判断所有样本，通常提供 `response`。可在创建指标时传入 `rubrics` 替换默认量表，结果同时保留等级和反馈。

**输入**　无必填字段。可选 `user_input`、`response`、`retrieved_contexts`、`reference_contexts`、`reference`。

**计算**　`llm`。**输出**　1～5 的整数等级。

### metric.custom.rubric_with_reference

用同一份有参考量表判断所有样本，通常同时提供 `response` 与 `reference`。可在创建指标时传入 `rubrics` 替换默认量表。

**输入**　无必填字段。可选 `user_input`、`response`、`retrieved_contexts`、`reference_contexts`、`reference`。

**计算**　`llm`。**输出**　1～5 的整数等级。

### metric.custom.instance_rubric

当每道题有自己的得分标准时，把非空的 `rubrics` 字典写入样本，并提供该题的回答和量表所需的材料。结果保留等级和反馈。

**输入**　`rubrics`。可选 `user_input`、`response`、`retrieved_contexts`、`reference_contexts`、`reference`。

**计算**　`llm`。**输出**　1～5 的整数等级。

## Agent 与工作流

### metric.workflow.tool_sequence

按开始顺序取出成功调用的工具名，与 `expected_tools` 逐项比较。名称、顺序和次数全部相同得 1。

**输入**　`trace`、`expected_tools`。

**计算**　本地，无模型。**输出**　0 或 1。

### metric.workflow.tool_recall

期望工具去重后，统计其中有多少种至少成功调用过一次。结果明细列出漏掉的工具，期望集合为空时为 `undefined`。

**输入**　`trace`、`expected_tools`。

**计算**　本地，无模型。**输出**　0～1。

### metric.workflow.tool_precision

成功的工具调用中，使用期望工具的次数占比。重复成功调用逐次计数，没有成功调用时为 `undefined`。

**输入**　`trace`、`expected_tools`。

**计算**　本地，无模型。**输出**　0～1。

### metric.workflow.tool_arguments

`expected_calls` 的每项含 `name` 和 `arguments`。工具名称与参数对象都相同才匹配，每次实际成功调用最多匹配一次。没有期望调用或未记录成功调用的参数时为 `undefined`。

**输入**　`trace`、`expected_calls`。

**计算**　本地，无模型。**输出**　0～1。

### metric.workflow.step_efficiency

期望工具列表长度除以工具尝试次数，上限为 1。失败、重试均计入分母，没有尝试时为 `undefined`。该分数只衡量步数。

**输入**　`trace`、`expected_tools`。

**计算**　本地，无模型。**输出**　0～1。

### metric.workflow.forbidden_tools

按尝试过的工具名检查 `forbidden_tools`。即使调用失败，只要尝试过禁用工具也会得到 `violated`。

**输入**　`trace`、`forbidden_tools`。

**计算**　本地，无模型。**输出**　字符串 `compliant` 或 `violated`，前者表示合规。

### metric.workflow.final_state

沿 `expected_state` 的嵌套路径核对 `final_state`，统计期望值匹配的比例。额外的状态字段不扣分，明细列出不匹配路径。空的期望状态为 `undefined`。

**输入**　`final_state`、`expected_state`。

**计算**　本地，无模型。**输出**　0～1。

### metric.workflow.goal_completion

裁判根据用户任务和成功的工具调用结果判断目标是否达成。最终回答中的自我陈述不能单独证明达成。没有记录可供核对的工具结果时为 `undefined`。

**输入**　`user_input`、`trace`。

**计算**　`llm`。**输出**　0 或 1。

## 多轮对话

### metric.conversation.criteria

裁判读取完整 `messages`，按创建指标时提供的 `criteria` 判断整段对话。满足标准得 1，结果同时给出理由。

**输入**　`messages`。

**计算**　`llm`。**输出**　0 或 1。
