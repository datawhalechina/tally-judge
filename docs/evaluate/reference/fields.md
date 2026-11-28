---
title: 样本字段约定
description: 查阅内置指标使用的样本字段，并把数据集的列映射为指标输入。
---

内置指标按下面的字段名从样本里取值。这些名字是约定，不是内核的一部分：`tallyjudge.sdk` 和 `tallyjudge.engine` 的代码里不出现它们，内核只按每个指标声明的输入做校验。

数据集里的列名不同时，不用改数据，用 `MetricBinding(metric, input_map={"response": "answer"})` 把指标的字段映射到你的列；评测脚本里写成 `input_map = { response = "answer" }`。

这张表由 `python tools/gen_fields_doc.py` 生成。

| 字段 | 含义 | 使用它的内置指标 |
|---|---|---|
| `user_input` | 用户的问题或任务 | `custom.categorical`、`custom.instance_rubric`、`custom.numeric`、`custom.rubric_with_reference`、`custom.rubric_without_reference`、`rag.answer_accuracy`、`rag.answer_correctness`、`rag.answer_relevance`、`rag.context_precision_with_reference`、`rag.context_precision_without_reference`、`rag.context_recall`、`rag.context_relevance`、`rag.faithfulness`、`rag.noise_sensitivity`、`workflow.goal_completion` |
| `response` | 被测应用给出的回答 | `custom.categorical`、`custom.instance_rubric`、`custom.numeric`、`custom.rubric_with_reference`、`custom.rubric_without_reference`、`rag.answer_accuracy`、`rag.answer_correctness`、`rag.answer_relevance`、`rag.context_precision_without_reference`、`rag.factual_correctness`、`rag.faithfulness`、`rag.noise_sensitivity`、`rag.response_groundedness`、`text.exact_match`、`text.semantic_similarity`、`text.string_presence`、`text.string_similarity` |
| `reference` | 参考答案 | `custom.categorical`、`custom.instance_rubric`、`custom.numeric`、`custom.rubric_with_reference`、`custom.rubric_without_reference`、`rag.answer_accuracy`、`rag.answer_correctness`、`rag.context_entity_recall`、`rag.context_precision_with_reference`、`rag.context_recall`、`rag.factual_correctness`、`rag.noise_sensitivity`、`text.exact_match`、`text.semantic_similarity`、`text.string_presence`、`text.string_similarity` |
| `retrieved_contexts` | 检索器实际返回的段落，按原顺序 | `custom.categorical`、`custom.instance_rubric`、`custom.numeric`、`custom.rubric_with_reference`、`custom.rubric_without_reference`、`rag.context_entity_recall`、`rag.context_precision_with_reference`、`rag.context_precision_without_reference`、`rag.context_recall`、`rag.context_relevance`、`rag.faithfulness`、`rag.noise_sensitivity`、`rag.response_groundedness`、`retrieval.text_context_precision`、`retrieval.text_context_recall` |
| `reference_contexts` | 标准证据段落 | `custom.instance_rubric`、`custom.rubric_with_reference`、`custom.rubric_without_reference`、`retrieval.text_context_precision`、`retrieval.text_context_recall` |
| `retrieved_context_ids` | 检索到的文档 ID，按原顺序 | `retrieval.id_context_precision`、`retrieval.id_context_recall` |
| `reference_context_ids` | 标准证据的文档 ID | `retrieval.id_context_precision`、`retrieval.id_context_recall` |
| `messages` | 多轮对话：按顺序排列的消息，每条有 `role`（`system`、`user`、`assistant` 或 `tool`）和 `content` | `conversation.criteria` |
| `trace` | 一次运行的轨迹（步骤列表） | `workflow.forbidden_tools`、`workflow.goal_completion`、`workflow.step_efficiency`、`workflow.tool_arguments`、`workflow.tool_precision`、`workflow.tool_recall`、`workflow.tool_sequence` |
| `expected_tools` | 期望调用的工具名 | `workflow.step_efficiency`、`workflow.tool_precision`、`workflow.tool_recall`、`workflow.tool_sequence` |
| `expected_calls` | 期望的工具调用：工具名加参数 | `workflow.tool_arguments` |
| `forbidden_tools` | 不允许使用的工具名 | `workflow.forbidden_tools` |
| `final_state` | 应用运行之后重新读出的最终状态 | `workflow.final_state` |
| `expected_state` | 期望的最终状态，只写需要核对的部分 | `workflow.final_state` |
| `rubrics` | 这条样本自己的五档评分说明 | `custom.instance_rubric` |

`case_id` 是样本的标识，不是指标的输入。要对照两次运行或做配对比较，每行都要提供它。

`compare_experiments` 默认把 `response`、`retrieved_contexts`、`retrieved_context_ids` 当作被测应用的输出，允许它们在两次实验之间不同；其余字段必须相同才会给出分差。你的应用输出别的字段时，用 `output_fields` 指明。

问题和对照字段始终取数据集里的值：`user_input`、`reference`，以及以 `reference_`、`expected_`、`forbidden_` 开头的字段。被测应用即使返回了同名字段，评分时也不会采用；通过列名映射指向这些列时同样如此。
