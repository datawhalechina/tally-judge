# 可用指标列表 { #list-of-available-metrics }

Ragas 提供一套评估指标，帮助你客观衡量 LLM 应用的表现，涵盖 RAG、智能体工作流等多种应用与任务。

每项指标都采用特定的评估方法，衡量应用的某个维度。基于 LLM 的指标可能需要一次或多次模型调用才能得到分数。你也可以使用 Ragas 修改现有指标，或编写自己的指标。

## 检索增强生成 { #retrieval-augmented-generation }
- [上下文精确率](context_precision.md)
- [上下文召回率](context_recall.md)
- [上下文实体召回率](context_entities_recall.md)
- [噪声敏感度](noise_sensitivity.md)
- [回答相关性](answer_relevance.md)
- [忠实度](faithfulness.md)
- [多模态忠实度](multi_modal_faithfulness.md)
- [多模态相关性](multi_modal_relevance.md)

## NVIDIA 指标 { #nvidia-metrics }
- [答案准确性](nvidia_metrics.md#answer-accuracy)
- [上下文相关性](nvidia_metrics.md#context-relevance)
- [回答依据充分性](nvidia_metrics.md#response-groundedness)

## 智能体与工具使用场景 { #agents-or-tool-use-cases }

- [主题遵循度](agents.md#topic-adherence)
- [工具调用准确率](agents.md#tool-call-accuracy)
- [工具调用 F1](agents.md#tool-call-f1)
- [智能体目标达成准确率](agents.md#agent-goal-accuracy)

## 自然语言比较 { #natural-language-comparison }

- [事实正确性](factual_correctness.md)
- [语义相似度](semantic_similarity.md)
- [非 LLM 字符串相似度](traditional.md#non-llm-string-similarity)
- [BLEU 得分](traditional.md#bleu-score)
- [CHRF 得分](traditional.md#chrf-score)
- [ROUGE 得分](traditional.md#rouge-score)
- [字符串包含性](traditional.md#string-presence)
- [精确匹配](traditional.md#exact-match)


## SQL

- [基于执行结果的 Datacompy 得分](sql.md#execution-based-metrics)
- [SQL 查询语义等价性](sql.md#sql-semantic-equivalence)

## 通用指标 { #general-purpose }

- [维度评判](general_purpose.md#aspect-critic)
- [简单标准评分](general_purpose.md#simple-criteria-scoring)
- [基于评分细则的评分](general_purpose.md#rubrics-based-criteria-scoring)
- [样本专属评分细则](general_purpose.md#instance-specific-rubrics-criteria-scoring)

## 其他任务 { #other-tasks }

- [摘要评估](summarization_score.md)
