# API 参考手册 { #api-references }

本节详细介绍 Ragas 所有核心组件的 API，按以下主题组织。

## 核心组件 { #core-components }

- [提示词](prompt.md)——提示词管理与模板
- [大语言模型](llms.md)——语言模型接口与配置
- [嵌入模型](embeddings.md)——嵌入模型接口与工具
- [分词器](tokenizers.md)——文本切分所用的分词器接口
- [RunConfig](run_config.md)——评估运行时配置
- [Executor](executor.md)——评估执行引擎
- [缓存](cache.md)——LLM 调用的缓存机制

## 评估 { #evaluation }

- [数据结构](evaluation_schema.md)——评估所用的数据结构
- [指标](metrics.md)——可用的评估指标及其实现
- [evaluate()](evaluate.md)——主要评估函数 API

## 测试集生成 { #testset-generation }

- [数据结构](testset_schema.md)——测试数据的数据结构
- [图](graph.md)——知识图谱的创建与管理
- [变换](transforms.md)——数据变换工具
- [合成器](synthesizers.md)——测试数据生成组件
- [生成](generate.md)——测试数据生成 API

## 集成 { #integrations }

- [集成](integrations.md)——外部工具集成 API
