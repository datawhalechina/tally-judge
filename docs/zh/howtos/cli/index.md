# Ragas CLI

Ragas 命令行接口（CLI）提供工具，让你从终端快速创建评估项目并运行实验。

## 安装 { #installation }

CLI 随 ragas 包一同安装。

```sh
pip install ragas
```

也可以使用 `uvx`，无须预先安装。

```sh
uvx ragas --help
```

## 可用命令 { #available-commands }

### `ragas quickstart`

从模板创建完整评估项目，是快速上手 Ragas 的便捷方式。

```sh
ragas quickstart [TEMPLATE] [OPTIONS]
```

**参数**

- `TEMPLATE`，模板名称，可选。留空时显示可用模板。

**选项**

- `-o, --output-dir`，项目创建目录，默认为当前目录

**示例**

```sh
# List available templates
ragas quickstart

# Create a RAG evaluation project
ragas quickstart rag_eval

# Create project in a specific directory
ragas quickstart rag_eval --output-dir ./my-project
```

### `ragas evals`

使用评估文件对数据集运行评估。

```sh
ragas evals EVAL_FILE [OPTIONS]
```

**参数**

- `EVAL_FILE`，评估文件路径，必填

**选项**

- `--dataset`，项目中的数据集名称，必填
- `--metrics`，要评估的指标字段名，以逗号分隔，必填
- `--baseline`，用于比较的基线实验名称，可选
- `--name`，本次实验的名称，可选

**示例**

```sh
ragas evals evals.py --dataset test_data --metrics accuracy,relevance
```

### `ragas hello_world`

创建简单的 hello world 示例，验证安装是否成功。

```sh
ragas hello_world [DIRECTORY]
```

**参数**

- `DIRECTORY`，示例创建目录，默认为当前目录

## 快速入门模板 { #quickstart-templates }

### RAG 与检索 { #rag-retrieval }
- [RAG 评估（`rag_eval`）](rag_eval.md)，使用自定义指标评估 RAG 系统
- [改进 RAG（`improve_rag`）](improve_rag.md)，比较基础 RAG 与智能体 RAG

### 智能体评估 { #agent-evaluation }
- [智能体评估（`agent_evals`）](agent_evals.md)，评估数学问题求解智能体
- [LlamaIndex 智能体评估（`llamaIndex_agent_evals`）](llamaIndex_agent_evals.md)，通过工具调用指标评估 LlamaIndex 智能体

### 专门场景 { #specialized-use-cases }
- [文本转 SQL 评估（`text2sql`）](text2sql.md)，通过执行准确率评估文本转 SQL 系统
- [工作流评估（`workflow_eval`）](workflow_eval.md)，评估复杂 LLM 工作流
- [提示词评估（`prompt_evals`）](prompt_evals.md)，比较不同提示词变体

### LLM 测试 { #llm-testing }
- [裁判对齐（`judge_alignment`）](judge_alignment.md)，衡量 LLM 裁判与人工标准的一致性
- [LLM 基准测试（`benchmark_llm`）](benchmark_llm.md)，测试并比较不同 LLM

## 快速开始 { #quick-start }

60 秒内开始运行。

```sh
# Create project
uvx ragas quickstart rag_eval
cd rag_eval

# Install dependencies
uv sync

# Set API key
export OPENAI_API_KEY="your-key"

# Run evaluation
uv run python evals.py
```

## 下一步 { #next-steps }

- [RAG 评估指南](rag_eval.md)，详细介绍 rag_eval 模板
- [改进 RAG 指南](improve_rag.md)，比较基础 RAG 与智能体 RAG
- [自定义指标](../../concepts/metrics/overview/index.md)，创建自己的评估指标
