# 改进 RAG 快速入门 { #improve-rag-quickstart }

`improve_rag` 模板使用真实评估数据比较不同 RAG 方案，包含基础模式（单次检索）与智能体模式（多步检索）。

## 创建项目 { #create-the-project }

```sh
# Using uvx (no installation required)
uvx ragas quickstart improve_rag
cd improve_rag

# Or with ragas installed
ragas quickstart improve_rag
cd improve_rag
```

## 安装依赖 { #install-dependencies }

```sh
uv sync
```

也可以使用 pip。

```sh
pip install -e .
```

## 设置 API 密钥 { #set-your-api-key }

```sh
export OPENAI_API_KEY="your-openai-key"
```

## 运行评估 { #run-the-evaluation }

### 基础 RAG 模式（默认） { #naive-rag-mode-default }

```sh
uv run python evals.py
```

### 智能体 RAG 模式 { #agentic-rag-mode }

```sh
uv run python evals.py --agentic
```

!!! note "智能体模式依赖"
    智能体模式需要 `openai-agents` 包，安装方法如下。
    ```sh
    pip install openai-agents
    ```

## 可选，使用 MLflow 追踪 { #optional-mlflow-tracing }

要详细跟踪 LLM 调用，请先启动 MLflow。

```sh
mlflow ui --port 5000
```

然后运行评估。如果 MLflow 服务已启动，追踪记录会自动发送到服务端。

## 项目结构 { #project-structure }

```
improve_rag/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── rag.py                 # RAG implementation (naive & agentic)
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets (hf_doc_qa_eval.csv)
    ├── experiments/       # Evaluation results
    └── logs/              # Evaluation logs
```

## 理解两种 RAG 模式 { #understanding-the-rag-modes }

### 基础 RAG { #naive-rag }

基础方案只进行一次检索。

1. **查询** → BM25 检索前 k 个文档
2. **上下文** → 使用检索文档构成上下文
3. **生成** → LLM 根据上下文生成回答

```python
rag = RAG(llm_client=client, retriever=retriever, mode="naive")
result = await rag.query("What is the Diffusers library?")
```

**优点**

- 简单且快速
- 延迟可预测
- 成本较低，只需一次 LLM 调用

**局限**

- 可能遗漏使用不同术语的相关文档
- 无查询细化过程
- 仅采用单一检索策略

### 智能体 RAG { #agentic-rag }

智能体方案让智能体控制检索过程。

1. **查询** → 智能体分析问题
2. **搜索** → 智能体决定搜索内容，可进行多次搜索
3. **细化** → 智能体根据结果调整搜索
4. **生成** → 智能体综合形成最终答案

```python
rag = RAG(llm_client=client, retriever=retriever, mode="agentic")
result = await rag.query("What command uploads an ESPnet model?")
```

**优点**

- 可以尝试多种搜索策略
- 更善于寻找具体技术信息
- 根据初次结果调整搜索

**局限**

- 延迟较高，需要多次 LLM 调用
- 成本较高
- 行为较难预测

## 评估数据集 { #the-evaluation-dataset }

模板包含 `hf_doc_qa_eval.csv`，收录有关 HuggingFace 文档的问题。

| 字段 | 描述 |
|-------|-------------|
| `question` | 有关 HuggingFace 工具的技术问题 |
| `expected_answer` | 标准答案 |

示例问题如下。

- “情感分析流水线默认使用哪个模型检查点？”
- “上传 ESPnet 模型使用什么命令？”
- “Diffusers 库的用途是什么？”

## 理解代码 { #understanding-the-code }

### RAG 实现（`rag.py`） { #the-rag-implementation-ragpy }

#### BM25Retriever

使用 BM25（Best Matching 25）算法检索文档。

```python
class BM25Retriever:
    def __init__(self, dataset_name="m-ric/huggingface_doc"):
        # Loads HuggingFace documentation
        # Splits into chunks for better retrieval
        # Creates BM25 index

    def retrieve(self, query: str, top_k: int = 3):
        # Returns top-k most relevant documents
```

#### RAG 类 { #rag-class }

两种模式采用统一接口。

```python
class RAG:
    def __init__(self, llm_client, retriever, mode="naive"):
        self.mode = mode
        if mode == "agentic":
            self._setup_agent()

    async def query(self, question: str, top_k: int = 3):
        if self.mode == "naive":
            return await self._naive_query(question, top_k)
        else:
            return await self._agentic_query(question, top_k)
```

### 评估脚本（`evals.py`） { #the-evaluation-script-evalspy }

正确性指标将模型回答与预期答案比较。

```python
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer...
    Return 'pass' if correct, 'fail' if incorrect.""",
    allowed_values=["pass", "fail"],
)
```

## 自定义 { #customization }

### 更换知识库 { #change-the-knowledge-base }

用自己的文档替换 HuggingFace 文档。

```python
class CustomRetriever:
    def __init__(self, documents: list[str]):
        from langchain_community.retrievers import BM25Retriever
        self.retriever = BM25Retriever.from_texts(documents)

    def retrieve(self, query: str, top_k: int = 3):
        self.retriever.k = top_k
        return self.retriever.invoke(query)
```

### 使用其他模型 { #use-a-different-model }

在 `evals.py` 中更改模型。

```python
# Use GPT-4 for better accuracy
rag = RAG(llm_client=client, retriever=retriever, model="gpt-4o")

# Or use a different provider
from anthropic import Anthropic
client = Anthropic()
# Note: Would need to modify rag.py for non-OpenAI clients
```

### 添加自定义指标 { #add-custom-metrics }

评估其他维度。

```python
from ragas.metrics import NumericalMetric

completeness = NumericalMetric(
    name="completeness",
    prompt="""How complete is the response (1-5)?
    Question: {question}
    Expected: {expected_answer}
    Response: {response}
    Score:""",
    allowed_values=(1, 5),
)

# Add to experiment
result = {
    **row,
    "correctness": correctness_score.value,
    "completeness": completeness.score(...).value,
}
```

### 修改智能体行为 { #modify-the-agent-behavior }

在 `rag.py` 中定制智能体搜索策略。

```python
def _setup_agent(self):
    @function_tool
    def retrieve(query: str) -> str:
        """Custom tool description..."""
        docs = self.retriever.retrieve(query, self.default_k)
        return "\n\n".join([doc.page_content for doc in docs])

    self._agent = Agent(
        name="Custom RAG Assistant",
        instructions="Your custom instructions...",
        tools=[retrieve]
    )
```

## 比较结果 { #comparing-results }

分别运行两种模式并进行比较。

```sh
# Run naive mode
uv run python evals.py
# Results saved to experiments/YYYYMMDD-HHMMSS_naiverag.csv

# Run agentic mode
uv run python evals.py --agentic
# Results saved to experiments/YYYYMMDD-HHMMSS_agenticrag.csv
```

分析结果。

```python
import pandas as pd

naive = pd.read_csv("evals/experiments/..._naiverag.csv")
agentic = pd.read_csv("evals/experiments/..._agenticrag.csv")

print(f"Naive pass rate: {(naive['correctness_score'] == 'pass').mean():.1%}")
print(f"Agentic pass rate: {(agentic['correctness_score'] == 'pass').mean():.1%}")
```

## 故障排查 { #troubleshooting }

### MLflow 警告 { #mlflow-warnings }

如果出现追踪失败的 MLflow 警告，可以选择以下方式。

1. 启动 MLflow，运行 `mlflow ui --port 5000`。
2. 忽略警告，关闭追踪并不影响评估运行。

### 智能体模式无法运行 { #agentic-mode-not-working }

确保已安装 agents 包。

```sh
pip install openai-agents
```

### 首次运行较慢 { #slow-first-run }

首次运行需要下载约 300 MB 的 HuggingFace 文档数据集，后续会使用缓存。

## 下一步 { #next-steps }

- [RAG 评估指南](rag_eval.md)，更简单的评估配置
- [自定义指标](../../concepts/metrics/overview/index.md)，编写自己的指标
- [评估和改进 RAG](../applications/evaluate-and-improve-rag.md)，生产环境 RAG 评估
