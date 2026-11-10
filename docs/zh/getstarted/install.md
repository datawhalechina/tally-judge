# 安装 { #installation }

使用 `pip` 安装 Ragas，即可开始。

```bash
pip install ragas
```

如果希望体验最新功能，可以从主分支安装最新版本。

```bash
pip install git+https://github.com/vibrantlabsai/ragas.git
```

如果计划贡献代码或修改源码，请克隆仓库，并以[可编辑模式安装](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)。

```bash
git clone https://github.com/vibrantlabsai/ragas.git
pip install -e .
```

!!! note "LangChain OpenAI 依赖版本"
    如果使用 `langchain_openai`（例如 `ChatOpenAI`），请显式安装 `langchain-core` 和 `langchain-openai`，以避免版本不匹配。你可以根据环境调整版本范围，同时显式安装这两个包有助于避免严格版本约束造成的依赖冲突。
    ```bash
    pip install -U "langchain-core>=0.2,<0.3" "langchain-openai>=0.1,<0.2" openai
    ```
