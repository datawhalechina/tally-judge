# 参与 TallyJudge 开发

感谢你愿意改进 TallyJudge。本文说明如何搭建开发环境、运行检查，以及迁移 Ragas 功能时如何验证行为一致。

## 搭建开发环境

仓库使用 uv 管理 Python 和依赖。安装 uv 后，在本地用 Python 3.12 建立开发环境。[uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/datawhalechina/tally-judge.git
cd tally-judge
uv sync --python 3.12
```

uv 会在 `.venv` 中安装当前源码和开发工具。首次解析会生成 `uv.lock`，后续同步复用锁定的版本。锁文件也记录可选插件在不同 Python 版本和系统上的发行包。修改模型插件时，用 `uv sync --locked --group integration` 装入迁移对照所用的 SDK 和 Pydantic 版本。

## 提交前的检查

```bash
uv run pytest
uv run ruff check tallyjudge tests/tallyjudge
uv run ruff format --check tallyjudge tests/tallyjudge
uv run pyright
```

修改模型插件或迁移对照时，再运行 `uv run --group integration pytest tests/tallyjudge/integration tests/tallyjudge/migration`。改动打包范围或安装行为时，构建 wheel 并在仓库外验证：

```bash
uv build --out-dir <临时目录>
uv run python tests/tallyjudge/installation/test_installation.py --wheel-dir <临时目录>
```

用 `uv add` 或 `uv remove` 改动依赖时，uv 会同步更新 `pyproject.toml` 和 `uv.lock`。手工编辑依赖声明后运行 `uv lock`，用 `uv lock --check` 在本地确认锁文件与声明一致。

## 改动的写法

- 一次提交围绕一个可解释的行为或边界。改名和评分公式的变化分开提交。
- 遵守分层。`tallyjudge.sdk` 是插件可以导入的内部模块（合成、优化两类插件另从各自能力包的 `contract` 模块导入契约），它自己不导入包内其他模块；`tallyjudge.engine` 只导入 `sdk`；`tallyjudge.plugins` 只导入 `sdk`，其中优化器因为要驱动评测运行，可以另外导入 `engine`。`sdk` 和 `engine` 里不出现 `user_input` 这类领域字段名。这些规则由 `tests/tallyjudge/contracts/test_layering.py` 检查。
- 新增内置插件时，在 README 里写出它的插件 ID，`test_readme_examples.py` 会检查每个已注册的 ID 都出现过。
- 修改公开行为时，同步更新 docstring、行为测试和 README。
- 新增校验、默认值、异常处理或运行状态时，说明它对应的用户行为和负责的模块。
- 依赖集中写在根目录 `pyproject.toml`。默认安装保持轻量；插件需要的 SDK 放进可选依赖，并且不在模块顶层导入。
- 临时笔记放在 `_experiments/`，不要提交构建产物、虚拟环境、日志或密钥。

## 迁移原 Ragas 实现

原实现、旧测试、示例和文档网站保存在 [`legacy-ragas`](https://github.com/datawhalechina/tally-judge/tree/legacy-ragas) 分支，行为对照的基线提交是 `ec00339`。迁移一个能力时，在 PR 中写明：

| 项目 | 内容 |
|---|---|
| 来源 | 原文件与符号、基线提交或上游 PR |
| 目标 | 新文件与符号、稳定的插件 ID |
| 行为 | 保留的公式与参数；有意的差异及对应的评分 `revision` |
| 依赖 | 需要的第三方包，以及是否按需加载 |
| 验证 | 在旧环境取得的固定输入与输出、故障场景和测试结果 |

迁移的模块在文件开头保留来源和 Apache-2.0 许可说明，并注明改名、接口重构和评分行为的变化。整体来源说明见 [NOTICE](NOTICE)。

## 报告问题

普通缺陷和功能建议请提交 [Issue](https://github.com/datawhalechina/tally-judge/issues)，附上可复现的最小样本、期望结果和实际结果。安全问题请不要公开提交，使用仓库的 [私密漏洞报告](https://github.com/datawhalechina/tally-judge/security/advisories/new)。

提交代码即表示你同意按 [Apache License 2.0](LICENSE) 授权你的贡献。
