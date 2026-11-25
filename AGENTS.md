# TallyJudge 开发指引

本指引适用于根目录 `tallyjudge/` 和新包测试。维护者工作区如有本地 `plan/`，修改代码前按本次变更查阅其中的 `implementation-standards.zh-CN.md` 和相关实现说明。

- 选择入口、拆分模块或增加插件时，按实际调用者确定职责和依赖方向。
- 增加校验、默认值、异常处理或运行状态时，指出它对应的用户行为和负责边界。
- 新写或修改方法时，按算法、状态归属和资源生命周期决定拆分。修改公开行为时同步维护 docstring 与行为测试。
- 审查改动时，从公开入口证明调用结果，再检查导入、基线和状态语义。

## 工作区与范围

- 新实现：根目录 `tallyjudge/`，Python >=3.12。
- 新测试：`tests/tallyjudge/`。原实现、旧测试和示例在 `legacy-ragas` 分支，迁移基线为 `ec00339`。
- 依赖集中在根 `pyproject.toml`；不新增零散依赖 txt。默认环境保持轻量。
- 设计在 `plan/`，临时记录在 `_experiments/`，不提交构建制品、临时环境、日志或密钥。
- 保留迁移算法的来源、作者与许可证信息；注明改名、接口重构与评分行为变化。
- 新包的检查命令见下节。

## 检查命令

```bash
uv run pytest
uv run ruff check tallyjudge tests/tallyjudge
uv run ruff format --check tallyjudge tests/tallyjudge
uv run pyright
```

仅文档修改使用链接、示例和 diff 检查；其他修改按影响选择测试，达到验收标准后不无故重复测试。包范围或安装行为改变时才补相应构建与仓库外安装验证。

## 文档与交付

- 功能说明以公开入口和行为测试为据。迁移行为对照 `tests/tallyjudge/migration/` 中的回放用例。
- 公开用法有变更时更新 README，能力或模块边界有变更时更新相应实现说明。旧 MkDocs 网站及其构建配置保存在归档分支。
- 代码使用项目现有 Ruff/Pyright 配置，不新增一套风格工具；命名遵循完整规范。
- 只报告实际完成和验证过的事项。提交、推送、tag、发布、对外发消息和付费请求不从普通实现任务中推定授权。
