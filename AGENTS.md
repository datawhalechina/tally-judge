# TallyJudge 开发指引

[架构与编码总纲](plan/implementation-standards.zh-CN.md) 适用于根目录 `tallyjudge/` 和新包测试。阶段计划规定交付范围。修改代码前按本次变更读取相关条款。

- 选择入口、拆分模块或增加插件时，按实际调用者确定职责和依赖方向。[模块职责与依赖](plan/implementation-standards.zh-CN.md#模块职责与依赖)
- 增加校验、默认值、异常处理或运行状态时，指出它对应的用户行为和负责边界。[日常设计判断](plan/implementation-standards.zh-CN.md#日常设计判断)
- 新写或修改方法时，按算法、状态归属和资源生命周期决定拆分。修改公开行为时同步维护 docstring 与行为测试。[函数与类的拆分](plan/implementation-standards.zh-CN.md#函数与类的拆分) · [注释与文档字符串](plan/implementation-standards.zh-CN.md#注释与文档字符串)
- 审查改动时，从公开入口证明调用结果，再检查导入、基线和状态语义。[评审与验证](plan/implementation-standards.zh-CN.md#评审与验证)

## 工作区与范围

- 新实现：根目录 `tallyjudge/`，Python >=3.12。
- 新测试：`tests/tallyjudge/`。旧 `src/ragas/`、旧测试和 `examples/` 原位保留供迁移对照，除非用户明确要求修改。
- 依赖集中在根 `pyproject.toml`；不新增零散依赖 txt。默认环境保持轻量。
- 设计在 `plan/`，临时记录在 `_experiments/`，不提交构建制品、临时环境、日志或密钥。
- 保留迁移算法的来源、作者与许可证信息；注明改名、接口重构与评分行为变化。
- 不默认沿用 Makefile 或旧 pre-commit 中的 Ragas monorepo 命令；当前新包以以下命令为准。

## 检查命令

```bash
uv run pytest
uv run ruff check tallyjudge tests/tallyjudge
uv run ruff format --check tallyjudge tests/tallyjudge
uv run pyright
```

仅文档修改使用链接、示例和 diff 检查；其他修改按影响选择测试，达到验收标准后不无故重复测试。包范围或安装行为改变时才补相应构建与仓库外安装验证。

## 文档与交付

- 参考 [阶段 1](plan/phase-1.zh-CN.md)、[阶段 2](plan/phase-2.zh-CN.md) 与[总设计](plan/plugin-architecture.zh-CN.md)，不把目标接口写成已实现能力。
- MkDocs 文档按教程、操作指南、参考与解释组织；调整页面路径时检查导航和链接。
- 代码使用项目现有 Ruff/Pyright 配置，不新增一套风格工具；命名遵循完整规范。
- 只报告实际完成和验证过的事项。提交、推送、tag、发布、对外发消息和付费请求不从普通实现任务中推定授权。
