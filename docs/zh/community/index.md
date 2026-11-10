# Datawhale 社区

Datawhale 是一个关注数据科学与人工智能的开源学习社区。社区成员共同编写教程、组织学习活动，也参与开源项目的开发与维护。[了解 Datawhale](https://github.com/datawhalechina)

## 我们为什么叫 TallyJudge

TallyJudge 源自 Vibrant Labs 和 Ragas 社区开发的 Ragas。我们保留了上游的提交历史和贡献者署名，沿用 Apache License 2.0，由 Datawhale 社区在独立仓库中继续维护。[上游 Ragas](https://github.com/vibrantlabsai/ragas) · [项目许可证](https://github.com/datawhalechina/tally-judge/blob/main/LICENSE)

PyPI 上的 `ragas` 分发名已由上游项目使用。为了避免未来独立发包时发生名称冲突，也让使用者能辨认项目的维护方，我们将项目命名为 **TallyJudge**。过渡期的本地构建和 Python 导入仍沿用 `ragas`。[上游 PyPI 包](https://pypi.org/project/ragas/)

## 当前进度

我们正在梳理继承的代码和依赖，优先维护现有的 RAG 评估能力。第一步是复现上游积压的 Issue，核对相关 PR，修复兼容问题，并为受影响的行为补上回归测试。后续计划逐步完善提示词、AI 工作流和 Agent 的评估。[TallyJudge 仓库](https://github.com/datawhalechina/tally-judge)
