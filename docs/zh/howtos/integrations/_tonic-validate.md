# Tonic Validate
## [Tonic Validate](https://tonic.ai/validate)，可视化 Ragas 得分 { #tonic-validate-visualize-ragas-scores }

<center><img src="https://uploads-ssl.webflow.com/62e28cf08913e81176ba2c39/65e77bcde4a7dbf5d853d319_tonic_validate_ragas_screenshot.png" alt="Tonic Validate Screenshot with list of projects and example graphs" width="600"/></center>

Validate 可以可视化并持续跟踪 Ragas 得分，帮助理解 RAG 或 LLM 应用表现。已使用 Ragas 时，只需在 Python 项目中添加两行代码即可接入。

## 开始使用 { #getting-started }

先创建[免费 Validate 账户](https://validate.tonic.ai/signup)。登录后创建项目，通常一个项目对应一个待评估应用。命名后进入项目主页。

安装用于上传得分的 tonic-ragas-logger。

```bash
pip install tonic-ragas-logger
```

在现有项目运行 Ragas 的位置加入以下两行代码，将 `evaluate()` 生成的 `scores` 发送到 Tonic Validate。API 密钥和项目 ID 可在项目主页获取。

```python
validate_api = RagasValidateApi("<Validate API Key>")
validate_api.upload_results("<Project ID>", scores)
```

发送得分后，Validate 会生成图表和 run。每个 run 包含一次 `evaluate()` 调用的得分，可以跟踪平均分变化，也可以查看单次运行中各问题的表现。
<br/>
<br/>

<center><img src="https://uploads-ssl.webflow.com/62e28cf08913e81176ba2c39/65e77bcd0ce60786fccba1b0_tonic_validate_ragas_gif.gif
" width="900"/></center>



## 联系我们 👋 { #reaching-out }
如有问题或界面反馈，可在开源评估框架 [tonic-validate 仓库](https://github.com/tonicai/tonic_validate)提交 GitHub issue。


