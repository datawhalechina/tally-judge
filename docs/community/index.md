# The Datawhale community

Datawhale is an open-source learning community focused on data science and artificial intelligence. Its members write tutorials, organize learning activities, and develop and maintain open-source projects together. [About Datawhale](https://github.com/datawhalechina)

## Why TallyJudge?

TallyJudge builds on Ragas, developed by Vibrant Labs and the Ragas community. Datawhale maintains the project in an independent repository under the Apache License 2.0, preserving the upstream commit history and contributor attribution. [Upstream Ragas](https://github.com/vibrantlabsai/ragas) · [Project license](https://github.com/datawhalechina/tally-judge/blob/main/LICENSE)

The upstream project already uses the distribution name `ragas` on PyPI. We chose **TallyJudge** to avoid a naming conflict when we publish independently and to make it clear who maintains this project. Local builds and Python imports still use `ragas` during the transition. [Upstream PyPI package](https://pypi.org/project/ragas/)

## Current work

We are reviewing the inherited code and dependencies, with existing RAG evaluations as our first priority. We are starting by reproducing upstream issues, reviewing related pull requests, and adding regression tests for compatibility fixes. Later work will extend evaluation for prompts, AI workflows, and agents. [TallyJudge repository](https://github.com/datawhalechina/tally-judge)
