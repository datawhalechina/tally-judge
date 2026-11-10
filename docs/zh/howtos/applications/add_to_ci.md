---
search:
  exclude: true
---

# 使用 Pytest 将评估加入 CI 流水线 { #adding-to-your-ci-pipeline-with-pytest }

可以将 Ragas 评估纳入持续集成流水线，
跟踪 RAG 系统的回答质量。将它作为端到端测试套件的一部分，
在重大改动和发布前运行。

使用时，将 `evaluate()` 函数的
`in_ci` 参数设置为 `True`。Ragas 会以特殊模式运行指标，
提高结果的可复现性，但也会增加成本。

可以按以下方式编写 Pytest 测试。

!!! note
    示例数据集已包含参考 RAG 系统的输出。
    测试自己的系统时，应使用被测 RAG 流水线的输出。
    关于如何构建数据集，请参阅
    [使用自己的数据构建 HF Dataset](../../concepts/components/eval_dataset.md)。

```python
import pytest
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


def test_amnesty_e2e():
    # loading the V2 dataset
    amnesty_qa = load_dataset("vibrantlabsai/amnesty_qa", "english_v2")["eval"]


    result = evaluate(
        amnesty_qa,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        in_ci=True,
    )
    assert result["answer_relevancy"] >= 0.9
    assert result["context_recall"] >= 0.95
    assert result["context_precision"] >= 0.95
    assert_in_range(result["faithfulness"], value=0.4, plus_or_minus=0.1)
```

## 使用 Pytest 标记管理 Ragas 端到端测试 { #using-pytest-markers-for-ragas-e2e-tests }

端到端测试耗时较长，可以使用 [Pytest 标记](https://docs.pytest.org/en/latest/example/markers.html)为测试添加标签，便于只在需要时运行 Ragas 测试。

在 `conftest.py` 中添加以下内容，注册 `ragas_ci` 标签。
```python
def pytest_configure(config):
    """
    configure pytest
    """
    # add `ragas_ci`
    config.addinivalue_line(
        "markers", "ragas_ci: Set of tests that will be run as part of Ragas CI"
    )
```

然后使用 `ragas_ci` 标记属于 Ragas CI 的测试。

```python
import pytest
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


@pytest.mark.ragas_ci
def test_amnesty_e2e():
    # loading the V2 dataset
    amnesty_qa = load_dataset("vibrantlabsai/amnesty_qa", "english_v2")["eval"]


    result = evaluate(
        amnesty_qa,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        in_ci=True,
    )
    assert result["answer_relevancy"] >= 0.9
    assert result["context_recall"] >= 0.95
    assert result["context_precision"] >= 0.95
    assert_in_range(result["faithfulness"], value=0.4, plus_or_minus=0.1)
```
