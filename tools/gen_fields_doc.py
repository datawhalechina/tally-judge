import collections
from pathlib import Path

from tallyjudge.registry import registry

TARGET = Path(__file__).resolve().parents[1] / "docs" / "fields.zh-CN.md"
SETTINGS = {
    "metric.custom.numeric": {"criteria": "x"},
    "metric.custom.categorical": {"criteria": "x", "labels": ["a", "b"]},
    "metric.conversation.criteria": {"criteria": "x"},
}
MEANING = {
    "user_input": "用户的问题或任务",
    "response": "被测应用给出的回答",
    "reference": "参考答案",
    "retrieved_contexts": "检索器实际返回的段落，按原顺序",
    "reference_contexts": "标准证据段落",
    "retrieved_context_ids": "检索到的文档 ID，按原顺序",
    "reference_context_ids": "标准证据的文档 ID",
    "messages": "多轮对话：按顺序排列的消息，每条有 `role`（`system`、`user`、`assistant` 或 `tool`）和 `content`",
    "trace": "一次运行的轨迹（步骤列表）",
    "expected_tools": "期望调用的工具名",
    "expected_calls": "期望的工具调用：工具名加参数",
    "forbidden_tools": "不允许使用的工具名",
    "final_state": "应用运行之后重新读出的最终状态",
    "expected_state": "期望的最终状态，只写需要核对的部分",
    "rubrics": "这条样本自己的五档评分说明",
}
HEAD = """# 样本字段约定

内置指标按下面的字段名从样本里取值。这些名字是约定，不是内核的一部分：`tallyjudge.sdk` 和 `tallyjudge.engine` 的代码里不出现它们，内核只按每个指标声明的输入做校验。

数据集里的列名不同时，不用改数据，用 `MetricBinding(metric, input_map={"response": "answer"})` 把指标的字段映射到你的列；评测脚本里写成 `input_map = { response = "answer" }`。

这张表由 `python tools/gen_fields_doc.py` 生成。

| 字段 | 含义 | 使用它的内置指标 |
|---|---|---|
"""
TAIL = """
`case_id` 是样本的标识，不是指标的输入。要对照两次运行或做配对比较，每行都要提供它。

`compare_experiments` 默认把 `response`、`retrieved_contexts`、`retrieved_context_ids` 当作被测应用的输出，允许它们在两次实验之间不同；其余字段必须相同才会给出分差。你的应用输出别的字段时，用 `output_fields` 指明。

问题和对照字段始终取数据集里的值：`user_input`、`reference`，以及以 `reference_`、`expected_`、`forbidden_` 开头的字段。被测应用即使返回了同名字段，评分时也不会采用；通过列名映射指向这些列时同样如此。
"""

if __name__ == "__main__":
    users = collections.defaultdict(list)
    for spec in registry.list(kind="metric"):
        described = registry.describe(spec.id, **SETTINGS.get(spec.id, {}))
        for name in described.inputs:
            users[name].append(f"`{spec.id.removeprefix('metric.')}`")
    rows = "".join(
        f"| `{name}` | {MEANING[name]} | {'、'.join(sorted(users[name]))} |\n"
        for name in MEANING
    )
    assert set(users) == set(MEANING), set(users) ^ set(MEANING)
    TARGET.write_text(HEAD + rows + TAIL, encoding="utf-8")
