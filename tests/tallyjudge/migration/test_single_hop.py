"""Projected SingleTurnSample fields of Ragas ec00339 with a scripted single-hop response.

5A selects stable chunk IDs without the archived graph and persona transforms.
The question, reference and original passage retain their archived meanings.
"""

import json
from types import SimpleNamespace

import pytest

from tallyjudge import create_synthesizer


class PassageJudge:
    plugin_id = "model.test.passage"
    usage_tracking = "untracked"

    def __init__(self):
        self.settings = {"model": "scripted"}
        self.suffix = ""

    def config_snapshot(self):
        return self.settings

    async def agenerate(self, prompt, response_model):
        question, reference = {
            "图书馆周一闭馆。": ("图书馆哪天闭馆？", "周一"),
            "图书借期为14天。": ("图书借期多长？", "14天"),
        }[
            json.loads(
                prompt.messages[0]
                .content.rsplit("input: ", 1)[1]
                .removesuffix("\nOutput: ")
            )["context"]
        ]
        return SimpleNamespace(
            data=response_model(query=question + self.suffix, answer=reference)
        )


@pytest.mark.asyncio
async def test_archived_output_fields_stable_ids_and_changed_manifest():
    judge = PassageJudge()
    key = "synthesis.testset.single_hop"
    generator = create_synthesizer(key, llm=judge, seed=7)
    sources = [
        {"chunk_id": "hours", "document_id": "library", "text": "图书馆周一闭馆。"},
        {"chunk_id": "loans", "document_id": "library", "text": "图书借期为14天。"},
    ]
    first = await generator.arun(sources, testset_size=2)
    expected = [
        ("图书馆哪天闭馆？", "周一", ["图书馆周一闭馆。"]),
        ("图书借期多长？", "14天", ["图书借期为14天。"]),
    ]
    assert [
        (row["user_input"], row["reference"], row["reference_contexts"])
        for row in first.to_records()
    ] == expected
    assert first.usage.untracked_sources == [judge.plugin_id]
    assert not first.usage.generation.complete
    judge.suffix = " (reworded)"
    second = await generator.arun(reversed(sources), testset_size=2)
    assert [c.case_id for c in first.candidates] == [
        c.case_id for c in second.candidates
    ]
    assert first.to_dataset().manifest != second.to_dataset().manifest
    judge.settings["model"] = "second-choice"
    third = await generator.arun(sources, testset_size=2)
    assert first.config_snapshot["llm"]["config"]["model"] == "scripted"
    assert {c.case_id for c in first.candidates}.isdisjoint(
        c.case_id for c in third.candidates
    )
