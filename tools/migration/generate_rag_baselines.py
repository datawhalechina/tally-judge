"""Generate deterministic RAG fixtures using the isolated ec00339 environment."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import warnings
from pathlib import Path

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._context_precision import IDBasedContextPrecision
from ragas.metrics._context_recall import IDBasedContextRecall
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.context_recall.metric import ContextRecall
from ragas.metrics.collections.context_recall.util import ContextRecallPrompt
from ragas.metrics.collections.context_precision.metric import ContextPrecisionWithReference, ContextPrecisionWithoutReference
from ragas.metrics.collections.context_precision.util import ContextPrecisionPrompt
from ragas.metrics.collections.answer_relevancy.metric import AnswerRelevancy
from ragas.metrics.collections.answer_relevancy.util import AnswerRelevancePrompt
from ragas.metrics.collections.faithfulness.metric import Faithfulness
from ragas.metrics.collections.faithfulness.util import StatementGeneratorPrompt
from ragas.prompt.metrics.base_prompt import BasePrompt

BASE = "ec003395bb37c223fc981e6567162513aec52a71"
QUESTION = "法国的首都是哪里？"
ANSWER = "巴黎。"

# Expected TallyJudge outcomes for cases where the upstream scorer raised or returned NaN.
INTENTIONAL_DIFFERENCES = {
    "no_classifications": {"status": "undefined", "reason": "no_classifications"},
    "no_relevant_contexts": {"status": "undefined", "reason": "no_relevant_contexts"},
    "empty_passages": {"status": "error", "reason": "validation_error"},
    "no_statements": {"status": "undefined", "reason": "no_statements"},
    "empty_verdicts": {"status": "undefined", "reason": "no_verdicts"},
    "zero_vector": {"status": "undefined", "reason": "zero_norm_embedding"},
    "invalid_verdict": {"status": "error", "reason": "schema_validation"},
    "string_context": {"status": "error", "reason": "validation_error"},
}


class Judge(InstructorBaseRagasLLM):
    def __init__(self, replies, calls, prompts):
        self.replies = iter(replies)
        self.calls = calls
        self.prompts = prompts

    def generate(self, *args, **kwargs):
        raise AssertionError("The migration must use the async interface")

    async def agenerate(self, prompt, response_model, **kwargs):
        key = hashlib.sha256(prompt.encode()).hexdigest()
        self.prompts[key] = prompt
        self.calls.append({"generate": key, "schema": response_model.__name__})
        return response_model.model_validate(next(self.replies))


class Embeddings(BaseRagasEmbedding):
    def __init__(self, vectors, calls):
        self.vectors = vectors
        self.calls = calls

    def embed_text(self, *args, **kwargs):
        raise AssertionError("The migration must use the async interface")

    async def aembed_text(self, text, **kwargs):
        return (await self.aembed_texts([text]))[0]

    async def aembed_texts(self, texts, **kwargs):
        self.calls.append({"embed": texts})
        return [self.vectors[text] for text in texts]


ID_METRICS = {"id_context_precision": IDBasedContextPrecision, "id_context_recall": IDBasedContextRecall}


CONTEXT_METRICS = {
    "context_recall": ContextRecall,
    "context_precision_with_reference": ContextPrecisionWithReference,
    "context_precision_without_reference": ContextPrecisionWithoutReference,
}


def cases():
    faith_inputs = dict(
        user_input=QUESTION,
        response=ANSWER,
        retrieved_contexts=["法国位于欧洲。", "首都是巴黎。"],
    )
    statements = {"statements": ["甲", "乙", "丙"]}
    for name, verdicts in [
        ("supported", [1, 1, 1]),
        ("mixed", [0, 1, 0]),
        ("mismatch", [1, 0]),
        ("empty_verdicts", []),
        ("coerced_verdicts", ["1", True, 1.0]),
        ("invalid_verdict", [2]),
    ]:
        replies = [
            statements,
            {
                "statements": [
                    {"statement": str(i), "reason": "fixture", "verdict": value}
                    for i, value in enumerate(verdicts)
                ]
            },
        ]
        yield "faithfulness", name, faith_inputs, replies, {}
    yield "faithfulness", "no_statements", faith_inputs, [{"statements": []}], {}
    yield (
        "faithfulness",
        "string_context",
        dict(faith_inputs, retrieved_contexts="巴黎。"),
        [],
        {},
    )
    vectors = {
        QUESTION: [1.0, 0.0],
        "A": [1.0, 0.0],
        "B": [-1.0, 0.0],
        "C": [0.0, 1.0],
        "Z": [0.0, 0.0],
    }
    for name, questions in [
        ("negative", [("B", 0)]),
        ("mixed", [("A", 1), ("", 0), ("C", 0)]),
        ("all_evasive", [("A", 2), ("B", -1), ("C", 1)]),
        ("empty", [("", 1)]),
        ("zero_vector", [("Z", 1)]),
    ]:
        yield (
            "answer_relevance",
            name,
            dict(user_input=QUESTION, response=ANSWER),
            [{"question": q, "noncommittal": n} for q, n in questions],
            vectors,
        )

    for name, retrieved, reference in (
        ("duplicate_rank", ["B", "A", "A"], ["A", "C", "D"]),
        ("reordered", ["A", "B", "A"], ["D", "A", "C", "C"]),
        ("mixed_ids", [1, "1", "01", 2], ["1", 2, "2"]),
        ("no_overlap", ["A"], ["B"]),
        ("empty_retrieval", [], ["A"]),
        ("empty_reference", ["A"], []),
        ("both_empty", [], []),
    ):
        for metric_id in ID_METRICS:
            yield metric_id, name, dict(retrieved_context_ids=retrieved, reference_context_ids=reference), [], {}

    for metric_id, field, answer in (
        ("context_precision_with_reference", "reference", "gold"),
        ("context_precision_without_reference", "response", "observed"),
    ):
        for name, passages, flags in (
            ("ranked", ["first", "second", "first"], [1, 0, 1]),
            ("reordered", ["second", "first", "first"], [0, 1, 1]),
            ("no_relevant_contexts", ["first", "second"], [0, 0]),
            ("empty_passages", [], []),
            ("empty_passage_text", [""], [1]),
        ):
            inputs = {"user_input": "question", field: answer, "retrieved_contexts": passages}
            yield metric_id, name, inputs, [{"reason": "fixture", "verdict": v} for v in flags], {}

    for name, passages, flags in (
        ("mixed_attribution", ["first", "second", "first"], [1, 0, 1]),
        ("unsupported", ["first", "second", "first"], [0, 0]),
        ("no_classifications", ["first", "second", "first"], []),
        ("empty_passages", [], []),
        ("empty_passage_text", [""], [1]),
    ):
        inputs = {"user_input": "question", "reference": "gold", "retrieved_contexts": passages}
        items = [{"statement": "same statement", "reason": "fixture", "attributed": flag} for flag in flags]
        yield "context_recall", name, inputs, [{"classifications": items}] if passages else [], {}


async def legacy_score(metric_id, inputs, replies, vectors, judge, calls):
    if metric_id in ID_METRICS:
        return await ID_METRICS[metric_id]().single_turn_ascore(SingleTurnSample(**inputs))
    if metric_id in CONTEXT_METRICS:
        metric = CONTEXT_METRICS[metric_id](llm=judge)
    elif metric_id == "faithfulness":
        metric = Faithfulness(llm=judge)
    else:
        metric = AnswerRelevancy(llm=judge, embeddings=Embeddings(vectors, calls), strictness=len(replies))
    return (await metric.ascore(**inputs)).value


async def generate(source_root):
    hashes = {}
    for cls in (
        SingleTurnSample,
        *ID_METRICS.values(),
        *CONTEXT_METRICS.values(),
        ContextPrecisionPrompt,
        ContextRecallPrompt,
        Faithfulness,
        AnswerRelevancy,
        StatementGeneratorPrompt,
        AnswerRelevancePrompt,
        BasePrompt,
    ):
        path = Path(inspect.getfile(cls))
        relative = "src/ragas/" + path.as_posix().rsplit("/ragas/", 1)[1]
        original = subprocess.check_output(
            ["git", "show", f"{BASE}:{relative}"],
            cwd=Path(__file__).resolve().parents[2],
        )
        if (
            path.read_text(encoding="utf-8") != original.decode()
            or (source_root / relative).read_text(encoding="utf-8") != original.decode()
        ):
            raise ValueError(f"Source mismatch for {relative}")
        hashes[relative] = hashlib.sha256(original).hexdigest()
    result = {
        "source_commit": BASE,
        "source_sha256": hashes,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p) for p in ("ragas", "pydantic", "numpy")
        },
        "prompts": {},
        "cases": [],
    }
    for metric_id, name, inputs, replies, vectors in cases():
        calls = []
        judge = Judge(replies, calls, result["prompts"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                score = await legacy_score(metric_id, inputs, replies, vectors, judge, calls)
            old = (
                {"value": score} if math.isfinite(score) else {"nonfinite": str(score)}
            )
        except (ValueError, TypeError) as exc:
            old = {"exception": type(exc).__name__}
        result["cases"].append(
            dict(
                metric=metric_id,
                name=name,
                inputs=inputs,
                replies=replies,
                vectors=vectors,
                calls=calls,
                old=old,
                intentional_difference=(
                    {"status": "undefined", "reason": "no_retrieved_ids" if metric_id == "id_context_precision" else "no_reference_ids"}
                    if metric_id in ID_METRICS and "nonfinite" in old
                    else INTENTIONAL_DIFFERENCES.get(name)
                ),
            )
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(generate(args.source_root))
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
