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
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics._context_precision import IDBasedContextPrecision
from ragas.metrics._context_recall import IDBasedContextRecall
from ragas.metrics.collections._semantic_similarity import SemanticSimilarity
from ragas.metrics.collections.answer_accuracy.metric import AnswerAccuracy
from ragas.metrics.collections.answer_correctness.metric import AnswerCorrectness
from ragas.metrics.collections.answer_correctness.util import (
    CorrectnessClassifierPrompt,
)
from ragas.metrics.collections.answer_relevancy.metric import AnswerRelevancy
from ragas.metrics.collections.answer_relevancy.util import AnswerRelevancePrompt
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.collections.context_entity_recall.metric import ContextEntityRecall
from ragas.metrics.collections.context_precision.metric import (
    ContextPrecisionWithoutReference,
    ContextPrecisionWithReference,
)
from ragas.metrics.collections.context_precision.util import ContextPrecisionPrompt
from ragas.metrics.collections.context_recall.metric import ContextRecall
from ragas.metrics.collections.context_recall.util import ContextRecallPrompt
from ragas.metrics.collections.context_relevance.metric import ContextRelevance
from ragas.metrics.collections.factual_correctness.metric import FactualCorrectness
from ragas.metrics.collections.faithfulness.metric import Faithfulness
from ragas.metrics.collections.faithfulness.util import StatementGeneratorPrompt
from ragas.metrics.collections.noise_sensitivity.metric import NoiseSensitivity
from ragas.metrics.collections.response_groundedness.metric import ResponseGroundedness
from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.prompt.metrics.common import statement_generator_prompt

BASE = "ec003395bb37c223fc981e6567162513aec52a71"
QUESTION = "法国的首都是哪里？"
ANSWER = "巴黎。"

# Expected TallyJudge outcomes for cases where the upstream scorer raised or returned NaN.
INTENTIONAL_DIFFERENCES = {
    "no_classified_statements": {
        "status": "undefined",
        "reason": "no_classified_statements",
    },
    "no_predicted_statements": {
        "status": "undefined",
        "reason": "no_predicted_statements",
    },
    "bad:None": {"status": "error", "reason": "invalid_embedding"},
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


ID_METRICS = {
    "id_context_precision": IDBasedContextPrecision,
    "id_context_recall": IDBasedContextRecall,
}


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
            yield (
                metric_id,
                name,
                dict(retrieved_context_ids=retrieved, reference_context_ids=reference),
                [],
                {},
            )

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
            inputs = {
                "user_input": "question",
                field: answer,
                "retrieved_contexts": passages,
            }
            yield (
                metric_id,
                name,
                inputs,
                [{"reason": "fixture", "verdict": v} for v in flags],
                {},
            )

    for name, passages, flags in (
        ("mixed_attribution", ["first", "second", "first"], [1, 0, 1]),
        ("unsupported", ["first", "second", "first"], [0, 0]),
        ("no_classifications", ["first", "second", "first"], []),
        ("empty_passages", [], []),
        ("empty_passage_text", [""], [1]),
    ):
        inputs = {
            "user_input": "question",
            "reference": "gold",
            "retrieved_contexts": passages,
        }
        items = [
            {"statement": "same statement", "reason": "fixture", "attributed": flag}
            for flag in flags
        ]
        yield (
            "context_recall",
            name,
            inputs,
            [{"classifications": items}] if passages else [],
            {},
        )


def configured_cases():
    for case in cases():
        yield (*case, {})
    vectors = {
        "X": [1.0, 0.0],
        "Y": [0.0, 1.0],
        "-X": [-1.0, 0.0],
        "Z": [0.0, 0.0],
        " ": [1.0, 0.0],
        "bad": [1.0],
    }
    for text, threshold in (
        ("X", None),
        ("Y", None),
        ("-X", None),
        ("Y", 0.0),
        ("-X", 0.0),
        ("X", 1.0),
        ("Z", None),
        ("Z", 0.0),
        ("", None),
        ("bad", None),
    ):
        name = "zero_vector" if text == "Z" else f"{text or 'blank'}:{threshold}"
        yield (
            "semantic_similarity",
            name,
            {"reference": "X", "response": text},
            [],
            vectors,
            {"threshold": threshold},
        )

    inputs = {"user_input": "question", "response": "observed", "reference": "gold"}
    vectors = {"observed": [-1.0, 0.0], "gold": [1.0, 0.0], "": [-1.0, 0.0]}
    mixed = (["A", "B"], ["A", "C", "D"], (1, 1, 2))
    answer_only = (["B"], [], (0, 1, 0))
    reference_only = ([], ["C"], (0, 0, 1))
    empty = ([], [], None)
    for name, statements, weights, beta in (
        ("weighted", mixed, [0.75, 0.25], 1.0),
        ("beta_two", mixed, [2.0, 0.0], 2.0),
        ("scaled_weights", mixed, [3.0, 1.0], 1.0),
        ("answer_only", answer_only, [1.0, 0.0], 1.0),
        ("reference_only", reference_only, [1.0, 0.0], 1.0),
        ("negative_total", answer_only, [0.75, 0.25], 1.0),
        ("empty_factual", empty, [1.0, 0.0], 1.0),
        ("empty_weighted", empty, [0.75, 0.25], 1.0),
        ("no_classified_statements", (["A"], ["B"], (0, 0, 0)), [0.75, 0.25], 1.0),
        ("no_predicted_statements", reference_only, [1.0, 0.0], 0.0),
        ("beta_zero", answer_only, [1.0, 0.0], 0.0),
        ("zero_vector", mixed, [0.75, 0.25], 1.0),
        ("only_similarity", mixed, [0.0, 1.0], 1.0),
        ("raw_empty", reference_only, [0.75, 0.25], 1.0),
    ):
        left, right, counts = statements
        replies = [{"statements": left}, {"statements": right}]
        if counts is not None:
            replies.append(
                {
                    key: [{"statement": "item", "reason": "fixture"}] * count
                    for key, count in zip(("TP", "FP", "FN"), counts, strict=True)
                }
            )
        row = dict(inputs, response="") if name == "raw_empty" else inputs
        directions = (
            dict(vectors, observed=[0.0, 0.0]) if name == "zero_vector" else vectors
        )
        yield (
            "answer_correctness",
            name,
            row,
            replies,
            directions,
            {"weights": weights, "beta": beta},
        )


DIAGNOSTIC_METRICS = {
    "context_entity_recall": ContextEntityRecall,
    "factual_correctness": FactualCorrectness,
    "noise_sensitivity": NoiseSensitivity,
    "answer_accuracy": AnswerAccuracy,
    "context_relevance": ContextRelevance,
    "response_groundedness": ResponseGroundedness,
}


def diagnostic_cases():
    """Inputs and responses for the six archived phase 6 scorers, in request order."""
    segments = ["first passage", "second passage"]
    entity_input = {
        "reference": "Paris is on the Seine.",
        "retrieved_contexts": segments,
    }
    for name, passages, gold, found in (
        ("duplicates", segments, ["巴黎", "塞纳河", "巴黎"], ["巴黎", "法国"]),
        ("case_sensitive", segments, ["Paris"], ["paris"]),
        ("no_reference_entities", segments, [], ["Paris"]),
        ("empty_retrieval", [], ["Paris"], []),
    ):
        row = dict(entity_input, retrieved_contexts=passages)
        yield (
            "context_entity_recall",
            name,
            row,
            [{"entities": gold}, {"entities": found}],
            {},
            {},
        )
    yield (
        "context_entity_recall",
        "string_context",
        dict(entity_input, retrieved_contexts="text"),
        [],
        {},
        {},
    )

    def judgments(flags):
        return {
            "statements": [
                {"statement": f"claim {i}", "reason": "fixture", "verdict": value}
                for i, value in enumerate(flags)
            ]
        }

    def direction(claims, flags):
        replies = [{"claims": claims}]
        if claims:
            replies.append(judgments(flags))
        return replies

    facts = {"response": "observed answer", "reference": "reference answer"}
    forward = direction(["A", "B", "C"], [1, 1, 0])
    reverse = direction(["D", "E"], [1, 0])
    empty = direction([], [])
    for name, config, replies in (
        ("precision", {"mode": "precision"}, forward),
        ("recall", {"mode": "recall"}, forward + reverse),
        ("f1", {}, forward + reverse),
        ("beta_two", {"beta": 2.0}, forward + reverse),
        (
            "high_atomicity_coverage",
            {"atomicity": "high", "coverage": "high"},
            forward + reverse,
        ),
        ("empty_response_claims", {}, empty + reverse),
        ("empty_reference_claims", {}, forward + empty),
        ("empty_verdicts", {}, direction(["A"], []) + reverse),
        ("all_unsupported", {}, direction(["A"], [0]) + direction(["D"], [0])),
        (
            "beta_zero_no_match_in_reverse",
            {"beta": 0.0},
            forward + direction(["D"], [0]),
        ),
        ("recall_empty_forward", {"mode": "recall"}, empty + reverse),
    ):
        yield "factual_correctness", name, facts, replies, {}, config

    noisy = dict(facts, user_input="question", retrieved_contexts=segments)

    def statements(items):
        return {"statements": items}

    decomposed = [statements(["gold"]), statements(["A", "B", "C", "D"])]
    first = [judgments([1]), judgments([1, 1, 1, 0])]
    second = [judgments([0]), judgments([0, 0, 1, 1])]
    final = judgments([1, 0, 0, 0])
    for mode in ("relevant", "irrelevant"):
        yield (
            "noise_sensitivity",
            mode,
            noisy,
            decomposed + first + second + [final],
            {},
            {"mode": mode},
        )
    for name, row, replies in (
        ("empty_retrieval", dict(noisy, retrieved_contexts=[]), []),
        (
            "empty_reference_statements",
            noisy,
            [
                statements([]),
                decomposed[1],
                judgments([]),
                first[1],
                judgments([]),
                second[1],
                final,
            ],
        ),
        (
            "empty_response_statements",
            noisy,
            [
                decomposed[0],
                statements([]),
                first[0],
                judgments([]),
                second[0],
                judgments([]),
                judgments([]),
            ],
        ),
        ("ragged_verdicts", noisy, decomposed + first + [second[0], judgments([0, 0])]),
    ):
        yield "noise_sensitivity", name, row, replies, {}, {}

    accuracy = dict(facts, user_input="question")
    relevance = {"user_input": "a long question", "retrieved_contexts": segments}
    grounded = {"response": "observed answer", "retrieved_contexts": segments}
    for metric, row, scale, other in (
        ("answer_accuracy", accuracy, 4, 2),
        ("context_relevance", relevance, 2, 1),
        ("response_groundedness", grounded, 2, 0),
    ):
        for name, ratings in (
            ("two_ratings", [scale, other]),
            ("retry_then_valid", [9, scale, other]),
            ("only_first_valid", [scale] + [9] * 5),
            ("only_second_valid", [9] * 5 + [scale]),
            ("both_invalid", [9] * 10),
        ):
            yield metric, name, row, [{"rating": value} for value in ratings], {}, {}
    yield (
        "answer_accuracy",
        "retry_invalid_schema",
        accuracy,
        [{}, {"rating": 4}, {"rating": 2}],
        {},
        {},
    )
    yield "answer_accuracy", "empty_answer", dict(accuracy, response=""), [], {}, {}
    yield (
        "answer_accuracy",
        "whitespace_answer",
        dict(accuracy, response=" "),
        [{"rating": 0}, {"rating": 0}],
        {},
        {},
    )
    for name, row in (
        ("contained", dict(relevance, retrieved_contexts=["long"])),
        (
            "equal_after_strip",
            dict(relevance, retrieved_contexts=[" a long question "]),
        ),
        ("whitespace", dict(relevance, retrieved_contexts=[" "])),
        ("empty_retrieval", dict(relevance, retrieved_contexts=[])),
    ):
        yield "context_relevance", name, row, [], {}, {}
    for name, row in (
        ("blank_response", dict(grounded, response=" ")),
        ("blank_context", dict(grounded, retrieved_contexts=[" "])),
        ("empty_retrieval", dict(grounded, retrieved_contexts=[])),
    ):
        yield "response_groundedness", name, row, [], {}, {}


async def legacy_score(metric_id, inputs, replies, vectors, judge, calls, settings):
    if metric_id in ID_METRICS:
        return await ID_METRICS[metric_id]().single_turn_ascore(
            SingleTurnSample(**inputs)
        )
    if metric_id in DIAGNOSTIC_METRICS:
        metric = DIAGNOSTIC_METRICS[metric_id](llm=judge, **settings)
    elif metric_id == "answer_correctness":
        embeddings = Embeddings(vectors, calls) if settings["weights"][1] else None
        metric = AnswerCorrectness(llm=judge, embeddings=embeddings, **settings)
    elif metric_id == "semantic_similarity":
        metric = SemanticSimilarity(embeddings=Embeddings(vectors, calls), **settings)
    elif metric_id in CONTEXT_METRICS:
        metric = CONTEXT_METRICS[metric_id](llm=judge)
    elif metric_id == "faithfulness":
        metric = Faithfulness(llm=judge)
    else:
        metric = AnswerRelevancy(
            llm=judge, embeddings=Embeddings(vectors, calls), strictness=len(replies)
        )
    return (await metric.ascore(**inputs)).value


def source_files(suite):
    if suite == "phase6":
        for metric in DIAGNOSTIC_METRICS.values():
            path = Path(inspect.getfile(metric))
            yield path
            yield path.with_name("util.py")
        for source in (
            BaseMetric,
            MetricResult,
            BasePrompt,
            statement_generator_prompt,
        ):
            yield Path(inspect.getfile(source))
        return
    for cls in (
        AnswerCorrectness,
        CorrectnessClassifierPrompt,
        SemanticSimilarity,
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
        yield Path(inspect.getfile(cls))


async def generate(source_root, suite="core"):
    hashes = {}
    for path in source_files(suite):
        relative = "src/ragas/" + path.as_posix().rsplit("/ragas/", 1)[1]
        original = subprocess.check_output(
            ["git", "show", f"{BASE}:{relative}"],
            cwd=source_root,
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
    samples = diagnostic_cases() if suite == "phase6" else configured_cases()
    for metric_id, name, inputs, replies, vectors, settings in samples:
        calls = []
        judge = Judge(replies, calls, result["prompts"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                score = await legacy_score(
                    metric_id, inputs, replies, vectors, judge, calls, settings
                )
            old = (
                {"value": score} if math.isfinite(score) else {"nonfinite": str(score)}
            )
        except (ValueError, TypeError) as exc:
            old = {"exception": type(exc).__name__}
        observation = dict(
            metric=metric_id,
            name=name,
            inputs=inputs,
            replies=replies,
            settings=settings,
            vectors=vectors,
            calls=calls,
            old=old,
        )
        if suite == "phase6":
            # The old dual judges swallow exhausted scripts as ordinary failed ratings.
            if len(calls) != len(replies):
                raise AssertionError(
                    f"{metric_id}/{name}: {len(calls)} calls for {len(replies)} replies"
                )
        else:
            observation["intentional_difference"] = (
                {
                    "status": "undefined",
                    "reason": "no_retrieved_ids"
                    if metric_id == "id_context_precision"
                    else "no_reference_ids",
                }
                if metric_id in ID_METRICS and "nonfinite" in old
                else INTENTIONAL_DIFFERENCES.get(name)
            )
        result["cases"].append(observation)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--suite", choices=("core", "phase6"), default="core")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(generate(args.source_root, args.suite))
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.output.write_text(
        payload, encoding="utf-8", newline="\n" if args.suite == "phase6" else None
    )
    if args.suite == "phase6":
        samples = result["cases"]
        print(
            f"{len(samples)} cases, {sum(len(s['calls']) for s in samples)} generation calls, "
            f"{len(result['prompts'])} distinct prompts"
        )
        print(f"SHA-256: {hashlib.sha256(payload.encode('utf-8')).hexdigest()}")
