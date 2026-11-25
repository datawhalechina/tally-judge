"""Compare a generate_rag_baselines.py report with the installed TallyJudge package."""

import argparse
import asyncio
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

from tallyjudge import Dataset, Evaluator, create_metric
from tallyjudge.sdk.usage import UsageSummary
from tallyjudge.sdk.models import EmbeddingResult, LanguageModelBase


# The public score schema records these degenerate cases with a reason instead of an archived number.
UNDEFINED = {
    (
        "context_precision_with_reference",
        "no_relevant_contexts",
    ): "no_relevant_contexts",
    (
        "context_precision_without_reference",
        "no_relevant_contexts",
    ): "no_relevant_contexts",
    ("semantic_similarity", "zero_vector"): "zero_norm_embedding",
    ("answer_correctness", "no_classified_statements"): "no_classified_statements",
    ("answer_correctness", "no_predicted_statements"): "no_predicted_statements",
    ("answer_correctness", "zero_vector"): "zero_norm_embedding",
    ("context_entity_recall", "no_reference_entities"): "no_reference_entities",
    ("factual_correctness", "empty_response_claims"): "no_response_statements",
    ("factual_correctness", "empty_reference_claims"): "no_reference_statements",
    ("factual_correctness", "empty_verdicts"): "no_verdicts",
    ("noise_sensitivity", "empty_reference_statements"): "no_reference_statements",
}


class RecordedModel(LanguageModelBase):
    """Replay each model prompt and embedding from the current archived sample."""

    plugin_id = "model.fixture.archive_replay"
    usage_tracking = "untracked"

    def __init__(self, case):
        self.vectors = case["vectors"]
        self._json_retries = int(case["name"] == "retry_invalid_schema")
        self.schemas = {}
        self.responses = defaultdict(deque)
        self.requests = 0
        self.unmatched = []
        calls = (c for c in case["calls"] if "generate" in c)
        for call, reply in zip(calls, case["replies"]):
            digest = call["generate"]
            self.schemas[digest] = call["schema"]
            self.responses[digest].append(reply)

    @property
    def structured_retries(self):
        return self._json_retries

    def config_snapshot(self):
        return {}

    async def agenerate(self, prompt, response_model):
        (message,) = prompt.messages
        digest = hashlib.sha256(message.content.encode()).hexdigest()
        if (
            message.role != "user"
            or self.schemas.get(digest) != response_model.__name__
        ):
            self.unmatched.append((digest, response_model.__name__))
            raise AssertionError("Unrecorded prompt or output schema")
        return await super().agenerate(prompt, response_model)

    async def _acomplete(self, messages, records):
        digest = hashlib.sha256(messages[0].content.encode()).hexdigest()
        if not self.responses[digest]:
            self.unmatched.append((digest, "extra attempt"))
            raise AssertionError("More attempts than in the archive")
        self.requests += 1
        return json.dumps(self.responses[digest].popleft()), None

    async def aembed_texts(self, texts):
        return EmbeddingResult([self.vectors[text] for text in texts], UsageSummary())


def replay_metric(case, model):
    name = case["metric"]
    settings = dict(case["settings"])
    if name.startswith("id_context_"):
        return create_metric("metric.retrieval." + name, **settings)
    if name == "semantic_similarity":
        return create_metric("metric.text." + name, embeddings=model, **settings)
    if name == "answer_relevance":
        settings.update(embeddings=model, strictness=len(case["replies"]))
    if name == "answer_correctness" and settings["weights"][1]:
        settings["embeddings"] = model
    return create_metric("metric.rag." + name, llm=model, **settings)


async def compare(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    outcomes = Counter()
    requests = 0
    for case in report["cases"]:
        identity = case["metric"], case["name"]
        model = RecordedModel(case)
        metric = replay_metric(case, model)
        data = Dataset.from_records([case["inputs"]])
        result = await Evaluator([metric]).arun(data)
        score = result.rows[0].metrics[metric.spec.id]
        old = case["old"]
        reason = UNDEFINED.get(identity)
        if reason:
            match = score.status == "undefined" and score.error.code == reason
        elif "value" in old:
            match = score.status == "success" and math.isclose(
                score.value, old["value"], abs_tol=1e-12, rel_tol=1e-12
            )
        elif "nonfinite" in old:
            match = score.status == "undefined"
        else:
            match = score.status == "error"
        if not match or model.unmatched:
            raise AssertionError((identity, old, score, model.unmatched))
        outcomes[f"{next(iter(old))} -> {score.status}"] += 1
        requests += model.requests
    print(f"{path.name}: {len(report['cases'])} cases, {requests} matched requests")
    for key, count in sorted(outcomes.items()):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    logging.basicConfig(
        level=logging.CRITICAL
    )  # Expected failure cases are compared below.
    for path in parser.parse_args().reports:
        asyncio.run(compare(path))
