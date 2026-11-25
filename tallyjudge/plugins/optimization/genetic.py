"""Genetic prompt optimization adapted from Ragas ec00339, Apache-2.0.

Reverse engineering, feedback mutation, Hamming-neighbor crossover and fitness
target a business instruction. The bound objective supplies per-row fitness.
The parent search excludes self and uses the full symmetric Hamming distance.
"""

import json
import random
from typing import cast

from tallyjudge.plugins.optimization._experiment import (
    OptimizationConfig,
    OptimizerBase,
    Training,
)
from tallyjudge.sdk.config import EmptyConfig
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.prompts import Message, PromptInput


class _Instruction(EmptyConfig):
    instruction: str


class _Feedback(EmptyConfig):
    feedbacks: list[str]


class GeneticOptimizer(OptimizerBase):
    plugin = Plugin("optimization", dependencies={"llm": "language_model"})

    Config = OptimizationConfig

    plugin_id = "optimization.prompt.genetic"

    async def search(self, training: Training) -> None:
        rng = random.Random(self.settings.seed)
        examples = [
            {"user_input": row["user_input"], "reference": row["reference"]}
            for row in training.train.records
        ]
        spec = training.metric.metric.spec
        lower = spec.direction == "lower_is_better"
        optimum = spec.value_range[0 if lower else 1]
        if spec.levels:
            optimum = min(spec.levels) if lower else max(spec.levels)

        async def propose(action: str, data: dict) -> str:
            result = await self.model.agenerate(
                PromptInput(
                    (
                        Message(
                            "system",
                            action
                            + ' Return a JSON object with the string field "instruction".',
                        ),
                        Message("user", json.dumps(data, ensure_ascii=False)),
                    )
                ),
                _Instruction,
            )
            return result.data.instruction

        population = [training.original]
        for _ in range(max(1, (self.settings.max_candidates - 1) // 3)):
            sample = rng.sample(examples, min(3, len(examples)))
            candidate = await propose(
                "Infer a general instruction from these labeled examples.",
                {"examples": sample},
            )
            await training.consider(candidate)
            if candidate not in population:
                population.append(candidate)
        next_population = []
        for parent in population:
            if len(training.scores) >= self.settings.max_candidates:
                return
            feedback_examples = []
            for example, row in zip(examples, training.runs[parent].rows, strict=True):
                score = row.metrics[training.metric.key]
                if score.status == "success" and score.value != optimum:
                    assert row.task is not None and row.task.output is not None
                    value = cast(float, score.value)
                    feedback_examples.append(
                        {
                            **example,
                            "prediction": row.task.output["response"],
                            "score": value,
                        }
                    )
            if feedback_examples:
                feedback_examples.sort(key=lambda item: training.utility(item["score"]))
                feedback = await self.model.agenerate(
                    PromptInput(
                        (
                            Message(
                                "system",
                                "Give up to three concrete changes to the instruction based on its weakest scored predictions. Return a JSON object with a feedbacks array of strings.",
                            ),
                            Message(
                                "user",
                                json.dumps(
                                    {
                                        "instruction": parent,
                                        "direction": spec.direction,
                                        "errors": feedback_examples[:3],
                                    },
                                    ensure_ascii=False,
                                ),
                            ),
                        )
                    ),
                    _Feedback,
                )
                if feedback.data.feedbacks:
                    parent = await propose(
                        "Rewrite the instruction using the feedback.",
                        {"instruction": parent, "feedback": feedback.data.feedbacks},
                    )
                    await training.consider(parent)
            if parent not in next_population:
                next_population.append(parent)
        vectors = {}
        for parent in next_population:
            vectors[parent] = [
                row.metrics[training.metric.key].value
                for row in training.runs[parent].rows
            ]
        for parent in next_population:
            if (
                len(next_population) < 2
                or len(training.scores) >= self.settings.max_candidates
            ):
                break
            other = min(
                (p for p in next_population if p != parent),
                key=lambda p: sum(
                    a != b for a, b in zip(vectors[parent], vectors[p], strict=True)
                ),
            )
            offspring = await propose(
                "Cross over the two parent instructions into one instruction.",
                {"parents": [parent, other]},
            )
            await training.consider(offspring)
