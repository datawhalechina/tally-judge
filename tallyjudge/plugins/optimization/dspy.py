"""DSPy MIPROv2 adapter migrated from Ragas ec00339, Apache-2.0.

One business instruction maps to a predictor. Only training examples enter
DSPy; every proposal, prediction and judge request uses the caller's model loop.
"""

import asyncio
from importlib import import_module
from typing import Any, cast

from pydantic import RootModel

from tallyjudge.core.jobs import validate_output
from tallyjudge.core.prompts import Message, PromptInput
from tallyjudge.core.usage import RequestLimitReached
from tallyjudge.plugins.optimization._experiment import OptimizerBase, Training


class _JsonReply(RootModel[dict[str, Any]]):
    pass


class DspyOptimizer(OptimizerBase):
    """Use DSPy's MIPROv2 to select zero-shot instructions under the same transport budget."""

    plugin_id = "optimization.prompt.dspy_mipro"

    async def search(self, training: Training) -> None:
        loop = asyncio.get_running_loop()
        dspy = import_module("dspy")
        lm15 = import_module("dspy.lm15")
        active = True
        pending: set[asyncio.Task] = set()

        async def dispatch(function, *args):
            if not active:
                raise asyncio.CancelledError
            task = asyncio.current_task()
            assert task is not None
            pending.add(task)
            try:
                with training.budget.activate():
                    return await function(*args)
            finally:
                pending.remove(task)

        def submit(function, *args):
            if not active:
                raise asyncio.CancelledError
            return asyncio.run_coroutine_threadsafe(
                dispatch(function, *args), loop
            ).result()

        async def complete(messages):
            return await self.model.agenerate(PromptInput(tuple(messages)), _JsonReply)

        class Engine:
            def complete(self, request):
                def text(parts):
                    return "".join(
                        part.text for part in parts if isinstance(part, lm15.TextPart)
                    )

                messages = []
                if request.system is not None:
                    messages.append(
                        Message(
                            "system",
                            request.system
                            if isinstance(request.system, str)
                            else text(request.system),
                        )
                    )
                messages.extend(
                    Message(m.role, text(m.parts)) for m in request.messages
                )
                reply = submit(complete, messages)
                content = reply.data.model_dump_json()
                tokens = reply.usage.known_tokens
                return lm15.Response(
                    id=None,
                    model="tallyjudge",
                    message=lm15.Message.assistant([lm15.TextPart(content)]),
                    finish_reason="stop",
                    usage=lm15.Usage(
                        input_tokens=tokens["input_tokens"],
                        output_tokens=tokens["output_tokens"],
                    ),
                )

        async def objective(row, response):
            output = await training.metric.ascore_row({**row, "response": response})
            score = validate_output(output, training.metric.metric.spec)
            if score.status != "success":
                raise ValueError(f"DSPy objective: {score.status}")
            return training.utility(cast(float, score.value))

        def compile_instruction():
            lm = dspy.BaseLM(
                model="tallyjudge", engine=Engine(), num_retries=0, cache=False
            )
            examples = [
                dspy.Example(**row).with_inputs("user_input")
                for row in training.train.records
            ]

            def metric(example, prediction, trace=None):
                return submit(objective, example.toDict(), prediction.response)

            with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
                program = dspy.Predict(
                    dspy.Signature("user_input -> response", training.original)
                )
                optimizer = dspy.MIPROv2(
                    metric,
                    prompt_model=lm,
                    task_model=lm,
                    auto=None,
                    num_candidates=self.settings.max_candidates,
                    max_labeled_demos=0,
                    max_bootstrapped_demos=0,
                    num_threads=1,
                    max_errors=1,
                    seed=self.settings.seed,
                )
                best = optimizer.compile(
                    program,
                    trainset=examples,
                    valset=examples,
                    num_trials=self.settings.max_candidates,
                    minibatch=False,
                    program_aware_proposer=False,
                    data_aware_proposer=False,
                    fewshot_aware_proposer=False,
                )
                return best.predictors()[0].signature.instructions

        try:
            try:
                best = await asyncio.to_thread(compile_instruction)
            except Exception:
                if training.budget.exhausted:
                    raise RequestLimitReached from None
                raise
        finally:
            active = False
            for task in tuple(pending):
                task.cancel()
        if training.budget.exhausted:
            raise RequestLimitReached
        await training.consider(best)
