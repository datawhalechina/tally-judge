import copy
from collections import deque
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

from pydantic import BaseModel

from tallyjudge.sdk.cache import ScoreCache
from tallyjudge.sdk.dataset import Dataset, canonical_json
from tallyjudge.sdk.documents import SourceDocument
from tallyjudge.sdk.errors import InputValidationError, UndefinedScore
from tallyjudge.sdk.importing import Importer
from tallyjudge.sdk.metrics import Metric, validate_output
from tallyjudge.sdk.models import LanguageModel
from tallyjudge.sdk.plugin import plugin_spec
from tallyjudge.sdk.prompts import PromptInput
from tallyjudge.sdk.results import MetricResult, RunResult
from tallyjudge.sdk.storage import StorageBackend
from tallyjudge.sdk.usage import UsageSummary
from tallyjudge.sdk.workflow import WorkflowTrace


class ScriptedClient:
    def __init__(self):
        self.responses: deque[Any] = deque()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.generate))
        self.embeddings = SimpleNamespace(create=self.embed)

    def with_options(self, **options):
        return self

    async def generate(self, **kwargs):
        return await self.reply("generation", kwargs)

    async def embed(self, **kwargs):
        return await self.reply("embedding", kwargs)

    async def reply(self, kind, kwargs):
        self.calls.append((kind, kwargs))
        value = self.responses.popleft()
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return await cast(Callable[[], Awaitable[Any]], value)()
        return value

    def chat_reply(self, content, *, tokens=None, finish="stop", refusal=None):
        message = SimpleNamespace(content=content, refusal=refusal)
        self.responses.append(
            SimpleNamespace(
                model="served-chat",
                usage=tokens,
                choices=[SimpleNamespace(message=message, finish_reason=finish)],
            )
        )

    def embedding_reply(self, entries, *, tokens=None):
        self.responses.append(
            SimpleNamespace(
                model="served-embedding",
                usage=tokens,
                data=[SimpleNamespace(index=i, embedding=v) for i, v in entries],
            )
        )


class _Probe(BaseModel):
    answer: str


async def check_metric(metric: Metric, inputs: dict[str, Any]) -> None:
    spec = metric.spec
    snapshot = canonical_json(metric.config_snapshot())
    original = copy.deepcopy(inputs)
    try:
        score = validate_output(await metric.ascore_inputs(inputs), spec)
    except UndefinedScore:
        score = None
    assert score is None or score.status == "success", "output violates its spec"
    assert inputs == original, "scoring changed its inputs"
    assert canonical_json(metric.config_snapshot()) == snapshot, "snapshot changed"
    required = next((name for name in spec.inputs if name not in spec.defaults), None)
    if required is not None:
        incomplete = {k: v for k, v in inputs.items() if k != required}
        try:
            await metric.ascore_inputs(incomplete)
        except InputValidationError as error:
            assert error.field == required
        else:
            raise AssertionError("a missing input was accepted")


async def check_language_model(model: LanguageModel) -> None:
    assert model.plugin_id and model.usage_tracking in ("per_request", "untracked")
    canonical_json(model.config_snapshot())
    reply = await model.agenerate(PromptInput.from_text("probe"), _Probe)
    assert isinstance(reply.data, _Probe) and isinstance(reply.usage, UsageSummary)


def check_storage(backend: StorageBackend, result: RunResult) -> None:
    backend.save_run(result)
    backend.save_run(result)
    assert backend.load_run(result.run_id).to_dict() == result.to_dict()
    try:
        backend.load_run("missing-" + result.run_id)
    except FileNotFoundError:
        return
    raise AssertionError("a missing run did not raise FileNotFoundError")


def check_importer(importer: Importer, source: Any) -> None:
    produces = plugin_spec(type(importer)).produces
    result = importer.convert(source)
    if produces == "trace":
        assert isinstance(result, WorkflowTrace), "a trace importer returns a trace"
    elif produces == "documents":
        assert all(isinstance(item, SourceDocument) for item in result), (
            "a document importer yields source documents"
        )
    else:
        assert produces == "records", "the importer declares no known product"
        Dataset.from_records(result)


def check_cache(cache: ScoreCache) -> None:
    assert cache.namespace
    assert cache.get("tallyjudge-contract-probe-miss") is None
    score = MetricResult(1.0, "probe")
    cache.put("tallyjudge-contract-probe", score)
    assert cache.get("tallyjudge-contract-probe") == score
