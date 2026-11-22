"""Ragas ec00339 default document and prechunked graph transforms (Apache-2.0).

Keep caller-owned source IDs and the original transform order. The raw-document
pipeline selects one length branch for the corpus. Prechunked inputs skip splitting.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from math import sumprod
from typing import Any

from pydantic import BaseModel

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.prompts import StructuredPrompt
from tallyjudge.core.usage import UsageSummary, mark_usage_source, run_scoped_score
from tallyjudge.interfaces.models import EmbeddingModel, LanguageModel
from tallyjudge.interfaces.synthesis import SourceDocument
from tallyjudge.plugins._vectors import unit_vector
from tallyjudge.plugins.synthesis._planning_prompts import (
    DEFAULT_RUBRICS,
    HeadlinesExtractorPrompt,
    QuestionPotentialInput,
    QuestionPotentialPrompt,
    StringIO,
    SummaryExtractorPrompt,
    ThemesAndConceptsExtractorPrompt,
)
from tallyjudge.plugins.synthesis._prompts import NERPrompt, TextWithExtractionLimit
from tallyjudge.plugins.synthesis.graph import (
    EvidenceGraph,
    EvidenceRelation,
    entity_overlaps,
    extraction_windows,
)
from tallyjudge.plugins.synthesis.scenarios import EvidenceChunk
from tallyjudge.plugins.tokenizers import tiktoken_counter


class _Extractors:
    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.usage = UsageSummary()

    async def ask[I: BaseModel, O: BaseModel](
        self, template: StructuredPrompt[I, O], data: I
    ) -> O:
        mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
        reply = await run_scoped_score(
            self.llm.agenerate(template.render(data), template.output_model)
        )
        self.usage.add(reply.usage)
        self.usage.complete &= self.llm.usage_tracking == "per_request"
        return reply.data

    async def headlines(self, text: str) -> list[str]:
        headings = []
        for window in extraction_windows(text, 32000):
            reply = await self.ask(
                HeadlinesExtractorPrompt(),
                TextWithExtractionLimit(text=window, max_num=5),
            )
            headings.extend(reply.headlines)
        return headings

    async def annotate(self, row: dict[str, Any]) -> None:
        themes, entities = [], []
        for window in extraction_windows(row["text"], 32000):
            data = TextWithExtractionLimit(text=window, max_num=10)
            themes.extend(
                (await self.ask(ThemesAndConceptsExtractorPrompt(), data)).output
            )
            entities.extend((await self.ask(NERPrompt(), data)).entities)
        row.update(themes=themes, entities=entities)


def _headline_chunks(
    text: str, headings: Sequence[str], count: Callable[[str], int]
) -> list[str]:
    positions = [
        0,
        *(pos for title in headings if (pos := text.find(title)) != -1),
        len(text),
    ]
    chunks = []
    pending = ""
    for start, end in zip(positions, positions[1:]):
        section = text[start:end]
        size = count(section)
        while size > 1000:
            words = section.split()
            split_at = max(1, int(len(words) * (1000 / size)))
            chunks.append(" ".join(words[:split_at]))
            section = " ".join(words[split_at:])
            size = count(section)
        if size < 500:
            pending = pending + " " + section if pending else section
            if count(pending) >= 500:
                chunks.append(pending)
                pending = ""
        else:
            if pending:
                chunks.append(pending)
                pending = ""
            chunks.append(section)
    if pending:
        chunks.append(pending)
    return chunks


async def _summary_links(
    rows: Iterable[dict[str, Any]],
    embeddings: EmbeddingModel,
    threshold: float,
) -> tuple[list[EvidenceRelation], UsageSummary]:
    vectors: dict[str, list[float]] = {}
    embedding_usage = UsageSummary()
    dimension = None
    for row in rows:
        mark_usage_source("embedding", embeddings.plugin_id, embeddings.usage_tracking)
        batch = await run_scoped_score(embeddings.aembed_texts([row["summary"]]))
        if len(batch.vectors) != 1:
            raise ModelOutputError("invalid_embedding")
        vector = row["summary_embedding"] = batch.vectors[0]
        if dimension is not None and len(vector) != dimension:
            raise ModelOutputError("invalid_embedding")
        dimension = len(vector)
        # A zero norm yields no cosine edges in the archive. Persona grouping uses the raw vector.
        if any(vector):
            vectors[row["chunk_id"]] = unit_vector(vector)
        embedding_usage.add(batch.usage)
        embedding_usage.complete &= embeddings.usage_tracking == "per_request"

    semantic = []
    for (a, left), (b, right) in combinations(vectors.items(), 2):
        similarity = sumprod(left, right)
        if similarity >= threshold:
            semantic.append(
                EvidenceRelation(
                    type="summary_similarity",
                    source=a,
                    target=b,
                    bidirectional=True,
                    properties={"summary_similarity": similarity},
                )
            )
    return semantic, embedding_usage


async def _enrich(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    subjects: list[dict[str, Any]],
    edges: list[EvidenceRelation],
    *,
    extractor: _Extractors,
    embeddings: EmbeddingModel,
    threshold: float,
) -> EvidenceGraph:
    parents = {edge.target: edge.source for edge in edges if edge.type == "child"}
    by_id = {row["chunk_id"]: row for row in rows}
    for row in summaries:
        window = next(iter(extraction_windows(row["text"], 32000)))
        row["summary"] = (
            await extractor.ask(SummaryExtractorPrompt(), StringIO(text=window))
        ).text

    rejected = set()
    for row in subjects:
        key = row["chunk_id"]
        summary = by_id[parents.get(key, key)].get("summary", "")
        if summary:
            reply = await extractor.ask(
                QuestionPotentialPrompt(),
                QuestionPotentialInput(
                    document_summary=summary,
                    node_content=row["text"],
                    rubrics=DEFAULT_RUBRICS,
                ),
            )
            if reply.score <= 2:
                rejected.add(key)

    semantic, embedding_usage = await _summary_links(
        (row for row in summaries if row["chunk_id"] not in rejected),
        embeddings,
        threshold,
    )
    selected = [row for row in subjects if row["chunk_id"] not in rejected]
    for row in selected:
        await extractor.annotate(row)
    semantic.extend(entity_overlaps(selected))
    hierarchy = [
        e for e in edges if e.source not in rejected and e.target not in rejected
    ]
    return EvidenceGraph(
        (row for row in rows if row["chunk_id"] not in rejected),
        ((edge.source, edge.target) for edge in semantic),
        embedding_usage,
        relationships=[*hierarchy, *semantic],
        generation_usage=extractor.usage,
    )


async def build_chunk_graph(
    chunks: Iterable[Mapping[str, Any]],
    *,
    llm: LanguageModel,
    embeddings: EmbeddingModel,
) -> EvidenceGraph:
    """Apply the archived prechunked pipeline to identified passages of any length."""
    # Materialize and validate source identities before spending model requests.
    source = EvidenceGraph(chunks, ())
    rows = [dict(row, node_type="chunk") for row in source]
    for row in rows:
        EvidenceChunk.model_validate(row)
    return await _enrich(
        rows,
        rows,
        rows,
        [],
        extractor=_Extractors(llm),
        embeddings=embeddings,
        threshold=0.7,
    )


async def build_document_graph(
    documents: Iterable[SourceDocument],
    *,
    llm: LanguageModel,
    embeddings: EmbeddingModel,
    count_tokens: Callable[[str], int] | None = None,
) -> EvidenceGraph:
    """Build the archived raw-document graph, using cl100k_base for the corpus length bins.

    At least a quarter of documents must exceed 500 tokens or a quarter must fall
    in the 101–500 bin. Use build_chunk_graph for an already segmented knowledge base.
    """
    rows = []
    for doc in documents:
        key = (
            doc.document_id
            if doc.page is None
            else f"{doc.document_id}#page={doc.page}"
        )
        row: dict[str, Any] = {
            "document_id": doc.document_id,
            "chunk_id": key,
            "text": doc.text,
            "node_type": "document",
        }
        if doc.page is not None:
            row["page"] = doc.page
        rows.append(row)
    source = EvidenceGraph(rows, ())
    if not rows:
        raise ValueError("At least one source document is required")
    count = (
        count_tokens if count_tokens is not None else tiktoken_counter("cl100k_base")
    )
    sizes = [count(node.text) for node in source._evidence.values()]
    long = sum(size > 500 for size in sizes) * 4 >= len(sizes)
    if not long and sum(100 < size <= 500 for size in sizes) * 4 < len(sizes):
        raise ValueError(
            "Documents are too short for the archived length distribution. Use build_chunk_graph for prechunked passages."
        )
    summaries = [
        row
        for row, size in zip(rows, sizes, strict=True)
        if size > (500 if long else 100)
    ]
    extractor = _Extractors(llm)
    children = []
    edges = []
    if long:
        for row in summaries:
            text = row["text"]
            row["headlines"] = await extractor.headlines(text)
            parts = _headline_chunks(text, row["headlines"], count)
            if len(parts) <= 1:
                continue
            previous = None
            for number, part in enumerate(parts, 1):
                key = f"{row['chunk_id']}#chunk={number}"
                child = {
                    "document_id": row["document_id"],
                    "chunk_id": key,
                    "node_type": "chunk",
                    "text": part,
                }
                if "page" in row:
                    child["page"] = row["page"]
                children.append(child)
                edges.append(
                    EvidenceRelation(type="child", source=row["chunk_id"], target=key)
                )
                if previous is not None:
                    edges.append(
                        EvidenceRelation(type="next", source=previous, target=key)
                    )
                previous = key
    return await _enrich(
        rows + children,
        summaries,
        children if long else rows,
        edges,
        extractor=extractor,
        embeddings=embeddings,
        threshold=0.7 if long else 0.5,
    )
