"""Evidence relations for multi-hop synthesis.

Adapted from Ragas ec00339 (Apache-2.0), KnowledgeGraph.find_two_nodes_single_rel
and find_indirect_clusters, NERExtractor and OverlapScoreBuilder. Caller-owned
chunk IDs replace graph node UUIDs.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cached_property
from importlib import import_module
from itertools import combinations
from typing import Any

from pydantic import BaseModel, Field

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.usage import UsageSummary, mark_usage_source, run_scoped_score
from tallyjudge.interfaces.models import EmbeddingModel, LanguageModel
from tallyjudge.plugins._vectors import unit_vector
from tallyjudge.plugins.synthesis._prompts import (
    NEROutput,
    NERPrompt,
    TextWithExtractionLimit,
)
from tallyjudge.plugins.synthesis.scenarios import EvidenceChunk


class EvidenceRelation(BaseModel):
    """A directed or bidirectional link, including the properties consumed by query strategies."""

    type: str
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)
    bidirectional: bool = False
    id: str | None = None


class _GraphSnapshot(BaseModel):
    chunks: list[dict[str, Any]]
    relations: list[tuple[str, str]]
    generation_usage: UsageSummary
    embedding_usage: UsageSummary | None


class _TypedGraphSnapshot(_GraphSnapshot):
    relationships: list[EvidenceRelation]


@dataclass(init=False)
class EvidenceGraph:
    """An owned snapshot of chunk records and undirected evidence pairs."""

    _records: dict[str, dict[str, Any]]
    _relations: tuple[tuple[str, str], ...]
    _relationships: tuple[EvidenceRelation, ...]
    _embedding_usage: UsageSummary | None
    _generation_usage: UsageSummary

    def __init__(
        self,
        chunks: Iterable[Mapping[str, Any]],
        relations: Iterable[tuple[str, str]],
        embedding_usage: UsageSummary | None = None,
        *,
        relationships: Iterable[EvidenceRelation] = (),
        generation_usage: UsageSummary | None = None,
    ):
        self._records = {}
        for row in chunks:
            key = row["chunk_id"]
            if key in self._records:
                raise ValueError(f"Duplicate chunk_id: {key!r}")
            self._records[key] = deepcopy(dict(row))
        self._relationships = tuple(deepcopy(edge) for edge in relationships)
        for edge in self._relationships:
            if edge.source not in self._records or edge.target not in self._records:
                raise ValueError("Evidence relationship refers to an unknown chunk")
        self._set_relations(relations)
        self._embedding_usage = embedding_usage
        self._generation_usage = generation_usage or UsageSummary()

    def _set_relations(self, relations: Iterable[tuple[str, str]]) -> None:
        pairs = set()
        for a, b in relations:
            if a not in self._records or b not in self._records:
                raise ValueError("Evidence relation refers to an unknown chunk")
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        self._relations = tuple(sorted(pairs))

    @cached_property
    def _evidence(self) -> dict[str, EvidenceChunk]:
        """Validate the scoring fields once when a generation or embedding needs them."""
        return {
            key: EvidenceChunk.model_validate(row) for key, row in self._records.items()
        }

    @property
    def chunks(self) -> tuple[Mapping[str, Any], ...]:
        """Return detached records, including original offsets and metadata."""
        return tuple(self)

    @property
    def relations(self) -> tuple[tuple[str, str], ...]:
        """Return the canonical undirected edges."""
        return self._relations

    @property
    def relationships(self) -> tuple[EvidenceRelation, ...]:
        """Return typed links without merging independent relationships between two nodes."""
        return deepcopy(self._relationships)

    @property
    def embedding_usage(self) -> UsageSummary | None:
        """Return usage from constructing a similarity graph."""
        return self._embedding_usage

    @property
    def generation_usage(self) -> UsageSummary:
        """Return model usage for the language-model transforms that built this graph."""
        return self._generation_usage

    def to_dict(self) -> dict[str, Any]:
        """Export the evidence, relations and original build usage for JSON storage."""
        return {
            "schema_version": 2,
            "relationships": [
                e.model_dump(exclude_none=True) for e in self._relationships
            ],
            "chunks": list(self),
            "relations": [list(edge) for edge in self.relations],
            "generation_usage": asdict(self.generation_usage),
            "embedding_usage": (
                asdict(self.embedding_usage)
                if self.embedding_usage is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGraph":
        """Restore a native graph snapshot without extracting entities or embedding again."""
        version = data["schema_version"]
        if type(version) is not int or version not in (1, 2):
            raise ValueError(f"Unsupported evidence graph schema version: {version!r}")
        if version == 2:
            snapshot = _TypedGraphSnapshot.model_validate(data)
            relationships = snapshot.relationships
        else:
            snapshot = _GraphSnapshot.model_validate(data)
            relationships = ()
        return cls(
            snapshot.chunks,
            snapshot.relations,
            snapshot.embedding_usage,
            relationships=relationships,
            generation_usage=snapshot.generation_usage,
        )

    @classmethod
    def from_ragas(
        cls,
        data: Mapping[str, Any],
        *,
        relation_types: tuple[str, ...] = ("entities_overlap", "summary_similarity"),
    ) -> "EvidenceGraph":
        """Read archival KnowledgeGraph.save JSON with node IDs and explicit relation types."""
        edges = data["relationships"]
        parents = {
            rel["target"]: rel["source"] for rel in edges if rel["type"] == "child"
        }
        nodes = {node["id"]: node for node in data["nodes"]}
        chunks = []
        for key, node in nodes.items():
            props = node["properties"]
            if node["type"] not in ("chunk", "document") or not props.get(
                "page_content"
            ):
                continue
            parent = parents.get(key, key)
            metadata = nodes[parent]["properties"].get("document_metadata", {})
            chunks.append(
                {
                    **{k: deepcopy(v) for k, v in props.items() if k != "page_content"},
                    "node_type": node["type"],
                    "chunk_id": key,
                    "document_id": str(metadata.get("source") or parent),
                    "text": props["page_content"],
                }
            )
        eligible = {row["chunk_id"] for row in chunks}
        typed = [
            EvidenceRelation.model_validate(row)
            for row in edges
            if row["source"] in eligible
            and row["target"] in eligible
            and row["type"] in (*relation_types, "child", "next")
        ]
        pairs = [
            (edge.source, edge.target) for edge in typed if edge.type in relation_types
        ]
        return cls(chunks, pairs, relationships=typed)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return (deepcopy(row) for row in self._records.values())


def entity_overlaps(chunks: Iterable[Mapping[str, Any]]) -> list[EvidenceRelation]:
    """Compute archived overlap scores and retain matching pairs in their original spelling."""
    # OverlapScoreBuilder (Ragas ec00339) ignores the most frequent 5% of all entity labels.
    rows = [(str(row["chunk_id"]), row.get("entities", ())) for row in chunks]
    counts = Counter(label for _, labels in rows for label in labels)
    noisy = {label for label, _ in counts.most_common(max(1, len(counts) // 20))}
    eligible = [
        (key, [label for label in labels if label not in noisy]) for key, labels in rows
    ]
    eligible = [(key, labels) for key, labels in eligible if labels]
    if len(eligible) < 2:
        return []
    distance = import_module("rapidfuzz.distance").JaroWinkler.distance

    edges = []
    for (first, left), (second, right) in combinations(eligible, 2):
        matched = [
            [a, b]
            for a in left
            for b in right
            if 1 - distance(a.lower(), b.lower()) >= 0.9
        ]
        score = len(matched) / (len(left) * len(right))
        if score >= 0.01:
            edges.append(
                EvidenceRelation(
                    type="entities_overlap",
                    source=first,
                    target=second,
                    bidirectional=True,
                    properties={
                        "entities_overlap_score": score,
                        "overlapped_items": matched,
                    },
                )
            )
    return edges


def build_graph(
    chunks: Iterable[Mapping[str, Any]], *, indirect: bool = False
) -> EvidenceGraph:
    """Connect entity overlaps using the archived frequency and similarity thresholds. Leiden community pairs require the graph extra."""
    evidence = EvidenceGraph(chunks, ())
    overlaps = entity_overlaps(evidence._records.values())
    evidence._relationships = tuple(overlaps)
    pairs = {(min(e.source, e.target), max(e.source, e.target)) for e in overlaps}
    if indirect and pairs:
        nx = import_module("networkx")
        graph = nx.from_edgelist(sorted(pairs))
        names = sorted(graph)
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=names, format="csr")
        labels = (
            import_module("sknetwork.clustering")
            .Leiden(random_state=42)
            .fit_predict(import_module("scipy.sparse").csr_matrix(adjacency))
        )
        groups = defaultdict(list)
        for key, label in zip(names, labels, strict=True):
            groups[int(label)].append(key)
        for members in groups.values():
            pairs.update(combinations(sorted(members), 2))
    evidence._set_relations(pairs)
    return evidence


def extraction_windows(text: str, limit: int) -> Iterable[str]:
    """Yield the original extractor's o200k_base token windows."""
    if not text:
        return
    # One token consumes at least one UTF-8 byte. Ordinary short chunks round-trip unchanged.
    if len(text.encode("utf-8")) <= limit:
        yield text
        return
    codec = import_module("tiktoken").get_encoding("o200k_base")
    tokens = codec.encode(text, disallowed_special=())
    for start in range(0, len(tokens), limit):
        yield codec.decode(tokens[start : start + limit])


async def build_entity_graph(
    chunks: Iterable[Mapping[str, Any]],
    *,
    llm: LanguageModel,
    max_token_limit: int = 32000,
) -> EvidenceGraph:
    """Extract missing labels from individual chunks and connect shared mentions.

    Preannotated chunks retain their labels. The archive extracts entities even for
    a single chunk. Graph usage is recorded separately from question generation.
    """
    evidence = EvidenceGraph(chunks, ())
    if max_token_limit < 1:
        raise ValueError("max_token_limit must be positive")
    template = NERPrompt()
    for record in evidence._records.values():
        if record.get("entities"):
            continue
        entities = []
        for window in extraction_windows(record["text"], max_token_limit):
            mark_usage_source("generation", llm.plugin_id, llm.usage_tracking)
            reply = await run_scoped_score(
                llm.agenerate(
                    template.render(TextWithExtractionLimit(text=window, max_num=10)),
                    NEROutput,
                )
            )
            entities.extend(reply.data.entities)
            evidence._generation_usage.add(reply.usage)
            evidence._generation_usage.complete &= llm.usage_tracking == "per_request"
        record["entities"] = entities
    overlaps = entity_overlaps(evidence._records.values())
    evidence._relationships = tuple(overlaps)
    evidence._set_relations((edge.source, edge.target) for edge in overlaps)
    return evidence


async def build_similarity_graph(
    chunks: Iterable[Mapping[str, Any]],
    *,
    embeddings: EmbeddingModel,
    threshold: float = 0.8,
) -> EvidenceGraph:
    """Embed a batch once and connect pairs meeting the signed cosine threshold."""
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be within [-1, 1]")
    evidence = EvidenceGraph(chunks, ())
    nodes = list(evidence._evidence.values())
    if len(nodes) < 2:
        return evidence
    batch = await embeddings.aembed_texts([node.text for node in nodes])
    if len(batch.vectors) != len(nodes) or len({len(v) for v in batch.vectors}) != 1:
        raise ModelOutputError("invalid_embedding")
    vectors = [unit_vector(v) for v in batch.vectors]
    pairs = []
    for a, b in combinations(range(len(nodes)), 2):
        dot = math.fsum(x * y for x, y in zip(vectors[a], vectors[b], strict=True))
        if dot >= threshold:
            pairs.append((nodes[a].chunk_id, nodes[b].chunk_id))
    evidence._set_relations(pairs)
    evidence._embedding_usage = batch.usage
    return evidence
