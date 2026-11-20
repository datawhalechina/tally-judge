"""Evidence relations for multi-hop synthesis.

Adapted from Ragas ec00339 (Apache-2.0), KnowledgeGraph.find_two_nodes_single_rel
and find_indirect_clusters. Caller-owned chunk IDs replace graph node UUIDs.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Any

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.usage import UsageSummary
from tallyjudge.interfaces.models import EmbeddingModel
from tallyjudge.plugins._vectors import unit_vector
from tallyjudge.plugins.synthesis._evidence import _Chunk


@dataclass(init=False)
class EvidenceGraph:
    """An owned snapshot of chunk records and undirected evidence pairs."""

    _records: dict[str, dict[str, Any]]
    _relations: tuple[tuple[str, str], ...]
    _embedding_usage: UsageSummary | None

    def __init__(
        self,
        chunks: Iterable[Mapping[str, Any]],
        relations: Iterable[tuple[str, str]],
        embedding_usage: UsageSummary | None = None,
    ):
        self._records = {}
        for row in chunks:
            key = row["chunk_id"]
            if key in self._records:
                raise ValueError(f"Duplicate chunk_id: {key!r}")
            self._records[key] = deepcopy(dict(row))
        self._set_relations(relations)
        self._embedding_usage = embedding_usage

    def _set_relations(self, relations: Iterable[tuple[str, str]]) -> None:
        pairs = set()
        for a, b in relations:
            if a not in self._records or b not in self._records:
                raise ValueError("Evidence relation refers to an unknown chunk")
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        self._relations = tuple(sorted(pairs))

    @cached_property
    def _evidence(self) -> dict[str, _Chunk]:
        """Validate the scoring fields once when a generation or embedding needs them."""
        return {key: _Chunk.model_validate(row) for key, row in self._records.items()}

    @property
    def chunks(self) -> tuple[Mapping[str, Any], ...]:
        """Return detached records, including original offsets and metadata."""
        return tuple(self)

    @property
    def relations(self) -> tuple[tuple[str, str], ...]:
        """Return the canonical undirected edges."""
        return self._relations

    @property
    def embedding_usage(self) -> UsageSummary | None:
        """Return usage from constructing a similarity graph."""
        return self._embedding_usage

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
                    "chunk_id": key,
                    "document_id": str(metadata.get("source") or parent),
                    "text": props["page_content"],
                    "entities": props.get("entities", []),
                }
            )
        eligible = {row["chunk_id"] for row in chunks}
        pairs = (
            (r["source"], r["target"])
            for r in edges
            if r["type"] in relation_types
            and r["source"] in eligible
            and r["target"] in eligible
        )
        return cls(chunks, pairs)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return (deepcopy(row) for row in self._records.values())


def build_graph(
    chunks: Iterable[Mapping[str, Any]], *, indirect: bool = False
) -> EvidenceGraph:
    """Connect chunks sharing an explicit entity, optionally including Leiden community pairs."""
    from importlib import import_module

    nx = import_module("networkx")
    evidence = EvidenceGraph(chunks, ())
    graph = nx.Graph()
    entities = defaultdict(set)
    for key, record in evidence._records.items():
        for entity in record.get("entities", ()):
            label = entity.strip().casefold()
            if label:
                entities[label].add(key)
    for keys in entities.values():
        graph.add_edges_from(combinations(sorted(keys), 2))
    pairs = {tuple(sorted(edge)) for edge in graph.edges}
    if indirect and pairs:
        # Explicit integer indices keep "01" and "1" distinct across graph SDKs.
        names = sorted({key for edge in pairs for key in edge})
        indices = {key: i for i, key in enumerate(names)}
        adjacency = import_module("sknetwork.data").from_edge_list(
            [(indices[a], indices[b]) for a, b in sorted(pairs)],
            directed=False,
            matrix_only=True,
        )
        labels = (
            import_module("sknetwork.clustering")
            .Leiden(random_state=42)
            .fit_predict(adjacency)
        )
        groups = defaultdict(list)
        for key, label in zip(names, labels, strict=True):
            groups[int(label)].append(key)
        for members in groups.values():
            pairs.update(combinations(sorted(members), 2))
    evidence._set_relations(pairs)
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
