"""Scenario selection from Ragas ec00339 (Apache-2.0).

The three eligible query families share the archived independent ceiling quotas.
A run-local RNG and stable ordering replace process-global random and set iteration.
"""

import asyncio
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from hashlib import sha256
from itertools import product
from math import ceil, sumprod
from random import Random

from pydantic import BaseModel

from tallyjudge.plugins._prompting import StructuredPrompt
from tallyjudge.plugins.synthesis._planning_prompts import (
    ConceptCombinationPrompt,
    ConceptsList,
    PersonaGenerationPrompt,
    StringIO,
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)
from tallyjudge.plugins.synthesis._prompts import Persona
from tallyjudge.plugins.synthesis.graph import EvidenceGraph, EvidenceRelation
from tallyjudge.plugins.synthesis.scenarios import (
    EvidenceChunk,
    MultiHopScenario,
    QueryLength,
    QueryStyle,
    SingleHopScenario,
)
from tallyjudge.sdk.models import LanguageModel

SINGLE = "single_hop_specific_query_synthesizer"
ABSTRACT = "multi_hop_abstract_query_synthesizer"
SPECIFIC = "multi_hop_specific_query_synthesizer"
type Scenario = SingleHopScenario | MultiHopScenario


def _indirect_clusters(
    edges: Sequence[EvidenceRelation], n: int
) -> list[tuple[str, ...]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    unique_edges = set()
    for edge in edges:
        neighbors[edge.source].add(edge.target)
        if edge.bidirectional:
            neighbors[edge.target].add(edge.source)
        unique_edges.add(frozenset((edge.source, edge.target)))
    starts = sorted(neighbors)
    node_ids = "".join(s.replace("-", "") for s in starts)
    Random(int(sha256(node_ids.encode()).hexdigest()[:8], 16)).shuffle(starts)
    node_count = len(set().union(*unique_edges))
    limit = (n - 1) * 3 + 1 if len(unique_edges) < node_count else max(n, 10)
    by_start = deque()
    for start in starts[:limit]:
        paths: set[frozenset[str]] = set()

        def walk(key: str, path: frozenset[str]) -> None:
            if len(paths) > limit:
                return
            choices = neighbors.get(key, set()) - path
            if (len(path) == 3 or not choices) and len(path) > 1:
                paths.add(path)
            else:
                for nxt in sorted(choices):
                    walk(nxt, path | {nxt})

        walk(start, frozenset([start]))
        if paths:
            by_start.append(deque(sorted(paths, key=lambda p: tuple(sorted(p)))))
    chosen: list[frozenset[str]] = []
    while by_start and len(chosen) < n:
        group = by_start.popleft()
        path = group.popleft()
        if not any(path <= old for old in chosen):
            chosen = [old for old in chosen if not old < path]
            chosen.append(path)
        if group:
            by_start.append(group)
    return [tuple(sorted(path)) for path in chosen]


def _groups(items: list | dict) -> list[tuple[str, ...]]:
    return sorted({(item,) if isinstance(item, str) else tuple(item) for item in items})


def _representative_summaries(items: list[tuple[str, list[float]]]) -> list[str]:
    representatives = []
    visited = set()
    for i, (summary, vector) in enumerate(items):
        if i in visited:
            continue
        group = [summary]
        for j in range(i + 1, len(items)):
            other, other_vector = items[j]
            if sumprod(vector, other_vector) > 0.75:
                visited.add(j)
                group.append(other)
        representatives.append(max(group, key=len))
    return representatives


class ScenarioPlanner:
    def __init__(self, graph: EvidenceGraph, llm: LanguageModel, rng: Random):
        self.rows = {row["chunk_id"]: row for row in graph}
        if any("node_type" not in row for row in self.rows.values()):
            raise ValueError(
                "This graph has no node types. Use build_chunk_graph or import a Ragas graph."
            )
        self.nodes = {
            k: EvidenceChunk(chunk_id=k, document_id=r["document_id"], text=r["text"])
            for k, r in self.rows.items()
        }
        self.llm = llm
        self.rng = rng
        self.children: dict[str, list[str]] = defaultdict(list)
        self.overlaps = []
        self.similarities = []
        for link in graph.relationships:
            if link.type == "child":
                self.children[link.source].append(link.target)
            if link.type == "entities_overlap" and link.source != link.target:
                self.overlaps.append(link)
            if link.properties.get("summary_similarity"):
                self.similarities.append(link)
        counts = Counter(
            row["node_type"]
            for row in self.rows.values()
            if row.get("entities") is not None
        )
        kind = "chunk" if counts["chunk"] > counts["document"] else "document"
        self.single_nodes = [
            key for key, row in self.rows.items() if row["node_type"] == kind
        ]

    async def _ask[I: BaseModel, O: BaseModel](
        self, template: StructuredPrompt[I, O], data: I
    ) -> O:
        return (
            await self.llm.agenerate(template.render(data), template.output_model)
        ).data

    async def personas(self, maximum: int) -> list[Persona]:
        summaries = [
            (row["summary"], row["summary_embedding"])
            for row in self.rows.values()
            if row.get("summary_embedding") is not None
        ]
        if not summaries:
            raise ValueError(
                "Automatic persona generation requires a graph with summary embeddings"
            )
        representatives = await asyncio.to_thread(_representative_summaries, summaries)
        count = min(maximum, len(summaries))
        representatives += self.rng.choices(
            representatives, k=max(0, count - len(representatives))
        )
        return [
            await self._ask(PersonaGenerationPrompt(), StringIO(text=text))
            for text in representatives[:count]
        ]

    async def _mapping(
        self, themes: list[str], personas: list[Persona]
    ) -> dict[str, list[str]]:
        reply = await self._ask(
            ThemesPersonasMatchingPrompt(),
            ThemesPersonasInput(themes=themes, personas=personas),
        )
        return reply.mapping

    def _sample(
        self,
        keys: Sequence[str],
        groups: list[tuple[str, ...]],
        personas: list[Persona],
        mapping: dict[str, list[str]],
        *,
        field: str,
        count: int,
        single: bool = False,
    ) -> list[Scenario]:
        by_name = {p.name: p for p in reversed(personas)}
        options = []
        all_terms = {s.lower() for group in groups for s in group}
        for group in groups:
            terms = {s.lower() for s in group}
            allowed = [
                by_name[name]
                for name, labels in mapping.items()
                if (all_terms if single else terms).intersection(
                    s.lower() for s in labels
                )
            ]
            nodes = [
                self.nodes[k]
                for k in keys
                if single
                or terms.intersection(s.lower() for s in self.rows[k].get(field, []))
            ]
            if nodes:
                options.extend(
                    (group, nodes, p, style, length)
                    for p, style, length in product(allowed, QueryStyle, QueryLength)
                )
        self.rng.shuffle(options)
        result = []
        for group, nodes, persona, style, length in options[:count]:
            if single:
                scenario = SingleHopScenario(
                    term=group[0],
                    nodes=nodes,
                    persona=persona,
                    style=style,
                    length=length,
                )
            else:
                scenario = MultiHopScenario(
                    combinations=list(group),
                    nodes=nodes,
                    persona=persona,
                    style=style,
                    length=length,
                )
            result.append(scenario)
        return result

    async def _single(self, n: int, personas: list[Persona]) -> list[Scenario]:
        scenarios = []
        each = ceil(n / len(self.single_nodes))
        for key in self.single_nodes:
            if len(scenarios) >= n:
                break
            themes = sorted(
                {
                    s
                    for group in _groups(self.rows[key].get("entities", []))
                    for s in group
                }
            )
            if themes:
                mapping = await self._mapping(themes, personas)
                scenarios.extend(
                    self._sample(
                        [key],
                        [(s,) for s in themes],
                        personas,
                        mapping,
                        field="entities",
                        count=each,
                        single=True,
                    )
                )
        return scenarios

    async def _specific(self, n: int, personas: list[Persona]) -> list[Scenario]:
        scenarios = []
        each = ceil(n / len(self.overlaps))
        for link in self.overlaps:
            if len(scenarios) >= n:
                break
            groups = _groups(link.properties["overlapped_items"])
            if groups:
                themes = sorted({s for group in groups for s in group})
                mapping = await self._mapping(themes, personas)
                scenarios.extend(
                    self._sample(
                        sorted((link.source, link.target)),
                        groups,
                        personas,
                        mapping,
                        field="entities",
                        count=each,
                    )
                )
        return scenarios

    async def _abstract(self, n: int, personas: list[Persona]) -> list[Scenario]:
        clusters = await asyncio.to_thread(_indirect_clusters, self.similarities, n)
        scenarios = []
        each = ceil(n / len(clusters))
        for cluster in clusters:
            if len(scenarios) >= n:
                break
            keys = [child for key in cluster for child in self.children.get(key, [key])]
            concepts = await self._ask(
                ConceptCombinationPrompt(),
                ConceptsList(
                    lists_of_concepts=[
                        self.rows[key].get("themes", []) for key in keys
                    ],
                    max_combinations=each,
                ),
            )
            mapping = await self._mapping(
                [s for group in concepts.combinations for s in group], personas
            )
            scenarios.extend(
                self._sample(
                    keys,
                    [tuple(group) for group in concepts.combinations],
                    personas,
                    mapping,
                    field="themes",
                    count=each,
                )
            )
        return scenarios

    async def select(
        self, size: int, supplied: Sequence[Persona], maximum: int
    ) -> list[tuple[str, Scenario]]:
        strategies = []
        if self.single_nodes:
            strategies.append((SINGLE, self._single))
        if self.similarities and await asyncio.to_thread(
            _indirect_clusters, self.similarities, 1
        ):
            strategies.append((ABSTRACT, self._abstract))
        if self.overlaps:
            strategies.append((SPECIFIC, self._specific))
        if not strategies:
            raise ValueError("No compatible query synthesizers for the provided graph")
        if supplied:
            personas = list(supplied)
            self.rng.shuffle(personas)
            personas = personas[:maximum]
        else:
            personas = await self.personas(maximum)
        quota = ceil(size * (1 / len(strategies)))
        result = []
        for name, choose in strategies:
            result.extend(
                (name, scenario) for scenario in await choose(quota, personas)
            )
        return result
