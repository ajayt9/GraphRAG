"""Community summarization: produces a human-readable summary for each community
detected in the graph, using Azure OpenAI (see src/llm.py). There is no offline
fallback -- Azure OpenAI must be configured for this to work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from src.llm import chat_complete


@dataclass
class CommunitySummary:
    community_id: int
    title: str
    entities: list[str]
    entity_types: dict[str, str]
    key_relations: list[str]
    summary: str
    evidence_sentences: list[str] = field(default_factory=list)


def _key_relation_strings(graph: nx.Graph, members: set[str]) -> list[str]:
    lines = []
    for u, v, data in graph.edges(data=True):
        if u in members and v in members:
            labels = ", ".join(sorted(data.get("labels", [])))
            lines.append(f"{u} --[{labels}]--> {v} (mentioned {data['weight']}x)")
    return sorted(lines, key=lambda s: s, reverse=False)


def _collect_evidence(graph: nx.Graph, members: set[str], limit: int = 8) -> list[str]:
    seen: list[str] = []
    for u, v, data in graph.edges(data=True):
        if u in members and v in members:
            for sent in data.get("evidence", []):
                if sent not in seen:
                    seen.append(sent)
    return seen[:limit]


def _llm_summary(title: str, entities: list[str], relations: list[str], evidence: list[str]) -> str:
    prompt = (
        "Summarize the following knowledge-graph community in 3-5 sentences, "
        "written for an analyst. Only use the facts given.\n\n"
        f"Entities: {', '.join(entities)}\n"
        f"Relations:\n" + "\n".join(relations[:10]) + "\n\n"
        f"Evidence sentences:\n" + "\n".join(evidence[:8])
    )
    return chat_complete(
        system="You write concise, factual summaries of knowledge-graph communities.",
        user=prompt,
    )


def summarize_communities(graph: nx.Graph, node_to_community: dict[str, int]) -> dict[int, CommunitySummary]:
    community_members: dict[int, set[str]] = {}
    for node, cid in node_to_community.items():
        community_members.setdefault(cid, set()).add(node)

    summaries: dict[int, CommunitySummary] = {}
    for cid, members in community_members.items():
        # Rank entities within the community by degree (within-community connections first).
        ranked = sorted(
            members,
            key=lambda n: graph.degree(n, weight="weight"),
            reverse=True,
        )
        top_entities = ranked[:2] if len(ranked) >= 2 else ranked
        title = " & ".join(top_entities) if top_entities else f"Community {cid}"

        relations = _key_relation_strings(graph, members)
        evidence = _collect_evidence(graph, members)

        text = _llm_summary(title, ranked, relations, evidence)

        summaries[cid] = CommunitySummary(
            community_id=cid,
            title=title,
            entities=ranked,
            entity_types={e: graph.nodes[e].get("type", "OTHER") for e in ranked},
            key_relations=relations,
            summary=text,
            evidence_sentences=evidence,
        )

    return summaries
