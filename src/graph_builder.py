"""Builds the knowledge graph (networkx) from extraction results, 
and detectscommunities (Leiden-style Louvain modularity clustering, 
matching the approach used by Microsoft's GraphRAG reference 
implementation) to enable "global search".
"""
from __future__ import annotations

import networkx as nx

from src.extraction import ExtractionResult


def build_graph(extraction: ExtractionResult) -> nx.Graph:
    graph = nx.Graph()

    for entity, entity_type in extraction.entity_type.items():
        mentions = extraction.mentions[entity]
        description = " ".join(dict.fromkeys(m.sentence for m in mentions))  # dedup, keep order
        description = description or entity  # fall back to the entity name if no mention text was found
        docs = sorted({m.doc_id for m in mentions})
        graph.add_node(
            entity,
            type=entity_type,
            description=description,
            mention_count=len(mentions),
            source_docs=docs,
        )

    for rel in extraction.relations:
        src, tgt = rel["source"], rel["target"]
        if src == tgt:
            continue
        if graph.has_edge(src, tgt):
            data = graph[src][tgt]
            data["weight"] += 1
            data["labels"].add(rel["label"])
            if len(data["evidence"]) < 5:
                data["evidence"].append(rel["sentence"])
        else:
            graph.add_edge(
                src,
                tgt,
                weight=1,
                labels={rel["label"]},
                evidence=[rel["sentence"]],
            )

    return graph


def detect_communities(graph: nx.Graph, seed: int = 42) -> dict[str, int]:
    """Partition the graph into communities and return node -> community_id."""
    if graph.number_of_edges() == 0:
        return {node: i for i, node in enumerate(graph.nodes)}

    communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=seed)
    node_to_community: dict[str, int] = {}
    for community_id, members in enumerate(communities):
        for node in members:
            node_to_community[node] = community_id
    return node_to_community
