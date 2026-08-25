"""End-to-end pipeline: load documents -> extract entities/relations -> build
graph -> detect communities -> summarize -> build retrieval indexes.

Call `build_index()` to get a fully assembled `GraphRAGIndex` ready for querying.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import networkx as nx

from src.embeddings import RETRIEVAL_METHODS, VectorIndex, build_index as build_vector_index
from src.extraction import ExtractionResult, extract, extract_llm
from src.graph_builder import build_graph, detect_communities
from src.summarizer import CommunitySummary, summarize_communities

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")

EXTRACTION_METHODS = ("regex", "llm")


@dataclass
class GraphRAGIndex:
    documents: dict[str, str]
    extraction: ExtractionResult
    graph: nx.Graph
    node_to_community: dict[str, int]
    community_summaries: dict[int, CommunitySummary]
    community_vector_index: VectorIndex
    entity_vector_index: VectorIndex
    sentence_vector_index: VectorIndex


def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> dict[str, str]:
    documents = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            documents[doc_id] = f.read()
    if not documents:
        raise FileNotFoundError(f"No .txt documents found in {data_dir}")
    return documents


def build_index(
    data_dir: str = DEFAULT_DATA_DIR,
    extraction_method: str = "regex",
    retrieval_method: str = "tfidf",
) -> GraphRAGIndex:
    if extraction_method not in EXTRACTION_METHODS:
        raise ValueError(f"extraction_method must be one of {EXTRACTION_METHODS}, got {extraction_method!r}")
    if retrieval_method not in RETRIEVAL_METHODS:
        raise ValueError(f"retrieval_method must be one of {RETRIEVAL_METHODS}, got {retrieval_method!r}")

    documents = load_documents(data_dir)
    extraction = extract_llm(documents) if extraction_method == "llm" else extract(documents)
    graph = build_graph(extraction)
    node_to_community = detect_communities(graph)
    community_summaries = summarize_communities(graph, node_to_community)

    community_vector_index = build_vector_index(
        {str(cid): summary.summary for cid, summary in community_summaries.items()}, mode=retrieval_method
    )
    entity_vector_index = build_vector_index(
        {node: data.get("description", node) for node, data in graph.nodes(data=True)}, mode=retrieval_method
    )
    sentence_vector_index = build_vector_index(
        {str(s["idx"]): s["text"] for s in extraction.sentences}, mode=retrieval_method
    )

    return GraphRAGIndex(
        documents=documents,
        extraction=extraction,
        graph=graph,
        node_to_community=node_to_community,
        community_summaries=community_summaries,
        community_vector_index=community_vector_index,
        entity_vector_index=entity_vector_index,
        sentence_vector_index=sentence_vector_index,
    )
