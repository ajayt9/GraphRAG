"""Query engine implementing GraphRAG-style global + local search, plus a final
answer synthesis step via Azure OpenAI. There is no offline/extractive fallback --
Azure OpenAI must be configured (see src/llm.py) for answers to be generated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.llm import chat_complete
from src.pipeline import GraphRAGIndex


@dataclass
class RetrievedContext:
    communities: list[int] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    relation_facts: list[str] = field(default_factory=list)


def global_search(index: GraphRAGIndex, query: str, top_k: int = 2) -> list[int]:
    hits = index.community_vector_index.query(query, top_k=top_k)
    return [int(cid) for cid, _score in hits]


def local_search(index: GraphRAGIndex, query: str, top_k: int = 5) -> list[str]:
    hits = index.entity_vector_index.query(query, top_k=top_k)
    return [entity for entity, _score in hits]


def _relation_facts_for_entities(index: GraphRAGIndex, entities: list[str]) -> list[str]:
    facts = []
    entity_set = set(entities)
    for u, v, data in index.graph.edges(data=True):
        if u in entity_set or v in entity_set:
            labels = ", ".join(sorted(data.get("labels", [])))
            facts.append(f"{u} --[{labels}]--> {v}")
    return facts


def retrieve(index: GraphRAGIndex, query: str) -> RetrievedContext:
    communities = global_search(index, query, top_k=2)
    entities = local_search(index, query, top_k=5)
    sentence_hits = index.sentence_vector_index.query(query, top_k=5)
    sentence_lookup = {str(s["idx"]): s["text"] for s in index.extraction.sentences}
    sentences = [sentence_lookup[sid] for sid, _score in sentence_hits]
    relation_facts = _relation_facts_for_entities(index, entities)

    return RetrievedContext(
        communities=communities,
        entities=entities,
        sentences=sentences,
        relation_facts=relation_facts,
    )


def _build_context_block(index: GraphRAGIndex, ctx: RetrievedContext) -> str:
    parts = []
    if ctx.communities:
        parts.append("=== Community summaries (global search) ===")
        for cid in ctx.communities:
            summary = index.community_summaries.get(cid)
            if summary:
                parts.append(f"[Community {cid}: {summary.title}]\n{summary.summary}")
    if ctx.relation_facts:
        parts.append("=== Relevant relations (local search) ===")
        parts.extend(ctx.relation_facts[:10])
    if ctx.sentences:
        parts.append("=== Supporting sentences ===")
        parts.extend(f"- {s}" for s in ctx.sentences)
    return "\n\n".join(parts)


def _llm_answer(query: str, context_block: str) -> str:
    return chat_complete(
        system=(
            "You are a GraphRAG assistant. Answer the user's question using ONLY the "
            "provided context (community summaries, graph relations, and source "
            "sentences). If the answer is not in the context, say so."
        ),
        user=f"Context:\n{context_block}\n\nQuestion: {query}",
    )


def answer(index: GraphRAGIndex, query: str) -> tuple[str, RetrievedContext]:
    ctx = retrieve(index, query)
    context_block = _build_context_block(index, ctx)
    result = _llm_answer(query, context_block)
    return result, ctx
