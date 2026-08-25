"""Command-line interface for the GraphRAG demo.

Usage:
    python -m src.cli "Who leaked the Project Chimera documents?"
    python -m src.cli --stats
    python -m src.cli --extract llm "Who leaked the Project Chimera documents?"
    python -m src.cli --retrieval azure_embedding "Who leaked the Project Chimera documents?"
"""
from __future__ import annotations

import sys

from src.llm import is_llm_available
from src.pipeline import EXTRACTION_METHODS, RETRIEVAL_METHODS, build_index
from src.query_engine import answer


def print_stats(index) -> None:
    g = index.graph
    print(f"Documents indexed : {len(index.documents)}")
    print(f"Entities (nodes)  : {g.number_of_nodes()}")
    print(f"Relations (edges) : {g.number_of_edges()}")
    print(f"Communities found : {len(index.community_summaries)}")
    print()
    for cid, summary in sorted(index.community_summaries.items()):
        print(f"--- Community {cid}: {summary.title} ---")
        print(f"Entities: {', '.join(summary.entities)}")
        print()


def main(argv: list[str]) -> int:
    if not is_llm_available():
        print(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY, "
            "and AZURE_OPENAI_DEPLOYMENT in a .env file (see .env.example)."
        )
        return 1

    extraction_method = "regex"
    if "--extract" in argv:
        i = argv.index("--extract")
        if i + 1 >= len(argv) or argv[i + 1] not in EXTRACTION_METHODS:
            print(f"--extract must be followed by one of {EXTRACTION_METHODS}")
            return 1
        extraction_method = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    retrieval_method = "tfidf"
    if "--retrieval" in argv:
        i = argv.index("--retrieval")
        if i + 1 >= len(argv) or argv[i + 1] not in RETRIEVAL_METHODS:
            print(f"--retrieval must be followed by one of {RETRIEVAL_METHODS}")
            return 1
        retrieval_method = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    print(
        f"Building GraphRAG index from data/documents "
        f"(extraction={extraction_method}, retrieval={retrieval_method}) ...\n"
    )
    try:
        index = build_index(extraction_method=extraction_method, retrieval_method=retrieval_method)
    except Exception as e:
        print(f"Failed to build index: {type(e).__name__}: {e}")
        return 1

    if not argv or argv[0] in ("--stats", "-s"):
        print_stats(index)
        if not argv:
            print("Provide a question as an argument to ask something, e.g.:")
            print('  python -m src.cli "What is Project Chimera?"')
        return 0

    query = " ".join(argv)
    print(f"Question: {query}\n")
    try:
        result, ctx = answer(index, query)
    except Exception as e:
        print(f"Answer generation failed: {type(e).__name__}: {e}")
        return 1
    print(result)
    print()
    print(f"(retrieved communities={ctx.communities}, entities={ctx.entities})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
