"""Streamlit UI for the GraphRAG demo.

Run with:
    streamlit run src/app.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from src.llm import is_llm_available, chat_complete
from src.pipeline import GraphRAGIndex, build_index
from src.query_engine import answer

st.set_page_config(page_title="GraphRAG Demo", layout="wide")

_TYPE_COLORS = {
    "PERSON": "#4C9AFF",
    "ORG": "#F2994A",
    "GPE": "#27AE60",
    "EVENT": "#9B51E0",
    "OTHER": "#BDBDBD",
}

_COMMUNITY_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
]

SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents"
)


@st.cache_resource(show_spinner="Building knowledge graph from documents...")
def get_index(data_dir: str, extraction_method: str, retrieval_method: str) -> GraphRAGIndex:
    return build_index(data_dir=data_dir, extraction_method=extraction_method, retrieval_method=retrieval_method)


def save_uploaded_documents(files) -> str:
    """Write uploaded files to a temp dir keyed by content hash, so re-runs with
    the same upload reuse the cached index instead of rebuilding."""
    hasher = hashlib.sha256()
    for f in files:
        hasher.update(f.name.encode("utf-8"))
        hasher.update(f.getvalue())
    upload_dir = os.path.join(tempfile.gettempdir(), "graphrag_uploads", hasher.hexdigest()[:16])
    if not os.path.isdir(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
        for f in files:
            name = os.path.splitext(f.name)[0] + ".txt"
            with open(os.path.join(upload_dir, name), "wb") as out:
                out.write(f.getvalue())
    return upload_dir


def generate_example_questions(index: GraphRAGIndex) -> list[str]:
    """Derive example questions from whatever documents were actually loaded,
    instead of hard-coding text tied to one sample scenario."""
    questions: list[str] = []

    nodes_by_mentions = sorted(
        index.graph.nodes(data=True), key=lambda item: item[1].get("mention_count", 0), reverse=True
    )
    top_entities = [name for name, _ in nodes_by_mentions[:5]]
    if top_entities:
        questions.append(f"Who is {top_entities[0]} and what role do they play?")

    edges_by_weight = sorted(
        index.graph.edges(data=True), key=lambda item: item[2].get("weight", 0), reverse=True
    )
    if edges_by_weight:
        a, b, _ = edges_by_weight[0]
        questions.append(f"What is the relationship between {a} and {b}?")

    if len(top_entities) > 1:
        questions.append(f"What role did {top_entities[1]} play in the events described?")

    summaries = list(index.community_summaries.values())
    if summaries:
        questions.append(f"Summarize the key events involving {summaries[0].title}.")

    return questions


def render_graph(index: GraphRAGIndex, color_by: str) -> str:
    net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white", notebook=False)
    net.barnes_hut(gravity=-4000, spring_length=150)

    for node, data in index.graph.nodes(data=True):
        if color_by == "Community":
            cid = index.node_to_community.get(node, 0)
            color = _COMMUNITY_PALETTE[cid % len(_COMMUNITY_PALETTE)]
        else:
            color = _TYPE_COLORS.get(data.get("type", "OTHER"), "#BDBDBD")
        size = 10 + 4 * data.get("mention_count", 1)
        title = f"{node} ({data.get('type')})\nCommunity: {index.node_to_community.get(node)}\n{data.get('description', '')[:300]}"
        net.add_node(node, label=node, title=title, color=color, size=size)

    for u, v, data in index.graph.edges(data=True):
        labels = ", ".join(sorted(data.get("labels", [])))
        net.add_edge(u, v, value=data.get("weight", 1), title=labels)

    net.set_options(
        """
        {
          "physics": {"stabilization": {"iterations": 150}},
          "edges": {"color": {"color": "#888888"}, "smooth": false}
        }
        """
    )
    return net.generate_html(notebook=False)


def main() -> None:
    st.title("GraphRAG Demo")
    st.caption(
        "A GraphRAG pipeline: entity/relation extraction \u2192 knowledge graph "
        "\u2192 community detection \u2192 community summarization \u2192 global + local retrieval "
        "\u2192 answer synthesis via Azure OpenAI."
    )

    if not is_llm_available():
        st.error(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY, "
            "and AZURE_OPENAI_DEPLOYMENT in a .env file (see .env.example), then restart the app."
        )
        st.stop()

    data_dir = None
    extraction_label = "LLM (Azure OpenAI)"
    retrieval_label = "Azure OpenAI embeddings"

    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload .txt documents to build the knowledge graph",
            type=["txt"],
            accept_multiple_files=True,
        )
        if st.button("Load Sample Data"):
            st.session_state["use_sample_data"] = True

        if uploaded_files:
            st.session_state["use_sample_data"] = False
            data_dir = save_uploaded_documents(uploaded_files)
            st.caption(f"Using {len(uploaded_files)} uploaded document(s).")
        elif st.session_state.get("use_sample_data"):
            data_dir = SAMPLE_DATA_DIR
            st.caption(f"Using sample dataset ({len(os.listdir(SAMPLE_DATA_DIR))} document(s)).")

        if data_dir is not None:
            st.header("Extraction method")
            extraction_label = st.radio(
                "Entity/relation extraction",
                ["Regex (heuristic)", "LLM (Azure OpenAI)"],
                index=1,
                help=(
                    "Regex: fast, offline capitalization/keyword heuristics. "
                    "LLM: calls Azure OpenAI once per document to extract entities and relations."
                ),
            )
            st.header("Retrieval method")
            retrieval_label = st.radio(
                "Vector retrieval",
                ["TF-IDF", "Azure OpenAI embeddings"],
                index=1,
                help=(
                    "TF-IDF: offline keyword-overlap similarity (scikit-learn). "
                    "Azure OpenAI embeddings: semantic similarity using your "
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT (e.g. text-embedding-3-small)."
                ),
            )

    if data_dir is None:
        st.info(
            "Upload one or more .txt documents in the sidebar, or click "
            "'Load Sample Data' to try the demo with the built-in sample dataset."
        )
        st.stop()

    extraction_method = "llm" if extraction_label.startswith("LLM") else "regex"
    retrieval_method = "azure_embedding" if retrieval_label.startswith("Azure") else "tfidf"

    try:
        index = get_index(data_dir, extraction_method, retrieval_method)
    except Exception as e:
        st.error(f"Failed to build the index (community summaries require Azure OpenAI): {type(e).__name__}: {e}")
        st.stop()

    with st.sidebar:
        st.header("Index stats")
        st.caption(f"Extraction: {extraction_label} | Retrieval: {retrieval_label}")
        st.metric("Documents", len(index.documents))
        st.metric("Entities", index.graph.number_of_nodes())
        st.metric("Relations", index.graph.number_of_edges())
        st.metric("Communities", len(index.community_summaries))
        st.divider()
        st.success("Azure OpenAI configured.")
        if st.button("Test LLM connection"):
            try:
                reply = chat_complete(system="You are a test.", user="Reply with the single word: OK")
                st.success(f"LLM call succeeded. Response: {reply!r}")
            except Exception as e:
                st.error(f"LLM call failed: {type(e).__name__}: {e}")

    tab_graph, tab_ask, tab_communities = st.tabs(["Explore graph", "Ask a question", "Communities"])

    with tab_ask:
        st.subheader("Ask a question about the dataset")
        example_questions = generate_example_questions(index)
        chosen = st.selectbox("Example questions", ["(type your own below)"] + example_questions)
        default_text = "" if chosen == "(type your own below)" else chosen
        query = st.text_input(
            "Your question", value=default_text, placeholder="Ask anything about the uploaded documents..."
        )

        if st.button("Ask", type="primary") and query.strip():
            with st.spinner("Retrieving from graph + generating answer..."):
                try:
                    result, ctx = answer(index, query)
                except Exception as e:
                    st.error(f"Answer generation failed: {type(e).__name__}: {e}")
                    st.stop()
            st.markdown(result)
            with st.expander("Retrieved context (evidence trail)"):
                st.write("**Communities used:**", ctx.communities)
                st.write("**Entities used:**", ctx.entities)
                st.write("**Relation facts:**")
                for fact in ctx.relation_facts[:10]:
                    st.write(f"- {fact}")
                st.write("**Supporting sentences:**")
                for sent in ctx.sentences:
                    st.write(f"- {sent}")

    with tab_graph:
        st.subheader("Knowledge graph")
        color_by = st.radio("Color nodes by", ["Community", "Entity type"], horizontal=True)
        html = render_graph(index, color_by)
        components.html(html, height=650, scrolling=True)
        if color_by == "Entity type":
            st.write({k: v for k, v in _TYPE_COLORS.items()})

    with tab_communities:
        st.subheader("Detected communities (used for GraphRAG global search)")
        for cid, summary in sorted(index.community_summaries.items()):
            with st.expander(f"Community {cid}: {summary.title} ({len(summary.entities)} entities)"):
                st.write("**Entities:**", ", ".join(summary.entities))
                st.markdown(summary.summary)


if __name__ == "__main__":
    main()
