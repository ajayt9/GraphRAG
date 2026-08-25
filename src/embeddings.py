"""Retrieval vector indexes. Supports two selectable modes:

  - "tfidf" (default): TF-IDF cosine similarity via scikit-learn. Fully offline,
    no torch/sentence-transformers dependency needed.
  - "azure_embedding": real semantic embeddings via Azure OpenAI
    (AZURE_OPENAI_EMBEDDING_DEPLOYMENT, e.g. "text-embedding-3-small").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RETRIEVAL_METHODS = ("tfidf", "azure_embedding")


@dataclass
class VectorIndex:
    ids: list[str]  # parallel to matrix rows
    matrix: np.ndarray
    mode: str = "tfidf"
    vectorizer: TfidfVectorizer | None = None

    def _query_vector(self, text: str) -> np.ndarray:
        if self.mode == "azure_embedding":
            from src.llm import embed_texts  # lazy import: keep TF-IDF path Azure-free

            return np.array(embed_texts([text]))
        return self.vectorizer.transform([text])

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        query_vec = self._query_vector(text)
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in ranked if sims[i] > 0]


def build_index(id_to_text: dict[str, str], mode: str = "tfidf") -> VectorIndex:
    if mode not in RETRIEVAL_METHODS:
        raise ValueError(f"mode must be one of {RETRIEVAL_METHODS}, got {mode!r}")

    ids = list(id_to_text.keys())
    texts = [id_to_text[i] for i in ids]
    if not texts:
        return VectorIndex(ids=[], matrix=np.zeros((0, 0)), mode=mode)

    if mode == "azure_embedding":
        from src.llm import embed_texts

        # The embeddings API rejects empty strings; fall back to the id as a
        # last resort so a blank description/summary can't blow up the call.
        safe_texts = [text.strip() or ids[i] for i, text in enumerate(texts)]
        matrix = np.array(embed_texts(safe_texts))
        return VectorIndex(ids=ids, matrix=matrix, mode=mode)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return VectorIndex(ids=ids, matrix=matrix, mode="tfidf", vectorizer=vectorizer)
