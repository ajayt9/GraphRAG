"""Lightweight, dependency-free information extraction for the GraphRAG demo.

This module deliberately avoids heavyweight NLP models (spaCy/transformers) so the
whole demo can run fully offline with a tiny dependency footprint. Instead it uses:

  1. A sentence splitter (regex based).
  2. A proper-noun-span detector (capitalization heuristics) to find entity mentions.
  3. Suffix / prefix / small world-knowledge heuristics to type each entity
     (PERSON, ORG, GPE, EVENT).
  4. Sentence co-occurrence to derive relations between entities, with a small
     keyword lexicon used to label the relation when possible.

The output feeds directly into the graph builder.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

# --- Sentence splitting -----------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# Titles like "Dr." would otherwise be misread as a sentence boundary; strip the
# period before splitting so "Dr. Amara Osei" stays one sentence/entity span.
_ABBREVIATION_RE = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof)\.")


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace/newlines
    if not text:
        return []
    text = _ABBREVIATION_RE.sub(r"\1", text)
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# --- Entity span detection ---------------------------------------------------

_STOPWORDS_SENTENCE_START = {
    "The", "A", "An", "In", "On", "At", "After", "Before", "During", "Following",
    "Under", "Six", "Two", "Three", "He", "She", "They", "It", "His", "Her",
    "Together", "Critics", "In", "Six weeks",
}

_ORG_SUFFIXES = (
    "Dynamics", "Group", "Initiative", "Council", "Press", "Corp", "Corporation",
    "Inc", "Ltd", "Agency", "Bureau", "Consortium",
)

_EVENT_PREFIXES = ("Project", "Operation", "The Accord", "The Black Harbor Leak")

# Small "world knowledge" gazetteer, analogous to what a trained NER model would
# already know about real-world place names.
_KNOWN_PLACES = {
    "Geneva", "Singapore", "Norway", "Lagos", "Europe", "Black Harbor Facility",
}

_TITLES = ("Dr.", "Mr.", "Ms.", "Mrs.", "Chief", "CEO")

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z.]*")


def _tokenize(sentence: str) -> list[str]:
    return _TOKEN_RE.findall(sentence)


def _find_capitalized_spans(sentence: str) -> list[str]:
    """Find runs of consecutive capitalized tokens (candidate proper-noun spans)."""
    tokens = sentence.split()
    spans: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            spans.append(" ".join(current))
            current.clear()

    for i, raw_tok in enumerate(tokens):
        tok = raw_tok.strip(",;:()\"'")
        if tok.endswith("'s"):
            tok = tok[:-2]
        if not tok:
            flush()
            continue
        is_cap = tok[0].isupper() and tok[0].isalpha()
        # Skip a lone sentence-initial stopword (e.g. "The", "After") - but still
        # allow it to combine with a following capitalized word ("The Accord").
        if is_cap and i == 0 and tok in _STOPWORDS_SENTENCE_START:
            # peek ahead: only keep if followed immediately by another capitalized word
            if i + 1 < len(tokens) and tokens[i + 1][:1].isupper():
                current.append(tok)
            else:
                flush()
            continue
        if is_cap:
            current.append(tok)
        else:
            flush()
    flush()
    # strip trailing punctuation-only fragments and single stopwords left alone
    cleaned = []
    for span in spans:
        span = span.strip(" .,;:")
        if not span:
            continue
        if span in _STOPWORDS_SENTENCE_START:
            continue
        cleaned.append(span)
    return cleaned


def _classify(span: str) -> str:
    if any(span.endswith(suf) for suf in _ORG_SUFFIXES):
        return "ORG"
    if span in _KNOWN_PLACES or any(place in span for place in _KNOWN_PLACES):
        return "GPE"
    if span.startswith(_EVENT_PREFIXES) or span in _EVENT_PREFIXES:
        return "EVENT"
    if span.startswith("The "):
        return "EVENT"
    words = span.split()
    if len(words) >= 2:
        return "PERSON"
    return "OTHER"


def _normalize_entity(span: str) -> str:
    # Collapse leading title tokens like "Dr." into the canonical name form used
    # consistently across the corpus ("Dr. Amara Osei" everywhere already, so this
    # mostly just trims stray punctuation).
    return span.strip(" .,;:")


@dataclass
class Mention:
    text: str
    doc_id: str
    sentence: str
    sentence_idx: int


@dataclass
class ExtractionResult:
    entity_type: dict[str, str] = field(default_factory=dict)
    mentions: dict[str, list[Mention]] = field(default_factory=lambda: defaultdict(list))
    sentences: list[dict] = field(default_factory=list)  # {doc_id, idx, text, entities}
    relations: list[dict] = field(default_factory=list)  # {source, target, label, sentence, doc_id}


_RELATION_KEYWORDS = [
    ("founded", ["founded"]),
    ("leads", ["leads", "led by", "leading"]),
    ("employed_by", ["engineer on", "joined", "works for", "worked with"]),
    ("regulates", ["regulator", "regulators", "ordered", "certify", "certifying"]),
    ("contracted", ["contracted", "hired"]),
    ("testified_before", ["testified", "testify", "appeared before"]),
    ("investigates", ["investigat", "traced", "flagged"]),
    ("reports_on", ["reporting", "reported", "reporter", "journalist"]),
    ("shares_information_with", ["shared", "leaked", "disclose", "copies"]),
    ("located_in", ["located", "based in", "headquarters"]),
    ("part_of", ["part of", "consortium", "affiliated"]),
]


def _infer_relation_label(sentence: str) -> str:
    lowered = sentence.lower()
    for label, keywords in _RELATION_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return label
    return "associated_with"


MIN_ENTITY_MENTIONS = 2  # filter out one-off capitalization noise


def _build_alias_map(raw_counts: dict[str, int]) -> dict[str, str]:
    """Merge "The X" spans into "X" when the bare form also occurs, so a leading
    sentence-initial "The" doesn't create a duplicate node for the same entity.
    """
    alias_map: dict[str, str] = {}
    for span in list(raw_counts):
        if span.startswith("The "):
            bare = span[len("The "):]
            if bare in raw_counts:
                alias_map[span] = bare
    return alias_map


def extract(documents: dict[str, str]) -> ExtractionResult:
    """Extract entities, per-sentence mentions, and co-occurrence relations.

    `documents` maps doc_id -> raw text.
    """
    result = ExtractionResult()
    raw_counts: dict[str, int] = defaultdict(int)

    doc_sentences: dict[str, list[str]] = {
        doc_id: split_sentences(text) for doc_id, text in documents.items()
    }

    # Pass 1: gather candidate spans and counts.
    for doc_id, sentences in doc_sentences.items():
        for sent in sentences:
            for span in _find_capitalized_spans(sent):
                raw_counts[_normalize_entity(span)] += 1

    alias_map = _build_alias_map(raw_counts)
    resolved_counts: dict[str, int] = defaultdict(int)
    for span, count in raw_counts.items():
        resolved_counts[alias_map.get(span, span)] += count

    valid_entities = {e for e, c in resolved_counts.items() if c >= MIN_ENTITY_MENTIONS}

    # Pass 2: build sentence records, mentions, and relations using only valid entities.
    sent_global_idx = 0
    for doc_id, sentences in doc_sentences.items():
        for local_idx, sent in enumerate(sentences):
            raw_spans = [_normalize_entity(s) for s in _find_capitalized_spans(sent)]
            spans = [alias_map.get(s, s) for s in raw_spans]
            entities_in_sentence = sorted({s for s in spans if s in valid_entities})

            for ent in entities_in_sentence:
                result.entity_type.setdefault(ent, _classify(ent))
                result.mentions[ent].append(
                    Mention(text=ent, doc_id=doc_id, sentence=sent, sentence_idx=sent_global_idx)
                )

            result.sentences.append(
                {
                    "doc_id": doc_id,
                    "idx": sent_global_idx,
                    "text": sent,
                    "entities": entities_in_sentence,
                }
            )

            if len(entities_in_sentence) >= 2:
                label = _infer_relation_label(sent)
                for i in range(len(entities_in_sentence)):
                    for j in range(i + 1, len(entities_in_sentence)):
                        result.relations.append(
                            {
                                "source": entities_in_sentence[i],
                                "target": entities_in_sentence[j],
                                "label": label,
                                "sentence": sent,
                                "doc_id": doc_id,
                            }
                        )
            sent_global_idx += 1

    return result


# --- LLM-based extraction (alternative to the regex heuristics above) -------

_LLM_EXTRACTION_SYSTEM = (
    "You extract named entities and relations from a document to build a knowledge graph. "
    "Respond with ONLY valid JSON (no markdown fences, no commentary) matching this schema:\n"
    '{"entities": [{"name": "string", "type": "PERSON|ORG|GPE|EVENT|OTHER"}], '
    '"relations": [{"source": "string", "target": "string", "label": "string", "sentence": "string"}]}\n'
    "Rules: use short snake_case relation labels (e.g. leads, founded, contracted, leaked_to, "
    "investigates, located_in, testified_before, reports_on, associated_with). "
    "The 'sentence' field must be copied verbatim from the document (the sentence supporting the "
    "relation). 'source' and 'target' must exactly match a 'name' from the entities list. Only include "
    "entities that are actually mentioned in the text."
)


def _llm_extract_document(doc_id: str, text: str) -> dict:
    from src.llm import chat_complete  # lazy import: keep regex path free of Azure OpenAI dependency

    raw = chat_complete(system=_LLM_EXTRACTION_SYSTEM, user=f"Document ID: {doc_id}\n\n{text}").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM extraction for '{doc_id}' returned invalid JSON: {e}\nRaw response: {raw[:500]}")


def extract_llm(documents: dict[str, str]) -> ExtractionResult:
    """Entity/relation extraction using Azure OpenAI instead of regex heuristics.

    Calls the LLM once per document to identify entities and relations, then
    builds the same `ExtractionResult` shape as `extract()` so it is a drop-in
    replacement anywhere in the pipeline (graph building, summarization, etc.).
    """
    result = ExtractionResult()
    sent_global_idx = 0

    for doc_id, text in documents.items():
        sentences = split_sentences(text)
        parsed = _llm_extract_document(doc_id, text)

        entity_types: dict[str, str] = {}
        for e in parsed.get("entities", []):
            name = str(e.get("name", "")).strip()
            if not name:
                continue
            etype = str(e.get("type", "OTHER")).upper()
            entity_types[name] = etype if etype in {"PERSON", "ORG", "GPE", "EVENT"} else "OTHER"
        for name, etype in entity_types.items():
            result.entity_type.setdefault(name, etype)

        for sent in sentences:
            present = sorted(name for name in entity_types if name in sent)
            for name in present:
                result.mentions[name].append(
                    Mention(text=name, doc_id=doc_id, sentence=sent, sentence_idx=sent_global_idx)
                )
            result.sentences.append(
                {"doc_id": doc_id, "idx": sent_global_idx, "text": sent, "entities": present}
            )
            sent_global_idx += 1

        for rel in parsed.get("relations", []):
            src = str(rel.get("source", "")).strip()
            tgt = str(rel.get("target", "")).strip()
            if not src or not tgt or src == tgt or src not in entity_types or tgt not in entity_types:
                continue
            label = str(rel.get("label") or "associated_with")
            sentence = str(rel.get("sentence") or "").strip() or text[:200]
            result.relations.append(
                {"source": src, "target": tgt, "label": label, "sentence": sentence, "doc_id": doc_id}
            )

    return result
