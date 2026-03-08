# ============================================================
# graph_rag.py — Knowledge Graph + Hybrid Search
#
# APPROACH: spaCy NER + noun phrase co-occurrence
#
# WHY CO-OCCURRENCE WORKS:
#   If two entities appear in the same chunk, they are related
#   by definition — the document author placed them together.
#   Co-occurrence weight = how many chunks they share.
#   High weight = strong association. Domain-agnostic.
#
# TWO-LAYER ENTITY EXTRACTION:
#   Layer 1 — spaCy NER: PERSON, ORG, PRODUCT, LAW, GPE, EVENT
#             Works for: research, financial, medical, compliance,
#             legal, lab reports
#   Layer 2 — Noun phrase extraction: multi-word technical terms
#             spaCy NER misses (e.g. "multi-head attention",
#             "data subject", "knowledge distillation")
#             Works for: programming docs, novels, any domain
#
# BUILD TIME: 3-10 seconds for 4 PDFs
# ============================================================

import re
import networkx as nx
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ============================================================
# 1. SPACY LOADER
# ============================================================

_nlp = None

def get_nlp():
    """Lazy loader — downloads en_core_web_sm once if not present."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import spacy
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ============================================================
# 2. ENTITY EXTRACTION — Two Layers
# ============================================================

# spaCy entity types meaningful across all domains
_KEEP_LABELS = {
    "PERSON",      # researchers, authors, characters, executives
    "ORG",         # companies, institutions, universities, WHO
    "GPE",         # countries, cities — useful for compliance/legal
    "PRODUCT",     # AWS, Python, drug names, software tools
    "LAW",         # GDPR, Article 17, HIPAA
    "EVENT",       # clinical trials, product launches, conferences
    "WORK_OF_ART", # paper titles, book titles, report names
    "NORP",        # nationalities, religious/political groups
}

# Words spaCy sometimes tags as entities but are just noise
_NOISE = {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "first", "second", "third", "fourth", "fifth",
    "many", "some", "all", "each", "both", "any", "every",
    "this", "that", "these", "those", "such", "same",
    "new", "old", "large", "small", "high", "low",
    "figure", "table", "section", "chapter", "page",
    "et al", "ibid", "e.g", "i.e",
}


def extract_ner_entities(text: str) -> List[Tuple[str, str]]:
    """
    Layer 1: Named entity recognition via spaCy.
    Returns [(entity_text, entity_label), ...]
    """
    nlp = get_nlp()
    doc = nlp(text[:100000])
    entities = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ not in _KEEP_LABELS:
            continue
        clean = re.sub(r'\s+', ' ', ent.text.strip().lower())
        if len(clean) < 2 or clean.isdigit() or clean in _NOISE:
            continue
        if clean not in seen:
            seen.add(clean)
            entities.append((clean, ent.label_))
    return entities


def extract_noun_phrases(text: str, min_words: int = 2, max_words: int = 5) -> List[str]:
    """
    Layer 2: Multi-word noun phrases spaCy NER misses.

    Targets domain-specific compound terms:
      Research:    "multi-head attention", "knowledge distillation"
      Programming: "neural network", "decision tree", "linked list"
      Medical:     "hand hygiene", "infection prevention"
      Compliance:  "data subject", "lawful basis", "right to erasure"
      Financial:   "operating income", "free cash flow"
      Novel:       character concept phrases

    Only phrases appearing in 2+ chunks are kept (filtered in
    build_from_chunks, not here).
    """
    nlp = get_nlp()
    doc = nlp(text[:100000])
    phrases = []
    seen = set()
    for chunk in doc.noun_chunks:
        # Strip leading determiners: "the transformer" -> "transformer"
        phrase = chunk.text.strip().lower()
        phrase = re.sub(
            r'^(the|a|an|this|that|these|those|its|their|our|my)\s+', '', phrase
        )
        phrase = re.sub(r'\s+', ' ', phrase).strip()
        word_count = len(phrase.split())
        if word_count < min_words or word_count > max_words:
            continue
        if len(phrase) < 4 or phrase.isdigit() or phrase in _NOISE:
            continue
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


# ============================================================
# 3. KNOWLEDGE GRAPH
# ============================================================

class KnowledgeGraph:
    """
    Co-occurrence knowledge graph built from document chunks.

    Nodes : Named entities + frequent noun phrases
    Edges : Co-occurrence (two entities appear in the same chunk)
    Weight: Number of chunks they co-occur in
            Higher weight = stronger documented association

    Build: 3-10 seconds, 0 API calls, deterministic.
    """

    def __init__(self):
        self.graph = nx.Graph()  # Undirected — co-occurrence has no direction
        self.entity_sources: Dict[str, List[Dict]] = defaultdict(list)
        self._phrase_freq: Dict[str, int] = defaultdict(int)

    def build_from_chunks(self, chunks: List) -> None:
        """
        Build co-occurrence graph from document chunks.

        Two passes:
          Pass 1: Count noun phrase frequencies globally.
                  Keep only phrases appearing in 2+ chunks.
          Pass 2: Build nodes and weighted co-occurrence edges.
        """
        print(f"[GraphRAG] Building graph from {len(chunks)} chunks...")

        # Pass 1 — count phrase frequencies
        chunk_phrases = []
        for chunk in chunks:
            phrases = extract_noun_phrases(chunk.page_content)
            chunk_phrases.append(phrases)
            for p in set(phrases):
                self._phrase_freq[p] += 1

        frequent = {p for p, c in self._phrase_freq.items() if c >= 2}

        # Pass 2 — build graph
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page   = chunk.metadata.get("page", 0)
            text   = chunk.page_content

            ner    = extract_ner_entities(text)
            np_    = [p for p in chunk_phrases[i] if p in frequent]

            # All entities for this chunk (deduplicated)
            all_ents = list({e[0] for e in ner} | set(np_))

            # Add/update nodes
            for ent in all_ents:
                if self.graph.has_node(ent):
                    self.graph.nodes[ent]["mention_count"] += 1
                else:
                    label = next((lab for txt, lab in ner if txt == ent), "CONCEPT")
                    self.graph.add_node(ent, label=label, mention_count=1, sources=[])

                src_ref = {"source": source, "page": page}
                if src_ref not in self.entity_sources[ent]:
                    self.entity_sources[ent].append(src_ref)
                    self.graph.nodes[ent]["sources"].append(src_ref)

            # Add co-occurrence edges for every pair in this chunk
            for j in range(len(all_ents)):
                for k in range(j + 1, len(all_ents)):
                    a, b = all_ents[j], all_ents[k]
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]["weight"] += 1
                    else:
                        self.graph.add_edge(a, b, weight=1, sources=[source])

        print(
            f"[GraphRAG] Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def traverse(
        self,
        query: str,
        max_hops: int = 2,
        top_k_entities: int = 5,
        min_edge_weight: int = 1,
    ) -> Dict:
        """
        Find entities and associations relevant to the query.

        1. Extract entities/phrases from query
        2. Match against graph nodes (exact + substring)
        3. BFS traversal up to max_hops
        4. Collect strongest edges from traversal
        5. Format as evidence string for LLM context

        min_edge_weight: raise to 2-3 for cleaner, more confident
        paths at the cost of some coverage.
        """
        if self.graph.number_of_nodes() == 0:
            return {"found": False, "evidence": "", "sources": []}

        query_terms = list(set(
            [e[0] for e in extract_ner_entities(query)] +
            extract_noun_phrases(query)
        ))
        query_lower = query.lower()
        graph_nodes = list(self.graph.nodes())

        # Match query terms to graph nodes
        matched = set()
        for term in query_terms:
            if term in self.graph:
                matched.add(term)
        for node in graph_nodes:
            if node in query_lower:
                matched.add(node)
            else:
                for term in query_terms:
                    if node in term or term in node:
                        matched.add(node)

        if not matched:
            return {
                "found": False,
                "query_entities": query_terms,
                "matched_nodes": [],
                "evidence": "",
                "sources": [],
            }

        # BFS traversal
        traversed = set(matched)
        edges = []
        sources = []

        for start in list(matched)[:top_k_entities]:
            try:
                reachable = nx.single_source_shortest_path_length(
                    self.graph, start, cutoff=max_hops
                )
                traversed.update(reachable.keys())
                for u, v, data in self.graph.edges(reachable.keys(), data=True):
                    w = data.get("weight", 1)
                    if w >= min_edge_weight:
                        edges.append((u, w, v))
                    for src in data.get("sources", []):
                        entry = {"source": src, "page": "?"}
                        if entry not in sources:
                            sources.append(entry)
            except (nx.NetworkXError, nx.NodeNotFound):
                continue

        for node in list(traversed)[:20]:
            for src in self.entity_sources.get(node, []):
                if src not in sources:
                    sources.append(src)

        evidence = self._format_evidence(list(matched), list(traversed), edges)

        return {
            "found": True,
            "query_entities": query_terms,
            "matched_nodes": list(matched),
            "traversed_nodes": list(traversed),
            "evidence": evidence,
            "sources": sources[:10],
        }

    def _format_evidence(
        self,
        matched: List[str],
        all_nodes: List[str],
        edges: List[Tuple],
        max_edges: int = 20,
    ) -> str:
        """Format traversal result as readable text for LLM context."""
        if not matched and not edges:
            return ""

        lines = ["[Knowledge Graph Evidence]"]

        if matched:
            lines.append("Matched Entities:")
            for node in matched[:8]:
                if node in self.graph:
                    label = self.graph.nodes[node].get("label", "CONCEPT")
                    count = self.graph.nodes[node].get("mention_count", 1)
                    lines.append(f"  • {node} ({label}, {count} mentions)")

        if edges:
            lines.append("\nStrongest Associations (by co-occurrence):")
            seen = set()
            for u, w, v in sorted(edges, key=lambda x: x[1], reverse=True)[:max_edges]:
                key = tuple(sorted([u, v]))
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  ({u}) ←→ ({v})  [co-occurs {w}x]")

        extra = [n for n in all_nodes if n not in matched]
        if extra:
            lines.append(f"\nRelated Entities: {', '.join(extra[:12])}")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Return graph statistics for sidebar monitoring."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4)
                       if self.graph.number_of_nodes() > 1 else 0,
            "top_entities": sorted(
                [
                    (n, self.graph.nodes[n].get("mention_count", 0))
                    for n in self.graph.nodes()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }

    def is_empty(self) -> bool:
        return self.graph.number_of_nodes() == 0


# ============================================================
# 4. HYBRID SEARCH COMBINER
# ============================================================

def combine_hybrid_results(
    vector_docs: List,
    graph_result: Dict,
    query: str,
) -> Tuple[List, str]:
    """
    Merge vector retrieval with graph evidence.

    If graph found relevant entities:
      - graph evidence is prepended to LLM context (via format_docs)
      - vector docs are re-sorted: chunks mentioning matched
        entities come first (boosts the most graph-relevant chunks)

    If graph found nothing: returns vector_docs unchanged.
    """
    graph_evidence = ""

    if graph_result.get("found") and graph_result.get("evidence"):
        graph_evidence = graph_result["evidence"]
        matched = set(graph_result.get("matched_nodes", []))
        if matched:
            vector_docs = sorted(
                vector_docs,
                key=lambda d: sum(
                    1 for node in matched if node in d.page_content.lower()
                ),
                reverse=True,
            )

    return vector_docs, graph_evidence
