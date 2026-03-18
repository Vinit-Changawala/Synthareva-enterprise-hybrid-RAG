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
    "PERSON",
    "ORG",
    "GPE",
    "PRODUCT",
    "LAW",
    "EVENT",
    "WORK_OF_ART",
    "NORP",
    # ── Added for Finance / Medical / Policy ──
    "MONEY",       # $383B revenue, €20M fine
    "PERCENT",     # 7% HbA1c, 2% inflation
    "QUANTITY",    # 2°C limit, 100k units
}

_NOISE = {
    # Numbers and ordinals (existing)
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "first", "second", "third", "fourth", "fifth",
    "many", "some", "all", "each", "both", "any", "every",
    "this", "that", "these", "those", "such", "same",
    "new", "old", "large", "small", "high", "low",
    "figure", "table", "section", "chapter", "page",
    "et al", "ibid", "e.g", "i.e",
    # ── Domain-generic words added ─────────────
    # Medical — appears in every clinical chunk
    "study", "studies", "analysis", "result", "results", "finding",
    "treatment", "level", "levels", "patient", "patients", "group",
    "data", "evidence", "outcome", "outcomes", "review",
    # Legal/Policy — appears in every regulatory chunk
    "provision", "article", "paragraph", "member", "state", "party",
    "regulation", "directive", "measure", "requirement", "obligation",
    # Finance — appears in every annual report chunk
    "period", "quarter", "year", "fiscal", "total", "amount",
    "increase", "decrease", "change", "basis", "rate",
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

# ── Regex patterns for domain term detection ─────────────────────────────
# ALL-CAPS acronyms:   GDPR, WHO, ICU, BERT, NASA, HIPAA, EBITDA
_ACRONYM_RE = re.compile(r'\b[A-Z]{2,10}\b')
# Hyphenated terms:    COVID-19, HbA1c, multi-hop, cross-encoder
_HYPHEN_RE  = re.compile(r'\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b')
# Mixed-case clinical: HbA1c, mRNA, eGFR, kDa
_MIXED_RE   = re.compile(r'\b[A-Za-z][a-z]*[A-Z0-9][A-Za-z0-9]{1,}\b')

# Single-word terms too broad to be useful
_SINGLE_NOISE = {
    "COVID", "AI", "IT", "HR", "PR", "US", "UK", "EU", "UN",
    "Fig", "Eq", "Sec", "Vol", "No",
    "Inc", "Ltd", "LLC", "Corp",
}


def extract_domain_terms(text: str) -> List[Tuple[str, str]]:
    """
    Layer 3: Single-word domain-specific terms missed by NER and noun phrases.

    Catches:
      ALL-CAPS acronyms  → GDPR, HbA1c, BERT, EBITDA, ICU, HIPAA
      Hyphenated terms   → COVID-19, multi-head, cross-encoder, non-profit
      Mixed-case clinical→ mRNA, eGFR, kDa

    These are the exact terms that dominate medical/policy query failures —
    1-word concepts that noun phrases (needs 2+ words) and NER both miss.

    Returns [(term, label), ...]
    """
    terms = []
    seen  = set()

    for pattern, label in [
        (_ACRONYM_RE, "ACRONYM"),
        (_HYPHEN_RE,  "COMPOUND"),
        (_MIXED_RE,   "MIXED_CASE"),
    ]:
        for match in pattern.finditer(text):
            raw   = match.group()
            clean = raw.strip().lower()
            if (clean in seen
                    or clean in _NOISE
                    or raw in _SINGLE_NOISE
                    or len(clean) < 2
                    or clean.isdigit()):
                continue
            seen.add(clean)
            terms.append((clean, label))

    return terms

# ============================================================
# 2b. Noun Phrase Extractor
# Catches domain concepts that NER misses:
#   Medical: "insulin resistance", "HbA1c target", "cardiovascular risk"
#   Legal:   "data controller", "personal data", "high-risk AI system"
#   Policy:  "national sovereignty", "cultural diversity", "carbon credit"
# ============================================================
def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract meaningful noun phrases from text as graph nodes.
    Complements NER — catches concepts, not just named entities.
    Filters: 2-4 words, no pure stopwords, min 5 chars.
    """
    nlp = get_nlp()
    doc = nlp(text[:100000])

    # spaCy stopwords to filter trivial phrases
    stopwords = nlp.Defaults.stop_words

    phrases = []
    seen = set()

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        phrase = re.sub(r'\s+', ' ', phrase)

        # Filter: 2-4 words, at least 5 chars, not pure stopword
        words = phrase.split()
        if len(words) < 2 or len(words) > 4:
            continue
        if len(phrase) < 5:
            continue
        # Skip if all content words are stopwords
        content_words = [w for w in words if w not in stopwords]
        if not content_words:
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

        THREE passes:
        Pass 1: Count noun phrase AND domain term frequencies globally.
                Dynamic threshold scales with corpus size.
        Pass 2: Build nodes and weighted co-occurrence edges using
                all three layers (NER + noun phrases + domain terms).
        Pass 3: Prune hyper-frequent nodes connected to >60% of all
                nodes — these are domain-generic words that survived
                the _NOISE filter but still add no signal.
        """
        print(f"[GraphRAG] Building graph from {len(chunks)} chunks...")

        # Pass 1 — count phrase + domain term frequencies
        chunk_phrases = []
        chunk_domain  = []

        for chunk in chunks:
            phrases = extract_noun_phrases(chunk.page_content)
            domains = extract_domain_terms(chunk.page_content)
            chunk_phrases.append(phrases)
            chunk_domain.append([t for t, _ in domains])

            for p in set(phrases):
                self._phrase_freq[p] += 1
            for t, _ in domains:
                self._phrase_freq[t] += 1

        # ── Dynamic frequency threshold ───────────────────────────────────
        # OLD: hardcoded >= 2 killed unique concepts in large documents.
        # A 300-page WHO guideline sampled to 200 chunks means
        # "insulin resistance" appears in only 1 chunk → was filtered out.
        #
        #   <100 chunks  → threshold = 2  (small corpus, need repetition)
        #   ≥100 chunks  → threshold = 1  (large corpus, keep everything sampled)
        n = len(chunks)
        freq_threshold = 2 if n < 100 else 1
        frequent = {p for p, c in self._phrase_freq.items() if c >= freq_threshold}

        # Pass 2 — build graph with all three extraction layers
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page   = chunk.metadata.get("page", 0)
            text   = chunk.page_content

            ner     = extract_ner_entities(text)
            np_     = [p for p in chunk_phrases[i] if p in frequent]
            domain_ = [t for t in chunk_domain[i]  if t in frequent]

            # Merge all three layers, deduplicate by text
            seen_ents: set = set()
            all_ents: List[Tuple[str, str]] = []
            for txt, lbl in ner:
                if txt not in seen_ents:
                    seen_ents.add(txt)
                    all_ents.append((txt, lbl))
            for txt in np_:
                if txt not in seen_ents:
                    seen_ents.add(txt)
                    all_ents.append((txt, "CONCEPT"))
            for txt in domain_:
                if txt not in seen_ents:
                    seen_ents.add(txt)
                    all_ents.append((txt, "DOMAIN_TERM"))

            ent_texts = [t for t, _ in all_ents]

            # Add/update nodes
            for ent_txt, ent_lbl in all_ents:
                if self.graph.has_node(ent_txt):
                    self.graph.nodes[ent_txt]["mention_count"] += 1
                else:
                    self.graph.add_node(
                        ent_txt, label=ent_lbl, mention_count=1, sources=[]
                    )
                src_ref = {"source": source, "page": page}
                if src_ref not in self.entity_sources[ent_txt]:
                    self.entity_sources[ent_txt].append(src_ref)
                    self.graph.nodes[ent_txt]["sources"].append(src_ref)

            # Co-occurrence edges for every pair in this chunk
            for j in range(len(ent_texts)):
                for k in range(j + 1, len(ent_texts)):
                    a, b = ent_texts[j], ent_texts[k]
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]["weight"] += 1
                    else:
                        self.graph.add_edge(a, b, weight=1, sources=[source])

        # Pass 3 — prune hyper-frequent noise nodes
        # A node connected to >60% of all nodes is a generic term
        # ("health care", "data processing", "member states") —
        # it connects everything without meaning anything specific.
        if self.graph.number_of_nodes() > 10:
            total_nodes     = self.graph.number_of_nodes()
            noise_threshold = total_nodes * 0.60
            to_remove = [
                n for n in self.graph.nodes()
                if self.graph.degree(n) > noise_threshold
            ]
            if to_remove:
                print(f"[GraphRAG] Pruning {len(to_remove)} hyper-frequent noise nodes")
                self.graph.remove_nodes_from(to_remove)

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

        # Extract all query terms from all three layers
        query_ner    = [e[0] for e in extract_ner_entities(query)]
        query_np     = extract_noun_phrases(query)
        query_domain = [t for t, _ in extract_domain_terms(query)]
        query_terms  = list(set(query_ner + query_np + query_domain))

        query_lower  = query.lower()
        graph_nodes  = list(self.graph.nodes())

        # ── Normalised node index (Fix Gap 4) ───────────────────────────
        # Pre-build a lookup: normalised form → original node name
        # Normalisation: lowercase + remove hyphens + strip trailing 's'
        # This makes "data-controller" match "data controller",
        # "insulin-resistance" match "insulin resistance",
        # "controllers" match "controller".
        def _normalise(s: str) -> str:
            s = s.lower().replace("-", " ").replace("_", " ")
            s = re.sub(r'\s+', ' ', s).strip()
            # Strip common plural/possessive suffixes for matching
            if s.endswith("'s"):
                s = s[:-2]
            elif s.endswith("s") and len(s) > 4:
                s = s[:-1]  # "controllers" → "controller"
            return s

        norm_to_node: Dict[str, str] = {}
        for node in graph_nodes:
            norm_to_node[_normalise(node)] = node

        # Match query terms to graph nodes using normalised comparison
        matched: set = set()

        for term in query_terms:
            term_norm = _normalise(term)
            # Exact normalised match
            if term_norm in norm_to_node:
                matched.add(norm_to_node[term_norm])
            # Substring: term is contained in a node or vice versa
            for node in graph_nodes:
                node_norm = _normalise(node)
                if term_norm in node_norm or node_norm in term_norm:
                    matched.add(node)

        # Also scan raw query string for node substrings
        # (catches cases where query term wasn't extracted but node name
        #  appears verbatim in the query: "how does GDPR define X")
        query_norm = _normalise(query_lower)
        for node in graph_nodes:
            node_norm = _normalise(node)
            if node_norm in query_norm and len(node_norm) > 3:
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
