# ============================================================
# main_fv.py — Enterprise Hybrid RAG — Backend
#
# PIPELINE:
#   PDF → Load → Chunk (per-source cap) → Embed → Chroma
#   Query → [Compliance] Hybrid BM25+ANN → Rerank → LLM
#         → [Comparative] Per-doc retrieval → Rerank → LLM
#   Answer → Faithfulness check → Cache → Metrics
# ============================================================

import os
import hashlib
import tempfile
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_mistralai import ChatMistralAI

from graph_rag import KnowledgeGraph, combine_hybrid_results

load_dotenv()


# ============================================================
# RELEVANCE THRESHOLD
#
# Cross-encoder/ms-marco-MiniLM-L-6-v2 outputs raw logits.
# Score ranges (empirical):
#   > 5.0   highly relevant
#   0 to 5  moderately relevant
#   -5 to 0 low relevance — LLM will say "I don't know" naturally
#   < -5.0  genuinely irrelevant — fire early refusal
#
# -5.0 only fires when documents have nothing to do with the query.
# ============================================================
RELEVANCE_THRESHOLD = -5.0


# ============================================================
# 1. INGESTION
#
# Metadata stored per page:
#   source     — filename (used by PDF filter + per-doc retrieval)
#   page       — page number (used in citations)
#   word_count — used by post_filter to remove short fragments
# ============================================================

def load_pdfs(uploaded_files) -> List:
    """Load PDFs and attach minimal essential metadata."""
    docs = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pdf_docs = loader.load()

        for doc in pdf_docs:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["word_count"] = len(doc.page_content.split())
            # "page" is already set by PyPDFLoader — no override needed

        docs.extend(pdf_docs)
        os.remove(tmp_path)
    return docs


# ============================================================
# 2. CHUNKING — Per-Source Cap
#
# Per-source cap prevents large PDFs from dominating the index.
# A 200-page paper would otherwise contribute 10x more chunks
# than a 20-page paper, biasing BM25 and vector search.
#
# Metadata stored per chunk:
#   source      — inherited from page, used by per-doc retrieval
#   page        — inherited from page, used in citations
#   word_count  — used by post_filter
#   has_numbers — used by numerical filter (TYPE_C queries)
# ============================================================

def split_docs(docs: List, max_chunks_per_source: int = 200) -> List:
    """Split documents into chunks with per-source cap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )

    docs_by_source = defaultdict(list)
    for doc in docs:
        docs_by_source[doc.metadata.get("source", "unknown")].append(doc)

    all_chunks = []
    for source, source_docs in docs_by_source.items():
        chunks = splitter.split_documents(source_docs)

        for chunk in chunks:
            chunk.metadata["word_count"] = len(chunk.page_content.split())
            chunk.metadata["has_numbers"] = any(
                c.isdigit() for c in chunk.page_content
            )

        # Sample evenly across document — don't just truncate
        if len(chunks) > max_chunks_per_source:
            step = len(chunks) / max_chunks_per_source
            chunks = [chunks[int(i * step)] for i in range(max_chunks_per_source)]

        all_chunks.extend(chunks)
    return all_chunks


# ============================================================
# 3. VECTOR STORE
# ============================================================

@st.cache_resource
def create_vectorstore(collection_id: str, _chunks):
    """Create Chroma vectorstore with HNSW cosine index."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        collection_name=collection_id,
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
        collection_metadata={"hnsw:space": "cosine"},
    )
    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(_chunks)
    return vectorstore


# ============================================================
# 4. RETRIEVAL
#
# Two retrieval strategies — selected by sidebar mode:
#
# COMPLIANCE mode → get_hybrid_retriever()
#   Global BM25 (0.3) + ANN/MMR (0.7) across all selected PDFs.
#   Best for factual single-document questions.
#
# COMPARATIVE mode → retrieve_per_document()
#   Top-N chunks pulled from EACH selected PDF independently.
#   Guarantees every document is represented in context.
#   Solves the same-domain crowding problem: without this, a
#   query about "encoder" pulls 5 BERT chunks and 0 Transformer
#   chunks because BERT has higher keyword frequency.
# ============================================================

def get_hybrid_retriever(chunks: List, vectorstore, prefilter: Optional[Dict] = None):
    """Build BM25 + ANN ensemble retriever for compliance mode."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5

    search_kwargs = {"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    if prefilter:
        search_kwargs["filter"] = prefilter

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.3, 0.7],
    )


def retrieve_per_document(
    query: str,
    vectorstore,
    selected_sources: List[str],
    chunks_per_doc: int = 3,
) -> List:
    """
    Per-document retrieval for comparative mode.

    Retrieves top chunks_per_doc chunks from each PDF separately
    using a source-scoped Chroma filter, then merges the results.

    With 4 PDFs and chunks_per_doc=3, returns up to 12 chunks total
    (3 from each PDF). The cross-encoder reranker then selects the
    best 5 from that merged set.

    Args:
        query           : user question
        vectorstore     : Chroma vectorstore
        selected_sources: PDF filenames selected in sidebar
        chunks_per_doc  : chunks to fetch per PDF (default 3)
    """
    all_docs = []
    for source in selected_sources:
        try:
            source_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": chunks_per_doc,
                    "fetch_k": chunks_per_doc * 4,
                    "lambda_mult": 0.7,
                    "filter": {"source": {"$eq": source}},
                },
            )
            all_docs.extend(source_retriever.invoke(query))
        except Exception as e:
            print(f"[PerDocRetrieval] Failed for {source}: {e}")
    return all_docs


# ============================================================
# 4b. QUERY EXPANSION
#
# WHY THIS IS NEEDED:
#   The retriever matches words and vectors — it cannot know
#   that "pretraining data" and "same corpus" mean the same thing.
#   Expanding the query with paraphrases bridges this vocabulary gap.
#
# HOW IT WORKS:
#   1. LLM generates 3 alternative phrasings of the query
#   2. Retriever runs on original + all alternatives
#   3. Results are deduplicated by content hash
#   4. Union of all results goes to reranker (which then sorts by quality)
#
# COST: 1 extra LLM call per query (fast — small output)
# BENEFIT: catches chunks that use different vocabulary for the same concept
# ============================================================

EXPANSION_PROMPT = """Given this question, write 3 alternative phrasings that mean the same thing but use different words.

Question: {query}

Output exactly 3 lines. Each line is one alternative phrasing. No numbering, no labels, no explanation.
"""

def expand_query(query: str) -> List[str]:
    """
    Generate alternative phrasings of the query to improve retrieval coverage.

    Returns the original query + up to 3 paraphrases.
    Falls back to [original] silently if LLM fails.
    """
    try:
        llm = get_llm()
        raw = llm.invoke(
            [HumanMessage(content=EXPANSION_PROMPT.format(query=query))]
        ).content.strip()
        alternatives = [
            line.strip()
            for line in raw.split("\n")
            if line.strip() and line.strip() != query
        ][:3]
        return [query] + alternatives
    except Exception:
        return [query]   # silent fallback — original query still works


def retrieve_with_expansion(query: str, retriever) -> List:
    """
    Retrieve using original query + expanded paraphrases.
    Deduplicates by first-100-chars content hash so the reranker
    receives a clean, non-repetitive candidate set.

    Args:
        query    : original user question
        retriever: EnsembleRetriever (BM25 + ANN)

    Returns:
        Deduplicated union of results from all query variants.
        The cross-encoder reranker in the pipeline handles final ranking.
    """
    queries = expand_query(query)
    seen_hashes = set()
    all_docs = []

    for q in queries:
        try:
            docs = retriever.invoke(q)
            for doc in docs:
                # Hash on first 100 chars — fast, avoids near-duplicates
                doc_hash = hash(doc.page_content[:100])
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    all_docs.append(doc)
        except Exception as e:
            print(f"[QueryExpansion] Retrieval failed for variant '{q[:40]}': {e}")

    return all_docs


def post_filter(docs: List, min_word_count: int = 50) -> List:
    """
    Remove chunks shorter than min_word_count words.
    Eliminates page headers, footers, and single-line fragments
    that add noise without contributing meaningful content.
    """
    filtered = [
        doc for doc in docs
        if doc.metadata.get("word_count", len(doc.page_content.split())) >= min_word_count
    ]
    return filtered if filtered else docs[:3]


# ============================================================
# 5. CROSS-ENCODER RERANKER
#
# Runs after retrieval on a small candidate set (~10-15 docs).
# Scores each (query, chunk) pair with a relevance score.
# Returns top_k docs AND the top score for the threshold gate.
#
# Score 1.0 returned as safe fallback when reranker unavailable —
# 1.0 > RELEVANCE_THRESHOLD so the gate never fires accidentally.
# ============================================================

_reranker = None

def get_reranker():
    """Lazy loader — downloads ~80MB model once, then cached."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[Reranker] Failed to load: {e}. Using retrieval order.")
    return _reranker


def rerank_docs(query: str, docs: List, top_k: int = 5) -> Tuple[List, float]:
    """
    Rerank docs by cross-encoder relevance score.

    Returns:
        (ranked_docs, top_score)
        top_score is used by the threshold gate in app_fv.py.
    """
    reranker = get_reranker()
    if reranker is None or len(docs) <= 1:
        return docs[:top_k], 1.0
    if len(docs) <= top_k:
        return docs, 1.0

    try:
        scores = reranker.predict([(query, doc.page_content) for doc in docs])
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]], float(scored[0][0])
    except Exception as e:
        print(f"[Reranker] Scoring failed: {e}. Using retrieval order.")
        return docs[:top_k], 1.0


# ============================================================
# 6. CONTEXT FORMATTER
#
# Only source and page are included — these are the fields
# that matter for citations. doc_type removed (it was always
# "general" for research papers because infer_doc_type used
# filename keywords that rarely matched).
# ============================================================

def format_docs(docs: List, graph_evidence: str = "") -> str:
    """Format retrieved chunks into LLM context string."""
    parts = []
    if graph_evidence:
        parts.append(graph_evidence)
        parts.append("")
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ============================================================
# 7. PROMPTS
#
# COMPLIANCE PROMPT:
#   Single-document factual questions.
#   Strict: answer only from context, cite every fact, say
#   "I don't know" if context doesn't contain the answer.
#
# COMPARATIVE PROMPT:
#   Give the LLM the chunks from each document,
#   tell it to find what each document says about the question,
#   and synthesize an honest answer. If a document is genuinely
#   silent on the topic, the LLM notes that — but it does not
#   refuse to answer entirely.
# ============================================================

COMPLIANCE_SAFE_PROMPT = ChatPromptTemplate.from_template("""
You are a precise document assistant. Answer questions strictly from the provided context.

## Conversation History
{history}

## Document Context
{context}

## Question
{question}

## Rules
1. Use ONLY information explicitly stated in the context above.
2. If the context does not contain the answer, say exactly:
   "I don't know. The documents do not contain information about this."
3. If context partially answers the question, answer the covered part and note what is missing.
4. After every fact, cite its source: [Source: filename.pdf, Page: N]
5. Be concise and words needed to be accurate.
6. Do not infer, deduce, or add information beyond what is written.
7. You may have knowledge about these topics from training — ignore it entirely.
8. Answer only from the context provided above, even if you know the answer from elsewhere.
9. - Frame the answer in the most user friendly way possible.

Answer:
""")


COMPARATIVE_PROMPT = ChatPromptTemplate.from_template("""
You are a document analyst comparing information across multiple sources.

## Conversation History
{history}

## Document Context
{context}

## Question
{question}

## Instructions
You have been given relevant passages from multiple documents.
Your task is to answer the question by synthesizing what each document says.

Step 1 — For each document in the context, identify what it says about the question topic.
Step 2 — Write your answer by presenting each document's position, then the synthesis.

Answer format:
[Document: filename] — What this document says about the topic (cite page).
[Document: filename] — What this document says about the topic (cite page).
[Synthesis] — What the documents together tell us: similarities, differences, or connections.

If a document does not contain relevant information on the topic, state that briefly and move on.
Do not refuse to answer just because documents cover the topic differently.
A partial or one-sided answer is better than no answer.

Rules:
- Use only information explicitly in the context. Do not infer beyond what is written.
- Cite every fact: [Source: filename.pdf, Page: N]
- You may have knowledge about these topics from training — ignore it entirely.
- Base your answer only on the provided document context.
- Frame the answer in the most user friendly way possible.

Answer:
""")


# ============================================================
# 8. LLM
# ============================================================

@st.cache_resource
def load_llm():
    """
    Mistral AI — free tier, 1B tokens/month.
    open-mistral-nemo = 12B model, 128K context window.
    Get API key at: https://console.mistral.ai
    """
    return ChatMistralAI(
        model="open-mistral-nemo",
        temperature=0,
        max_tokens=512,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

def get_llm():
    return load_llm()


# ============================================================
# 9. RAG CHAIN
# ============================================================

def build_chat_rag_chain(mode: str):
    """Build RAG chain for compliance or comparative mode."""
    prompt = COMPLIANCE_SAFE_PROMPT if mode == "compliance" else COMPARATIVE_PROMPT
    return (
        {
            "context":  RunnableLambda(lambda x: x["context"]),
            "question": RunnableLambda(lambda x: x["question"]),
            "history":  RunnableLambda(lambda x: x["history"]),
        }
        | prompt
        | get_llm()
    )


# ============================================================
# 10. CACHING
# ============================================================

def get_query_hash(query: str) -> str:
    return hashlib.md5(" ".join(query.lower().strip().split()).encode()).hexdigest()

def check_answer_cache(query: str, cache: Dict) -> Optional[Dict]:
    return cache.get(get_query_hash(query))

def write_answer_cache(query: str, result: Dict, cache: Dict) -> None:
    cache[get_query_hash(query)] = result

def check_retrieval_cache(query: str, cache: Dict) -> Optional[List]:
    return cache.get(get_query_hash(query))

def write_retrieval_cache(query: str, docs: List, cache: Dict) -> None:
    cache[get_query_hash(query)] = docs


# ============================================================
# 11. FAITHFULNESS CHECK
#
# Evaluates whether the answer is grounded in the context.
# Strict YES/NO extraction handles verbose LLM responses.
# ============================================================

def faithfulness_check(answer: str, context: str) -> str:
    eval_prompt = f"""
You are a faithfulness evaluator. Judge whether the answer is supported by the context.

## Answer
{answer}

## Context
{context}

## Rules
RULE 1 — If the answer says "I don't know" or that documents don't contain the info → YES
RULE 2 — If every factual claim in the answer is directly supported by the context → YES
RULE 3 — If any claim cannot be found in the context → NO
RULE 4 — If the answer correctly notes what is covered and what is missing → YES

Respond with exactly one word: YES or NO
No explanation. No punctuation. Just YES or NO.
"""
    raw = get_llm().invoke([HumanMessage(content=eval_prompt)]).content.strip().upper()
    first = raw.split()[0] if raw.split() else ""
    if first in ("YES", "NO"):
        return first
    if "YES" in raw and "NO" not in raw:
        return "YES"
    if "NO" in raw and "YES" not in raw:
        return "NO"
    return "NO"


# ============================================================
# 12. SUPPORTING DOCS FILTER
#
# Selects which retrieved chunks to show as citations.
# Scores by question keyword overlap (2x weight) + answer
# sentence containment (1x weight).
# Different questions get different citations even when the
# retrieved pool overlaps — the 2x question weight ensures this.
# ============================================================

def filter_supporting_docs(question: str, answer: str, docs: List) -> List:
    """Score and select most relevant docs for citations."""
    question_words = set(
        w.strip(".,?!:;\"'()[]")
        for w in question.lower().split()
        if len(w.strip(".,?!:;\"'()[]")) > 3
    )
    answer_sentences = [
        s.strip() for s in answer.lower().split(".")
        if len(s.strip()) > 10
    ]
    scored = []
    for doc in docs:
        content = doc.page_content.lower()
        q_score = len(question_words & set(content.split()))
        a_score = sum(1 for s in answer_sentences if s and s in content)
        scored.append(((q_score * 2) + a_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [doc for score, doc in scored if score > 0]
    return result[:3] if result else docs[:2]


# ============================================================
# 13. MONITORING
# ============================================================

def record_metric(metrics: Dict, key: str, value=1) -> None:
    if key not in metrics:
        metrics[key] = 0
    if isinstance(value, (int, float)):
        metrics[key] += value

def update_latency(metrics: Dict, latency_seconds: float) -> None:
    metrics.setdefault("total_latency", 0.0)
    metrics["total_latency"] += latency_seconds
    metrics["avg_latency_seconds"] = round(
        metrics["total_latency"] / max(metrics.get("total_queries", 1), 1), 2
    )

def update_retrieval_quality(metrics: Dict, top_score: float) -> None:
    """Track rolling average of top reranker score — proxy for retrieval quality."""
    metrics.setdefault("total_rerank_score", 0.0)
    metrics.setdefault("rerank_score_count", 0)
    metrics["total_rerank_score"] += top_score
    metrics["rerank_score_count"] += 1
    metrics["avg_top_rerank_score"] = round(
        metrics["total_rerank_score"] / metrics["rerank_score_count"], 2
    )


# ============================================================
# 14. UTILITIES
# ============================================================

def hash_files(files) -> str:
    hasher = hashlib.md5()
    for f in files:
        hasher.update(f.getvalue())
    return hasher.hexdigest()
