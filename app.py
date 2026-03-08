# ============================================================
# app_fv.py — Enterprise Hybrid RAG — UI
#
# SIDEBAR (5 controls):
#   1. Mode — Strict (Compliance) or Comparative
#   2. Documents — which PDFs to search
#   3. Smart Search — GraphRAG toggle
#   4. Session Stats — quality monitoring
#   5. PDF Preview
#
# RETRIEVAL STRATEGY (driven by sidebar mode):
#   Compliance  → BM25 + ANN global hybrid retrieval
#   Comparative → per-document retrieval (top-N from each PDF)
#
# PIPELINE PER QUERY:
#   Answer cache → Retrieval cache → Retrieve → Post-filter
#   → Rerank → Threshold gate → GraphRAG → LLM → Faithfulness
#   → Citations → Cache write → Metrics
# ============================================================

import base64
import time
import streamlit as st
import main as mf
from graph_rag import KnowledgeGraph, combine_hybrid_results

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Synthareva",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📚 Synthareva Enterprise Hybrid RAG")
st.caption("Hybrid Retrieval · GraphRAG · Cross-Encoder Reranking · Faithfulness Check")

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "kb_ready": False,
    "vectorstore": None,
    "retriever": None,
    "rag_chain": None,
    "file_hash": None,
    "chat_history": [],
    "rag_mode": "compliance",
    "chunks": [],
    "knowledge_graph": None,
    "answer_cache": {},
    "retrieval_cache": {},
    "chunk_usage_cache": {},
    "metrics": {
        "total_queries": 0,
        "faithfulness_yes": 0,
        "faithfulness_no": 0,
        "cache_hits": 0,
        "retrieval_cache_hits": 0,
        "i_dont_know_count": 0,
        "graph_hits": 0,
        "total_latency": 0.0,
        "avg_latency_seconds": 0.0,
        "total_rerank_score": 0.0,
        "rerank_score_count": 0,
        "avg_top_rerank_score": 0.0,
        "below_threshold_count": 0,
    },
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

selected_pdfs = []

# ============================================================
# HOW TO USE
# ============================================================
with st.expander("🚀 How to use", expanded=False):
    st.markdown("""
    ### Getting Started

    | Step | Action | Details |
    |------|--------|---------|
    | 1️⃣ | **Upload PDFs** | Use the uploader below. Best results with same-domain documents (e.g. all NLP papers, all financial reports, all legal contracts) |
    | 2️⃣ | **Select documents** | In the sidebar, choose which uploaded PDFs to search. Deselect any you want to exclude. |
    | 3️⃣ | **Pick a mode** | Choose **Strict** or **Comparative** depending on your question type (see below) |
    | 4️⃣ | **Smart Search** | Toggle ON in the sidebar for complex questions about connections between topics or entities |
    | 5️⃣ | **Ask away** | Type your question — every answer includes page citations and a faithfulness check ✅ |

    ---

    ### 🔍 Which mode should I use?

    **Strict mode** — *"Give me the facts from the best matching document"*
    - Searches across all selected PDFs and finds the most relevant passages
    - Answers are precise, grounded, and cited
    - ✅ Best for: *"What is scaled dot-product attention?"*, *"What optimizer did BERT use?"*, *"How many layers does DistilBERT have?"*

    **Comparative mode** — *"What does each document say about this?"*
    - Pulls passages from **every** selected PDF and synthesises them together
    - Shows what each document says, then draws connections across them
    - ✅ Best for: *"How does BERT's training differ from RoBERTa's?"*, *"Compare the architectures of Transformer and DistilBERT"*

    ---

    ### 🧠 What does Smart Search do?

    Smart Search builds a **knowledge graph** from your documents — a map of entities and how they relate to each other.
    It activates automatically in the background when you upload files (no re-upload needed to turn it on/off).

    - ✅ Best for: *"How are BERT and DistilBERT connected?"*, *"Trace the lineage from Transformer to RoBERTa"*
    - ⚪ Leave OFF for simple factual questions — it adds extra context that can be unnecessary

    ---

    ### 💡 Tips for better answers
    - **Be specific** — *"What BLEU score did the Transformer achieve on WMT 2014 English-German?"* works better than *"What were the results?"*
    - **Name your documents** — *"According to the RoBERTa paper, what batch size was used?"* helps Comparative mode focus
    - **Multi-hop questions** — questions like *"What did paper A borrow from paper B and then paper C improve upon?"* work best with Smart Search ON and Comparative mode
    """)


# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

# ============================================================
# RESET ON NEW UPLOAD
# ============================================================
if uploaded_files:
    new_hash = mf.hash_files(uploaded_files)
    if st.session_state.file_hash != new_hash:
        st.session_state.file_hash = new_hash
        st.session_state.kb_ready = False
        st.session_state.chat_history = []
        st.session_state.answer_cache = {}
        st.session_state.retrieval_cache = {}
        st.session_state.chunk_usage_cache = {}
        st.session_state.knowledge_graph = None
        st.session_state.chunks = []
        st.session_state.metrics = {
            "total_queries": 0,
            "faithfulness_yes": 0,
            "faithfulness_no": 0,
            "cache_hits": 0,
            "retrieval_cache_hits": 0,
            "i_dont_know_count": 0,
            "graph_hits": 0,
            "total_latency": 0.0,
            "avg_latency_seconds": 0.0,
            "total_rerank_score": 0.0,
            "rerank_score_count": 0,
            "avg_top_rerank_score": 0.0,
            "below_threshold_count": 0,
        }

# ============================================================
# SIDEBAR — 1. Mode
# ============================================================
st.sidebar.header("⚙️ Mode")
rag_mode = st.sidebar.radio(
    "Answer mode",
    options=["Strict (Compliance)", "Comparative"],
    index=0,
    help=(
        "Strict — answers only from the most relevant document passages. "
        "Best for factual questions about a specific topic.\n\n"
        "Comparative — pulls passages from every selected PDF and synthesises across them. "
        "Best for comparison questions across documents."
    )
)
mode_key = "compliance" if rag_mode == "Strict (Compliance)" else "comparative"

# ============================================================
# SIDEBAR — 2. Documents
# ============================================================
if uploaded_files:
    st.sidebar.header("📄 Documents")
    pdf_names = [f.name for f in uploaded_files]
    selected_pdfs = st.sidebar.multiselect(
        "Search in these documents",
        options=pdf_names,
        default=pdf_names,
        help=(
            "Select which PDFs to include in search. "
            "For Comparative mode, select all documents you want to compare."
        )
    )

# ============================================================
# SIDEBAR — 3. Smart Search (GraphRAG)
# ============================================================
st.sidebar.header("🧠 Smart Search")
enable_graph = st.sidebar.toggle(
    "Enable relationship search",
    value=False,
    help=(
        "Uses a knowledge graph of entities and relationships built from your documents. "
        "Useful for questions like 'How are X and Y connected?' or 'What influenced X?'\n\n"
        "Graph is built automatically on upload — toggle activates instantly, no re-upload needed."
    )
)

if st.session_state.knowledge_graph is not None:
    stats = st.session_state.knowledge_graph.get_stats()
    if stats["nodes"] > 0:
        status = "🟢 Active" if enable_graph else "⚪ Built (toggle to activate)"
        st.sidebar.caption(
            f"{status} · {stats['nodes']} entities · {stats['edges']} relationships"
        )
    elif st.session_state.kb_ready:
        st.sidebar.caption("⚠️ Graph built but empty — check spaCy model install")

# ============================================================
# SIDEBAR — 4. Session Stats
# ============================================================
if st.session_state.metrics["total_queries"] > 0:
    m = st.session_state.metrics
    total = m["total_queries"]

    st.sidebar.header("📊 Session Stats")

    col1, col2 = st.sidebar.columns(2)
    col1.metric("Questions", total)
    col2.metric(
        "Faithfulness",
        f"{m['faithfulness_yes'] / total * 100:.0f}%",
        help="% of answers where every claim was verified against source documents"
    )

    col3, col4 = st.sidebar.columns(2)
    col3.metric("Avg Response", f"{m['avg_latency_seconds']:.1f}s")
    col4.metric(
        "Refused to Guess",
        f"{m['i_dont_know_count'] / total * 100:.0f}%",
        help="% of questions where the app said 'I don't know' rather than hallucinating"
    )

    avg_score = m.get("avg_top_rerank_score", 0.0)
    quality_label = (
        "✅ Good" if avg_score > 2.0
        else "⚠️ Low" if avg_score > 0.0
        else "❌ Poor"
    )
    st.sidebar.metric(
        "Retrieval Quality",
        f"{avg_score:.1f} ({quality_label})",
        help=(
            "Average cross-encoder relevance score of best retrieved chunk. "
            ">2: retrieval is working well | 0–2: borderline | <0: check PDF selection"
        )
    )

    below = m.get("below_threshold_count", 0)
    if below > 0:
        st.sidebar.metric(
            "Hard Skipped",
            f"{below} ({below / total * 100:.0f}%)",
            help=(
                "Queries skipped entirely — retrieved chunks were completely irrelevant. "
                "High rate means your PDFs don't cover the topics being asked about."
            )
        )

    if enable_graph:
        st.sidebar.metric(
            "Graph Hits",
            f"{m['graph_hits'] / total * 100:.0f}%",
            help="% of queries where the knowledge graph found relevant relationships"
        )

# ============================================================
# SIDEBAR — 5. PDF Preview
# ============================================================
if uploaded_files:
    st.sidebar.header("📄 Preview")
    selected_preview = st.sidebar.selectbox(
        "Preview document",
        uploaded_files,
        format_func=lambda x: x.name,
    )
    base64_pdf = base64.b64encode(selected_preview.getvalue()).decode("utf-8")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        f'width="100%" height="500"></iframe>',
        unsafe_allow_html=True,
    )

# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================
if uploaded_files and not st.session_state.kb_ready:
    with st.spinner("📦 Indexing documents..."):
        progress = st.progress(0, text="Loading PDFs...")

        docs = mf.load_pdfs(uploaded_files)
        progress.progress(20, text="Splitting into chunks...")

        chunks = mf.split_docs(docs, max_chunks_per_source=200)
        st.session_state.chunks = chunks
        progress.progress(40, text="Building vector index...")

        vectorstore = mf.create_vectorstore(st.session_state.file_hash, chunks)
        progress.progress(60, text="Building retriever...")

        retriever = mf.get_hybrid_retriever(chunks, vectorstore)
        progress.progress(70, text="Building RAG chain...")

        rag_chain = mf.build_chat_rag_chain(mode=mode_key)
        progress.progress(80, text="Building knowledge graph...")

        # Always build — toggle only controls whether it is USED at query time.
        # No re-upload needed to activate Smart Search.
        kg = KnowledgeGraph()
        kg.build_from_chunks(chunks)
        st.session_state.knowledge_graph = kg

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever
        st.session_state.rag_chain = rag_chain
        st.session_state.kb_ready = True
        progress.progress(100, text="Ready!")

    doc_count = len(set(c.metadata.get("source") for c in chunks))
    st.success(f"✅ {doc_count} PDFs · {len(chunks)} chunks · Ready to query")

# ============================================================
# REBUILD CHAIN WHEN MODE CHANGES
# ============================================================
if uploaded_files and st.session_state.kb_ready:
    if st.session_state.rag_mode != mode_key:
        st.session_state.rag_chain = mf.build_chat_rag_chain(mode_key)
        st.session_state.rag_mode = mode_key

# ============================================================
# CITATION HELPER
# ============================================================
def build_citations(question: str, answer: str, docs: list) -> list:
    """Select supporting docs and format as citation dicts."""
    supporting = mf.filter_supporting_docs(question, answer, docs)

    # Update chunk usage cache for diversity tracking
    for doc in supporting:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        st.session_state.chunk_usage_cache[key] = (
            st.session_state.chunk_usage_cache.get(key, 0) + 1
        )

    citations = []
    seen = set()
    for doc in supporting:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            })
    return citations


# ============================================================
# QUERY PIPELINE
#
# Flow:
#   Answer cache hit → serve immediately
#   Retrieval cache hit → skip retrieval, go to rerank
#   Compliance mode  → hybrid BM25+ANN retrieval
#   Comparative mode → per-document retrieval (one per PDF)
#   Post-filter → remove sub-50-word fragments
#   Rerank → cross-encoder scores, returns (docs, top_score)
#   Threshold gate → if top_score < -5.0, skip LLM entirely
#   GraphRAG → append entity relationship evidence if enabled
#   LLM → generate answer
#   Faithfulness → verify answer against context
#   Cache write → store answer + citations
# ============================================================
if uploaded_files and st.session_state.kb_ready:
    retriever = st.session_state.retriever
    rag_chain = st.session_state.rag_chain
    knowledge_graph = st.session_state.knowledge_graph

    query = st.chat_input("Ask a question about your documents")

    if query:
        t_start = time.time()
        mf.record_metric(st.session_state.metrics, "total_queries")

        # ── Answer Cache ──────────────────────────────────────
        cached = mf.check_answer_cache(query, st.session_state.answer_cache)
        if cached:
            answer   = cached["answer"]
            citations = cached["citations"]
            faithful = cached["faithful"]
            mf.record_metric(st.session_state.metrics, "cache_hits")

        else:
            # ── Retrieval Cache ───────────────────────────────
            cached_docs = mf.check_retrieval_cache(
                query, st.session_state.retrieval_cache
            )

            if cached_docs:
                retrieved_docs = cached_docs
                mf.record_metric(st.session_state.metrics, "retrieval_cache_hits")
            else:
                # ── Retrieval — mode-driven ───────────────────
                expanded_queries = mf.expand_query(query)

                if mode_key == "comparative" and len(selected_pdfs) > 1:
                    # Run per-document retrieval for each expanded query variant,
                    # then deduplicate across all variants before reranking.
                    seen_hashes = set()
                    all_retrieved = []
                    for q_variant in expanded_queries:
                        docs = mf.retrieve_per_document(
                            q_variant,
                            st.session_state.vectorstore,
                            selected_pdfs,
                            chunks_per_doc=3,
                        )
                        for doc in docs:
                            doc_hash = hash(doc.page_content[:100])
                            if doc_hash not in seen_hashes:
                                seen_hashes.add(doc_hash)
                                all_retrieved.append(doc)
                else:
                    # Query expansion: retrieves on original + 3 paraphrases,
                    # deduplicates, then passes full candidate set to reranker.
                    all_retrieved = mf.retrieve_with_expansion(
                        query, retriever
                    )

                # Scope to user-selected PDFs
                retrieved_docs = [
                    d for d in all_retrieved
                    if d.metadata.get("source") in selected_pdfs
                ]

                mf.write_retrieval_cache(
                    query, retrieved_docs, st.session_state.retrieval_cache
                )

            if not retrieved_docs:
                st.warning(
                    "⚠️ No content found in selected documents. "
                    "Try selecting more PDFs in the sidebar."
                )
                st.stop()

            # ── Post-Filter ───────────────────────────────────
            # Remove chunks shorter than 50 words (headers, footers, fragments)
            retrieved_docs = mf.post_filter(retrieved_docs, min_word_count=50)

            # ── Diversity Sort ────────────────────────────────
            # Deprioritise chunks from (source, page) pairs used recently
            retrieved_docs = sorted(
                retrieved_docs,
                key=lambda d: -st.session_state.chunk_usage_cache.get(
                    (d.metadata.get("source"), d.metadata.get("page")), 0
                ),
                reverse=True,
            )

            # ── Cross-Encoder Reranking ───────────────────────
            retrieved_docs, top_score = mf.rerank_docs(query, retrieved_docs, top_k=5)
            mf.update_retrieval_quality(st.session_state.metrics, top_score)

            # ── Threshold Gate ────────────────────────────────
            # Only fires when best chunk is completely off-topic (score < -5.0)
            if top_score < mf.RELEVANCE_THRESHOLD:
                answer = (
                    "I don't know. The selected documents do not appear to contain "
                    "relevant information to answer this question."
                )
                citations = []
                faithful = "YES"
                mf.record_metric(st.session_state.metrics, "below_threshold_count")
                mf.record_metric(st.session_state.metrics, "i_dont_know_count")

            else:
                # ── GraphRAG ──────────────────────────────────
                graph_evidence = ""
                if enable_graph and knowledge_graph and not knowledge_graph.is_empty():
                    graph_result = knowledge_graph.traverse(query, max_hops=2)
                    if graph_result["found"]:
                        mf.record_metric(st.session_state.metrics, "graph_hits")
                    retrieved_docs, graph_evidence = combine_hybrid_results(
                        retrieved_docs, graph_result, query
                    )

                # ── Format Context ────────────────────────────
                context = mf.format_docs(retrieved_docs, graph_evidence)

                # ── Conversation History ──────────────────────
                history_text = "\n".join(
                    f"Q: {q}\nA: {a}"
                    for q, a, c, f in st.session_state.chat_history[-5:]
                )

                # ── LLM Generation ────────────────────────────
                answer = rag_chain.invoke({
                    "question": query,
                    "context": context,
                    "history": history_text,
                }).content.strip()

                # ── Citations ─────────────────────────────────
                citations = build_citations(query, answer, retrieved_docs)

                # ── Faithfulness Check ────────────────────────
                faithful = mf.faithfulness_check(answer, context)

            # ── Write Answer Cache ────────────────────────────
            mf.write_answer_cache(
                query,
                {"answer": answer, "citations": citations, "faithful": faithful},
                st.session_state.answer_cache,
            )

        # ── Metrics ───────────────────────────────────────────
        mf.update_latency(st.session_state.metrics, time.time() - t_start)
        if faithful.strip().upper() == "YES":
            mf.record_metric(st.session_state.metrics, "faithfulness_yes")
        else:
            mf.record_metric(st.session_state.metrics, "faithfulness_no")
        if "i don't know" in answer.lower() or "do not contain" in answer.lower():
            mf.record_metric(st.session_state.metrics, "i_dont_know_count")

        # ── Chat History ──────────────────────────────────────
        st.session_state.chat_history.append((query, answer, citations, faithful))

        # Force sidebar to re-render with updated metrics immediately
        # (without this, stats only appear after the second question)
        st.rerun()

# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================
if st.session_state.chat_history:
    st.divider()
    st.subheader("💬 Conversation")

    for i, (q, a, c, f) in enumerate(st.session_state.chat_history):
        st.chat_message("user").write(q)

        with st.chat_message("assistant"):
            st.write(a)

            # Faithfulness badge
            faith_icon  = "✅" if f.strip().upper() == "YES" else "❌"
            faith_color = "green" if f.strip().upper() == "YES" else "red"
            st.markdown(
                f"{faith_icon} **Faithfulness:** :{faith_color}[{f.strip().upper()}]"
            )

            # Citations — clean, no doc_type badge
            if c:
                st.markdown("**📚 Sources:**")
                for cite in c:
                    st.write(f"• {cite['source']} — Page {cite['page']}")
            else:
                st.write("📚 No sources cited.")

            # Response time — only on most recent answer
            if i == len(st.session_state.chat_history) - 1:
                st.caption(
                    f"⏱ {st.session_state.metrics.get('avg_latency_seconds', 0):.1f}s avg"
                )
