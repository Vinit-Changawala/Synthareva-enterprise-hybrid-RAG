# 🧠 Synthareva
### Hybrid RAG · Knowledge Graph · Cross-Encoder Reranking · Faithfulness Verification

> **Upload documents. Ask anything. Get cited, verified answers.**

---

## 🔤 What Does the Name Mean?

**Synthareva** is built from three words that describe exactly what it does:

| Fragment | Origin | Meaning |
|---|---|---|
| **Synth** | English — *Synthesis* | Combines information from multiple documents into one coherent answer |
| **ara** | Sanskrit — *essence* | Distils only the essential, relevant knowledge from your documents |
| **eva** | English — from *Retrieval* | Retrieves the right passages before synthesising the answer |

> *Synthareva — retrieving the essence, synthesising the truth.*

---

## 🤖 What Does It Do?

Synthareva is an enterprise-grade document intelligence system that goes far beyond simple keyword search. Upload any set of PDFs — research papers, legal contracts, financial reports, technical manuals — and ask questions in plain English. Synthareva retrieves the most relevant passages from across all your documents, reasons over them using a knowledge graph, and generates a precise answer with page-level citations and a faithfulness check on every single response.

It does not guess. If the answer is not in your documents, it says so.

---

## 🔗 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://synthareva.streamlit.app)

---

## ✨ Features

| Feature | What it does |
|---|---|
| **Hybrid Retrieval** | BM25 keyword search (30%) + semantic vector search (70%) combined for best coverage |
| **Knowledge Graph** | spaCy NER extracts entities and builds a traversable graph — finds cross-document connections |
| **Query Expansion** | LLM rewrites your question 3 ways before retrieval — fixes vocabulary mismatch |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` rescores every retrieved chunk by true relevance |
| **MMR Diversity** | Maximal Marginal Relevance prevents 5 chunks from the same page dominating results |
| **Faithfulness Check** | Every answer is verified against source context — no silent hallucinations |
| **Comparative Mode** | Per-document retrieval guarantees every PDF is represented in cross-document questions |
| **Answer + Retrieval Cache** | Identical queries served instantly without re-running the pipeline |
| **Session Analytics** | Live metrics: faithfulness %, avg response time, retrieval quality score, graph hit rate |

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────┐
│                  INDEXING                   │
│  PyPDFLoader → RecursiveTextSplitter        │
│  → BGE-base-en-v1.5 embeddings             │
│  → Chroma vectorstore (cosine HNSW)         │
│  → BM25 index                               │
│  → spaCy NER → NetworkX KnowledgeGraph      │
└─────────────────────────────────────────────┘
    │
    ▼
User Query
    │
    ├─[Strict mode]──────────────────────────────────────────┐
    │   Query Expansion (LLM → 3 paraphrases)                │
    │   → BM25 + ANN Ensemble Retriever                      │
    │   → Deduplicated candidate pool                        │
    │                                                        │
    ├─[Comparative mode]─────────────────────────────────────┤
    │   Per-document retrieval (top-N chunks per PDF)        │
    │   → Merged candidate pool                              │
    │                                                        │
    ▼                                                        │
Post-filter (remove < 50 word fragments) ◄──────────────────┘
    │
    ▼
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
    │
    ▼
Relevance Threshold Gate (score < -5.0 → early refusal)
    │
    ▼
[Smart Search ON] Knowledge Graph Traversal
    BFS up to 2 hops → entity relationships + graph evidence
    │
    ▼
LLM Generation (Mistral NeMo 12B via Mistral AI API)
    │
    ▼
Faithfulness Check (LLM-as-judge: YES / NO)
    │
    ▼
Answer + Citations + Cache Write
```

---

## 📁 Project Structure

```
synthareva/
│
├── app_fv.py          # Streamlit UI — sidebar, upload, query pipeline, chat display
├── main_fv.py         # Backend — ingestion, retrieval, reranking, prompts, caching
├── graph_rag.py       # Knowledge graph — entity extraction, graph build, BFS traversal
├── requirements.txt   # Python dependencies
├── .env               # API keys (not committed — see setup)
└── README.md
```

---

## ⚙️ How the Pipeline Works

### 1. Indexing (runs once per upload)
- PDFs loaded page-by-page with `PyPDFLoader`
- Split into 700-character chunks, 100-character overlap via `RecursiveCharacterTextSplitter`
- Per-source cap of 200 chunks — prevents large PDFs from dominating the index
- Embeddings: `BAAI/bge-base-en-v1.5` (768-dim, cosine similarity)
- Stored in **Chroma** with HNSW cosine index
- **BM25** index built from same chunks
- **Knowledge graph** built via spaCy NER — entities become nodes, co-occurrences become weighted edges

### 2. Query Expansion
Before retrieval, the LLM generates 3 alternative phrasings of the query. Bridges vocabulary gaps — e.g. *"pretraining data"* expands to also search *"training corpus"*, *"same dataset as BERT"*, etc. Results from all 4 queries are deduplicated before reranking.

### 3. Retrieval
- **Strict mode:** `EnsembleRetriever` — BM25 (weight 0.3) + ANN/MMR (weight 0.7). MMR `lambda=0.7` balances relevance vs diversity.
- **Comparative mode:** Separate retriever per PDF with `source` metadata filter. Guarantees every document contributes — solves the crowding problem where one high-frequency PDF would otherwise dominate.

### 4. Reranking
`CrossEncoder(ms-marco-MiniLM-L-6-v2)` scores every (query, chunk) pair. Returns top-5 by score. Top score feeds the relevance gate — queries below -5.0 get an early *"I don't know"* rather than a hallucinated answer.

### 5. Knowledge Graph (Smart Search)
Built during indexing using spaCy `en_core_web_sm`. At query time:
- Query entities detected with spaCy
- Matched against graph nodes (exact + partial match)
- BFS traversal up to 2 hops collects related entities and relationships
- Graph evidence prepended to LLM context
- Vector chunks re-sorted to boost those containing graph-matched entities

### 6. LLM Generation
`open-mistral-nemo` (12B, 128K context) via Mistral AI API. Temperature 0, max 512 tokens. Strict prompt: answer only from context, cite every fact, say "I don't know" if context is insufficient.

### 7. Faithfulness Verification
Second LLM call evaluates whether every claim in the answer is grounded in retrieved context. Returns YES or NO. Displayed as ✅ or ❌ on every response.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Mistral AI API key — free tier at [console.mistral.ai](https://console.mistral.ai) (1B tokens/month free)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/synthareva.git
cd synthareva
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 4. Set your API key
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 5. Run
```bash
streamlit run app_fv.py
```

---

## 📦 requirements.txt

```
streamlit
python-dotenv
langchain
langchain-community
langchain-huggingface
langchain-chroma
langchain-mistralai
langchain-core
langchain-text-splitters
pypdf
sentence-transformers
chromadb
rank-bm25
spacy
networkx
huggingface-hub
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | ✅ Yes | Mistral AI API key for LLM generation and faithfulness checks |

---

## 📊 Session Metrics Explained

| Metric | Meaning |
|---|---|
| **Faithfulness %** | % of answers where every claim was verified against source documents |
| **Avg Response** | Mean end-to-end latency per query including retrieval + reranking + LLM |
| **Refused to Guess** | % of queries where the system said "I don't know" instead of hallucinating |
| **Retrieval Quality** | Rolling average of the cross-encoder's top score. >2.0 = ✅ Good, 0–2.0 = ⚠️ Low, <0 = ❌ Poor |
| **Graph Hits** | % of queries where the knowledge graph found relevant entity relationships |
| **Hard Skipped** | Queries where top reranker score < -5.0 — documents genuinely don't cover the topic |

---

## 💡 Usage Tips

**For best retrieval quality:**
- Upload same-domain documents (e.g. all NLP papers, all financial reports, all legal contracts)
- Be specific — *"What BLEU score did the Transformer achieve on WMT 2014 EN-DE?"* beats *"What were the results?"*

**For comparative questions:**
- Select all relevant PDFs in the sidebar before asking
- Name documents explicitly — *"According to the RoBERTa paper, what batch size was used?"*

**For relationship questions (Smart Search):**
- Questions like *"How are BERT and DistilBERT connected?"* or *"Trace the lineage from Transformer to RoBERTa"*
- Toggle Smart Search ON — the graph is pre-built at upload time and activates instantly, no re-upload needed

---

## 🧪 Tested On

| Domain | Document Types |
|---|---|
| NLP Research | Transformer, BERT, RoBERTa, DistilBERT papers |
| Financial | Annual reports, earnings releases |
| Legal | Contracts, compliance documents |

---

## 🛠️ Model Details

| Component | Model | Purpose |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` | Semantic vector search |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Relevance scoring |
| LLM | `open-mistral-nemo` (12B) | Answer generation + faithfulness check |
| NER | `spacy/en_core_web_sm` | Entity extraction for knowledge graph |

---

## 🗺️ Roadmap

- [ ] Support for `.docx`, `.txt`, `.csv` file formats
- [ ] Multi-language document support
- [ ] Export conversation history as PDF report
- [ ] Persistent knowledge graph across sessions
- [ ] User-defined entity categories for domain-specific graphs

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built with [LangChain](https://github.com/langchain-ai/langchain) · [Chroma](https://github.com/chroma-core/chroma) · [Streamlit](https://streamlit.io) · [Mistral AI](https://mistral.ai) · [Hugging Face](https://huggingface.co) · [spaCy](https://spacy.io) · [NetworkX](https://networkx.org)
