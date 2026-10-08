# ⚡ Production-Grade SAP CDS View Recommender & Generator (GraphRAG + ADT Agentic Loop)

An enterprise-grade, full-stack AI system that connects directly to **SAP S/4HANA 2023** via **Eclipse ADT REST APIs**, parses CDS View entities, builds a dual **VectorDB (ChromaDB)** + **Knowledge Graph (Neo4j)** index, provides an intelligent UI for reusability checking and multi-view generation, and auto-corrects generated DDL code via a **live SAP ADT syntax check loop**.

---

## 🏛️ Architectural Blueprint (6 Integrated Engines)

```
[SAP S/4HANA 2023 (ADT)] 
       │
       ▼ (REST / Session + CSRF Pool)
[1. ADT Ingestion Engine] ───► [2. DDL AST/Regex Parser]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       [3. Vector Store]                       [4. Knowledge Graph]
  (ChromaDB: Signatures, Labels)          (Neo4j: Views, Fields, Joins)
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                           [5. Query & Routing Agent]
                           (Evaluate Reusability vs. Synthesis)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
     [Path A: 100% Reuse Match]             [Path B: Multi-View Generation]
     Return existing view metadata           LLM Mega-Prompt with Graph Traversal
                                                         │
                                                         ▼
                                            [6. Live ADT Syntax Check Loop]
                                            POST to /checkruns ──► Auto-Heal (Max 3x)
```

---

## 🚀 Key Architectural Achievements

1. **Zero Hallucination Joins:** Joins and foreign key expressions are derived directly from the Knowledge Graph topology extracted from the real SAP S/4HANA system (`$projection.CustomerID = _Customer.Customer`), eliminating fabricated join keys.
2. **True S/4HANA 2023 Compliance:** Strictly enforces modern `define view entity` syntax instead of obsolete `define view` (V1) DDIC-based views.
3. **Closed-Loop Reliability:** The live SAP ADT compiler (`/sap/bc/adt/checkruns` & `/syntaxcheck`) actively validates generated code, iteratively feeding compiler diagnostics back to the LLM agent until 0 errors are achieved.

---

## 📁 Project Directory Structure

```
CDS-RAG/
├── .env                       # Target SAP system and DB configuration
├── config.py                  # Pydantic central settings
├── core/
│   ├── sap_adt_client.py      # Module 1: SAP ADT REST Client (Auth, CSRF, QuickSearch, DDL, Syntax Check)
│   ├── ddl_parser.py          # Module 2: AST & Regex Feature Extractor (Entities, Fields, Associations)
│   ├── database.py            # Module 3: Dual Storage (ChromaDB Vector Store + Neo4j/NetworkX Graph)
│   ├── recommender.py         # Module 4: Reusability Evaluator & Knowledge Graph Traversal
│   ├── generator.py           # Module 5: LLM Mega-Prompt Generator (Anthropic/OpenAI/Fallback)
│   └── compiler_loop.py       # Module 6: Live ADT Auto-Healing & Compiler Feedback Loop
├── app/
│   └── main.py                # Module 7: Streamlit Enterprise Web UI
├── scripts/
│   └── ingest_views.py        # Batch Ingestion Pipeline from SAP S/4HANA into ChromaDB + Graph
├── tests/
│   ├── test_adt_connection.py # Live SAP ADT connection & CSRF unit tests
│   ├── test_ddl_parser.py     # CDS entity & classic view parsing tests
│   ├── test_recommender.py    # Reusability Path A and Path B tests
│   └── test_compiler_loop.py  # Live ADT syntax check & self-healing tests
├── run_app.py                 # Streamlit UI launcher script
└── README.md
```

---

## ⚙️ Configuration (`.env`)

```env
# SAP Target System Configuration
SAP_SYSTEM_ID=DEV
SAP_CLIENT=100
SAP_S4HANA_RELEASE=2023
SAP_BASE_URL=https://YAWSS4HSBX.sapyash.com:44301
SAP_USERNAME=AIENAPARTHIP
SAP_PASSWORD=Training@2026
SAP_VERIFY_SSL=false
SAP_REQUEST_TIMEOUT_SECONDS=30

# Graph and Vector DB
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
CHROMA_PERSIST_DIR=./data/chroma_db

# LLM Providers (OpenAI / Anthropic / Gemini)
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
DEFAULT_MODEL=claude-3-5-sonnet-20241022
```

---

## 🧪 Running Test Suite

Verify the entire pipeline against your live SAP S/4HANA system:

```bash
# Run all tests
python -m unittest discover tests

# Or run individual test suites:
python tests/test_adt_connection.py
python tests/test_ddl_parser.py
python tests/test_recommender.py
python tests/test_compiler_loop.py
```

---

## 📥 Ingesting Views from SAP S/4HANA

Extract and index live custom (`Z*`, `Y*`) and standard (`I_*`, `C_*`) views from your SAP system:

```bash
python scripts/ingest_views.py
```

---

## 🖥️ Launching the Web UI

Run the Streamlit application:

```bash
python run_app.py
# Or directly:
streamlit run app/main.py
```

Access the UI at `http://localhost:8501`.
