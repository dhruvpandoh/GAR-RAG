# GAR-RAG Search

A lightweight Retrieval-Augmented Generation (RAG) service that enhances user queries, embeds text, and retrieves the most relevant context for downstream LLM answering. Built as a modular FastAPI + FAISS + Hugging Face stack with an optional Streamlit UI.

---

## What this project does

GAR-RAG is designed to be a small, understandable RAG baseline you can extend. The core flow is:

1. **Query Enhancement**  
   A seq2seq model rewrites/expands the user’s query to improve retrieval quality.

2. **Embedding**  
   Enhanced queries and documents are converted into dense vectors using a SentenceTransformer model.

3. **Vector Retrieval (FAISS)**  
   Vectors are indexed and searched to fetch top-k relevant chunks.

4. **(Optional) Metadata / Persistence**  
   Postgres + SQLAlchemy scaffolding is included for storing documents, chunks, and retrieval logs.

5. **(Optional) UI**  
   Streamlit can be used to demo the RAG pipeline end-to-end.

---

## Features

- Query-enhanced retrieval to improve recall and ranking  
- Sentence-transformer embeddings for fast semantic search  
- FAISS vector index for scalable nearest-neighbor retrieval  
- FastAPI backend scaffold ready for REST endpoints  
- Postgres/SQLAlchemy + Alembic scaffold for persistence  
- Test + CI friendly layout (pytest)

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic  
- **ML / NLP:** Hugging Face Transformers, Sentence-Transformers, Torch, Accelerate, BitsAndBytes  
- **Vector Search:** FAISS (CPU)  
- **DB:** PostgreSQL, SQLAlchemy, Alembic  
- **UI:** Streamlit  
- **Tests:** Pytest  

---

## Repository Structure

```
gar-rag-main/
├─ app/
│  ├─ api/                 # FastAPI routes (scaffold)
│  ├─ core/
│  │  ├─ config.py         # env/config management
│  │  └─ database.py       # SQLAlchemy session + Base
│  ├─ models/              # ORM models (scaffold)
│  └─ services/
│     └─ model_service.py  # query enhancement + embeddings
├─ tests/                  # pytest scaffold
├─ requirements.txt
└─ README.md
```

---

## Setup

### 1) Prerequisites

- Python 3.9+ (3.10+ recommended)
- (Optional) PostgreSQL if you want persistence
- (Optional) GPU for faster model inference

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file in the repo root:

```env
# Database
DATABASE_URL=postgresql://localhost/gar_rag_search

# Models
QUERY_ENHANCEMENT_MODEL=<huggingface-model-name>
EMBEDDING_MODEL=<sentence-transformer-name>

# Optional local LLM path (if you later add generation)
LLAMA_MODEL_PATH=./models/llama-2-3.2b.gguf
```

> `config.py` loads `.env` automatically via Pydantic.

---

## Running the project

### Backend (FastAPI)

The repo currently includes the service layer and DB scaffold.  
To run an API server, add an entrypoint like `app/main.py`:

```python
from fastapi import FastAPI
from app.services.model_service import ModelService

app = FastAPI(title="GAR-RAG Search")
model_service = ModelService()

@app.get("/health")
def health():
    return {"status": "ok"}

# Add retrieval endpoints here:
# /ingest, /search, /rag, etc.
```

Start the server:

```bash
uvicorn app.main:app --reload
```

FastAPI docs:
- `http://127.0.0.1:8000/docs`

---

### UI (Streamlit)

If you add a Streamlit entrypoint (e.g., `streamlit_app.py`), run:

```bash
streamlit run streamlit_app.py
```

---

## Testing

```bash
pytest -q
```

---

## Extending / Roadmap ideas

If you want to take this from scaffold → full RAG system:

- **Document ingestion**
  - chunking strategies (fixed, recursive, semantic)
  - store chunks + metadata in Postgres
- **FAISS index management**
  - create/load/save index
  - incremental updates
- **Retrieval API**
  - `/ingest`, `/search`, `/rag`
  - return top-k chunks with scores + metadata
- **Generation**
  - plug in a local GGUF model or OpenAI-style API
  - return cited answers
- **Evaluation**
  - Recall@k, nDCG, MRR
  - ablation: enhanced vs raw queries

---

## Authors

- Adamay Mann  
- Dhruv Pandoh  
