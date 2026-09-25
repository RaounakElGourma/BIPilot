# BIPilot

BIPilot is a Business Intelligence copilot for exploring product data and customer reviews through natural-language questions.

It combines **Text-to-SQL**, **RAG**, and **Hybrid SQL + Review Analysis** to answer both structured business questions and qualitative customer-feedback questions.

## Features

- Natural-language to SQLite queries
- Semantic review retrieval with FAISS
- Hybrid analysis combining product metrics and customer feedback
- Interactive Plotly charts
- Customer review evidence
- Session history
- Full CSV export
- Read-only SQL execution and validation

## How it works

BIPilot routes each question to the appropriate pipeline:

- **SQL** — structured questions about products, brands, prices, ratings, and aggregates
- **RAG** — qualitative questions about customer opinions and reviews
- **Hybrid** — questions that require both structured data and customer feedback

```text
User Question
      |
      v
Question Router
   /    |    \
 SQL   RAG   HYBRID
  |     |       |
SQLite FAISS  SQL + Reviews
   \    |      /
      BIPilot
```

## Tech Stack

**Python · Streamlit · Google Gemini · SQLite · Pandas · Plotly · FAISS · Sentence Transformers**

## Project Structure

```text
BIPilot/
├── app.py
├── requirements.txt
├── .gitignore
│
└── src/
    ├── bipilot.py
    ├── database.py
    ├── hybrid_pipeline.py
    ├── schema.py
    ├── sql_repair.py
    ├── sql_semantic_check.py
    ├── text_to_sql.py
    ├── visualization.py
    │
    ├── rag/
    │   ├── __init__.py
    │   ├── rag_pipeline.py
    │   └── retriever.py
    │
    └── routing/
        ├── __init__.py
        └── question_router.py
```

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

## Data

The local dataset, SQLite database, and FAISS vector index are not included in this repository.
