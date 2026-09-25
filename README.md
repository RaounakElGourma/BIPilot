# BIPilot

> **Natural-language Business Intelligence for product data and customer reviews.**

BIPilot is a Streamlit-based BI copilot that combines **Text-to-SQL**, **RAG**, and **Hybrid SQL + Review Analysis** to answer business questions in natural language.

##  What it can do

- **Text-to-SQL** — converts business questions into read-only SQLite queries
- **RAG** — retrieves relevant customer reviews with FAISS + Sentence Transformers
- **Hybrid analysis** — combines structured product metrics with customer feedback
- **Visual insights** — interactive Plotly charts and result tables
- **Customer evidence** — shows the reviews supporting generated insights
- **History** — reopens previous analyses without rerunning the model
- **CSV export** — displays a compact result set while allowing full-result export

```

## ⚙️ Architecture

```text
User Question
     │
     ▼
Question Router
 ┌───────┬────────┬─────────┐
 │ SQL   │  RAG   │ HYBRID  │
 └───────┴────────┴─────────┘
     │       │         │
  SQLite   FAISS   SQL + Reviews
     └───────┴─────────┘
             │
             ▼
        BIPilot Insight
```

##  Tech Stack

`Python` · `Streamlit` · `Google Gemini` · `SQLite` · `Pandas` · `Plotly` · `FAISS` · `Sentence Transformers`

##  Core Structure

```text
BIPilot/
├── app.py
├── requirements.txt
├── .gitignore
└── src/
    ├── bipilot.py
    ├── database.py
    ├── hybrid_pipeline.py
    ├── schema.py
    ├── text_to_sql.py
    ├── visualization.py
    ├── rag/
    └── routing/
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create a local `.env` file for your Gemini API key:


## Note

The local dataset, SQLite database, and FAISS vector index are not included in the public repository.


