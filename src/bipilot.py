from src.routing.question_router import route_question
from src.text_to_sql import generate_sql
from src.database import execute_query
from src.rag.rag_pipeline import generate_rag_answer
from src.hybrid_pipeline import generate_hybrid_answer


def ask_bipilot(question: str):

    # 1. Decide which pipeline should answer
    route = route_question(question)

    # -----------------------
    # SQL
    # -----------------------
    if route == "SQL":

        sql = generate_sql(question)

        result = execute_query(sql)

        if result.empty:
            return {
                "route": "SQL",
                "sql": sql,
                "data": result,
                "answer": "I couldn't find any data matching your question.",
                "sources": None
             }

        return {
            "route": "SQL",
            "sql": sql,
            "data": result,
            "answer": None,
            "sources": None
        }

    # -----------------------
    # RAG
    # -----------------------
    elif route == "RAG":

        question_lower = question.lower()

        # Basic sentiment detection
        negative_words = [
            "complain",
            "complaints",
            "unhappy",
            "dislike",
            "hate",
            "problems",
            "issues",
            "bad",
            "negative"
        ]

        positive_words = [
            "love",
            "like",
            "praise",
            "positive",
            "recommend",
            "best"
        ]

        min_rating = None
        max_rating = None

        if any(word in question_lower for word in negative_words):
            max_rating = 2

        elif any(word in question_lower for word in positive_words):
            min_rating = 4

        answer, sources = generate_rag_answer(
            question=question,
            top_k=5,
            min_rating=min_rating,
            max_rating=max_rating
        )

        return {
            "route": "RAG",
            "sql": None,
            "data": None,
            "answer": answer,
            "sources": sources
        }

    # -----------------------
    # HYBRID
    # -----------------------
    elif route == "HYBRID":

        answer, sql, products, reviews = generate_hybrid_answer(
            question
        )

        return {
            "route": "HYBRID",
            "sql": sql,
            "data": products,
            "answer": answer,
            "sources": reviews
        }