from dotenv import load_dotenv
from google import genai

from src.rag.retriever import retrieve_reviews


load_dotenv()

client = genai.Client()


def generate_rag_answer(
    question: str,
    top_k: int = 5,
    min_rating=None,
    max_rating=None
):

    # 1. Retrieve relevant reviews
    reviews = retrieve_reviews(
        question=question,
        top_k=top_k,
        min_rating=min_rating,
        max_rating=max_rating
    )

    # 2. Build the context
    context_parts = []

    for _, row in reviews.iterrows():

        context_parts.append(
            f"""
Product: {row["product_name"]}
Brand: {row["brand_name"]}
Rating: {row["review_rating"]}
Review: {row["review_text"]}
"""
        )

    context = "\n---\n".join(context_parts)

    # 3. Give the question + retrieved reviews to Gemini
    prompt = f"""
You are a customer insights assistant.

Answer the user's question using only the customer reviews below.

Do not invent information.
Identify recurring themes when possible.
If there is not enough evidence, say so.

USER QUESTION:
{question}

CUSTOMER REVIEWS:
{context}

Give a concise and clear answer.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text, reviews