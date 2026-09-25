from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()


def route_question(question: str) -> str:

    prompt = f"""
You are a routing system for a Business Intelligence assistant.

Classify the user's question into exactly one of these categories:

SQL
RAG
HYBRID

Definitions:

SQL:
Use when the question can be answered using structured database fields,
numbers, filtering, sorting, counting, averages, rankings or aggregations.

Examples:
- What are the 10 most expensive products?
- Which brands have the highest average rating?
- Which products have the most 1-star reviews?
- How many products are out of stock?

RAG:
Use when answering requires reading and understanding written customer reviews,
opinions, complaints, praise, reasons or recurring themes.

Examples:
- Why do customers dislike moisturizers?
- What do customers say about product texture?
- What complaints appear in skincare reviews?
- What do customers love about this product?

HYBRID:
-Use when the question requires BOTH structured database analysis
AND understanding customer review text.

-Use HYBRID when the question asks to identify, rank, compare, or
filter products using structured data AND also asks about customer
opinions, feedback, praise, complaints, reasons, or sentiment.

- Words such as "most", "least", "highest", "lowest", "top", or
  "which products" combined with customer feedback usually require HYBRID.

Examples:
- Which expensive moisturizers have poor ratings, and why?
- Which highly rated brands still receive complaints, and what are they?
- Find popular products with bad customer feedback and explain the issues.



"Which moisturizers do customers compliment the most?"
→ HYBRID

"Which moisturizers receive the most complaints?"
→ HYBRID

"Which highly rated products do customers still complain about?"
→ HYBRID

"Why do customers like moisturizers?"
→ RAG

"What do customers complain about product texture?"
→ RAG

USER QUESTION:
{question}

Return only one word:
SQL, RAG or HYBRID.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    route = response.text.strip().upper()

    if route not in {"SQL", "RAG", "HYBRID"}:
        raise ValueError(f"Unexpected route: {route}")

    return route