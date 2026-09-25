import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.schema import DATABASE_SCHEMA


load_dotenv()

client = genai.Client()


def clean_sql(response: str) -> str:
    """
    Remove markdown code blocks if the LLM returns them.
    """

    response = response.strip()

    response = re.sub(
        r"^```sql\s*",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"^```\s*",
        "",
        response
    )

    response = re.sub(
        r"\s*```$",
        "",
        response
    )

    return response.strip()


def generate_sql(question: str) -> str:

    prompt = f"""
You are a Text-to-SQL assistant for a Sephora Business Intelligence system.

Convert the user's question into ONE valid SQLite query.

DATABASE SCHEMA:
{DATABASE_SCHEMA}

GENERAL RULES:

- Use valid SQLite syntax.
- Generate only a read-only SELECT query.
- You may use WITH clauses and common table expressions (CTEs)
  when needed.
- Never modify the database.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
  PRAGMA or ATTACH.
- Never invent tables or columns.
- Use the exact table and column names from the schema.
- Use JOIN only when necessary.
- products.product_id = reviews.product_id.
- product_rating is the overall product rating of a product.
- review_rating is the rating given in an individual customer review.
- Return SQL only.
- Do not explain the SQL.
- Do not use markdown.


AGGREGATION AND METRIC ACCURACY:

- Identify the entity being measured before calculating an
  aggregate: products, reviews, brands, categories, or another
  entity supported by the schema.

- Choose the correct aggregation level for each metric.

- Calculate product-level metrics from product-level data and
  review-level metrics from review-level data.

- When joining tables with a one-to-many relationship, avoid
  accidentally counting or averaging the same entity multiple
  times because of duplicated rows.

- When product-level metrics and review-level metrics are both
  needed, prefer aggregating reviews per product in a separate
  CTE or subquery before joining the review statistics to
  products.

- Do not use COUNT(*) after a one-to-many join as a product
  count unless the joined rows genuinely represent the entity
  the user wants counted.

- Use COUNT(DISTINCT entity_id) when distinct entity counting
  is required.

- Never return only the grouping column when the user requests
  an aggregate such as average, count, sum, minimum, or maximum.

- Always include the requested calculated metric in SELECT
  with a clear and descriptive alias.

- Exclude NULL values where necessary for a meaningful
  calculation, using the appropriate SQL aggregate behavior.


FILTERING AND COMPARATIVE METRICS:

- Determine which filters define the population of a metric
  and which filters are applied only to the final results.

- Do not let final-result filters accidentally change the
  population used to calculate a reference metric.

- When comparing an entity against a reference metric, such
  as an overall average, category average, brand average,
  or another benchmark, calculate the reference metric
  independently over its intended population.

- Use a CTE or subquery for the reference metric when this
  prevents the final-result filters, HAVING conditions,
  or joins from changing its population.

- Apply filters to the reference population when they are
  genuinely part of the user's requested definition of
  that reference metric.

- Do not calculate a benchmark using only the rows that
  remain after filtering for entities that exceed or fall
  below that same benchmark.

- If a reference metric is used to filter results and is
  also returned in SELECT, use the same definition and
  underlying calculation in both places.

- Use window functions only when their partition and input
  rows match the intended population of the calculation.

- Do not use a window function on an already-filtered or
  already-aggregated result if doing so changes the meaning
  of the requested metric.


REVIEWS AND CUSTOMER FEEDBACK:

- Keep product ratings and individual review ratings distinct.

- When calculating review counts, count review records at
  the appropriate review-level granularity.

- When calculating negative or positive review counts,
  use the review-rating criteria requested by the user.

- If the user does not provide a rating threshold and a
  rating-based definition is necessary, use review_rating
  <= 2 for low-rated reviews and review_rating >= 4 for
  high-rated reviews.

- Do not interpret a low review rating alone as proof of
  a specific complaint described in the review text.

- A count of low-rated reviews is not necessarily a count
  of all customer complaints.

- Do not multiply product-level values when joining
  products with multiple review records.


CATEGORY FILTERING:

- Product types and product categories such as skincare,
  moisturizer, cleanser, serum, shampoo, lipstick, etc.
  may appear in primary_category, secondary_category,
  or tertiary_category.

- When the user refers to a product type or category,
  search all three category columns using LOWER(...) LIKE,
  unless the user explicitly refers to one specific
  database category field.

- When a category filter is also used to calculate a
  benchmark such as an average price or average rating,
  use the same category definition in the benchmark
  population.

Example logic:

(
    LOWER(primary_category) LIKE '%skincare%'
    OR LOWER(secondary_category) LIKE '%skincare%'
    OR LOWER(tertiary_category) LIKE '%skincare%'
)


RESULT DISPLAY RULES:

- When the user asks for a list of products, return at most
  10 rows by default, unless the user explicitly requests
  a different number.

- If the user explicitly asks for all matching products,
  do not apply the default LIMIT.

- When listing products that match a condition, include
  COUNT(*) OVER() AS total_count when appropriate, so the
  total number of matching final-result rows is available
  before applying LIMIT.

- Calculate total_count over the final set of distinct
  result entities, after the intended filters and grouping,
  but before LIMIT.

- If the user explicitly asks for the top N or bottom N
  results, respect N. Do not confuse the number requested
  by the user with the total number of matching entities.

- Do not apply LIMIT to a query returning a single aggregate
  result, such as a single COUNT(*), AVG(), SUM(), MIN(),
  or MAX().

- When LIMIT is used to display a subset of results, use
  a meaningful ORDER BY where appropriate so the selected
  rows are deterministic.

- Do not include unnecessary columns.

- Use clear aliases for calculated columns.


FINAL SQL CHECK:

Before returning SQL, verify that:

1. All referenced tables and columns exist in the schema.
2. Each aggregate measures the intended entity and population.
3. One-to-many joins do not inflate counts, sums, or averages.
4. Reference metrics are calculated over the correct population.
5. Filtering and displayed metrics use consistent definitions.
6. The query answers the actual user question.
7. The query is valid SQLite and read-only.

Return only the final SQL query.


USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0
        )
    )

    return clean_sql(response.text)