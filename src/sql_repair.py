"""Shared SQL generation -> SQLite repair -> intent review -> execution.

Used by SQL and HYBRID routes. Semantic review adds one Gemini call per new
analysis; replays from Streamlit session history do not regenerate the result.
"""

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.database import SQLValidationError, execute_query, validate_query
from src.schema import DATABASE_SCHEMA
from src.sql_semantic_check import SQLIntentError, review_sql_intent
from src.text_to_sql import clean_sql


load_dotenv()
client = genai.Client()


def repair_sql(question: str, broken_sql: str, sqlite_error: str) -> str:
    """Repair SQLite compilation errors; never repair SQL blocked as unsafe."""
    prompt = f"""
You repair SQL for a read-only SQLite Business Intelligence system.
Fix the concrete SQLite compilation error while remaining faithful to the
original user's requirements. Return exactly ONE read-only SELECT statement,
optionally beginning with WITH. Return SQL only, without markdown.

Do not change required entities, filters, groupings, benchmark population,
metrics, ordering, or explicit result count. Never invent tables, columns,
data, or fixed thresholds. No PRAGMA, ATTACH, writes, or multiple statements.

DATABASE SCHEMA:
{DATABASE_SCHEMA}

ORIGINAL USER QUESTION:
{question}

INVALID SQL:
{broken_sql}

SQLITE ERROR:
{sqlite_error}
"""
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    if not response.text:
        raise SQLValidationError("Gemini returned no SQL while repairing a query.")
    return clean_sql(response.text)


def _validate_with_repair(question: str, sql: str, max_repairs: int) -> str:
    """Compile the statement first; only SQLValidationError is repairable."""
    for attempt in range(max_repairs + 1):
        try:
            validate_query(sql)
            return sql
        except SQLValidationError as exc:
            if attempt == max_repairs:
                raise SQLValidationError(
                    f"SQL remains invalid after {max_repairs} syntax repairs: {exc}"
                ) from exc
            sql = repair_sql(question, sql, str(exc))

    raise RuntimeError("Unexpected SQL repair state")


def generate_and_execute_sql(
    question: str,
    generator,
    max_repairs: int = 2,
    *,
    mode: str = "SQL",
    max_intent_revisions: int = 2,
):
    """Return (approved_sql, dataframe), or stop instead of executing bad SQL.

    Syntax is checked before each semantic review. Each semantic revision is
    rechecked by SQLite and the intent reviewer. Read-only restrictions are
    enforced by src.database, including for reviewer-generated revisions.
    """
    if max_repairs < 0 or max_intent_revisions < 0:
        raise ValueError("Repair limits cannot be negative.")

    sql = generator(question)

    for revision in range(max_intent_revisions + 1):
        sql = _validate_with_repair(question, sql, max_repairs)
        approved, corrected_sql, reason = review_sql_intent(
            question=question,
            sql=sql,
            mode=mode,
        )

        if approved:
            return sql, execute_query(sql)

        if not corrected_sql:
            raise SQLIntentError(
                "The generated SQL did not fully match the question, and "
                f"a safe correction was not available. Details: {reason}"
            )

        if revision == max_intent_revisions:
            raise SQLIntentError(
                "The generated SQL could not be confirmed as matching the "
                f"question after {max_intent_revisions} revisions. "
                f"Last issue: {reason}"
            )

        if corrected_sql.strip() == sql.strip():
            raise SQLIntentError(
                "The SQL reviewer flagged an issue but returned the same SQL. "
                f"Details: {reason}"
            )

        sql = corrected_sql

    raise RuntimeError("Unexpected SQL intent review state")
