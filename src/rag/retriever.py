from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]

INDEX_PATH = BASE_DIR / "vector_store" / "reviews.index"
METADATA_PATH = BASE_DIR / "vector_store" / "reviews_metadata.csv"


# Load FAISS index
index = faiss.read_index(str(INDEX_PATH))


# Load review metadata
metadata = pd.read_csv(METADATA_PATH)


# Load the SAME embedding model used when building the index
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def retrieve_reviews(
    question: str,
    top_k: int = 5,
    min_rating=None,
    max_rating=None,
    product_ids=None
):
    """
    Retrieve semantically relevant reviews.

    When specific products are provided, distribute the retrieved
    reviews across those products instead of allowing one product
    to dominate the entire review sample.

    Rating filters are optional and are only applied when supplied.
    """

    # --------------------------------------------------------
    # 1. Handle empty requests
    # --------------------------------------------------------

    if top_k <= 0 or index.ntotal == 0:
        return metadata.iloc[0:0].copy()

    if len(metadata) != index.ntotal:
        raise ValueError(
            "FAISS index and review metadata have different lengths. "
            "They must be generated from the same ordered review dataset."
        )

    if product_ids is not None and len(product_ids) == 0:
        return metadata.iloc[0:0].copy()

    # --------------------------------------------------------
    # 2. Generate the question embedding
    # --------------------------------------------------------

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # --------------------------------------------------------
    # 3. Retrieve candidate reviews
    # --------------------------------------------------------

    if product_ids is not None:

        # Search the complete index so that filtering by product
        # does not accidentally discard potentially relevant
        # reviews before they can be selected.
        #
        # Note: this is exhaustive and may become slow for
        # very large review datasets.
        candidate_k = index.ntotal

    else:

        # Retrieve extra candidates when optional rating
        # filters might discard some of the initial results.
        candidate_k = min(
            max(top_k * 30, 200),
            index.ntotal
        )

    scores, indices = index.search(
        question_embedding,
        candidate_k
    )

    # FAISS can return -1 for an unavailable result.
    valid = indices[0] >= 0

    results = metadata.iloc[
        indices[0][valid]
    ].copy()

    results["similarity_score"] = scores[0][valid]

    # Preserve the original semantic retrieval order.
    results["_retrieval_order"] = range(len(results))

    # --------------------------------------------------------
    # 4. Filter by selected products
    # --------------------------------------------------------

    if product_ids is not None:

        selected_ids = {
            str(pid).strip()
            for pid in product_ids
        }

        results["_product_key"] = (
            results["product_id"]
            .astype(str)
            .str.strip()
        )

        results = results[
            results["_product_key"].isin(selected_ids)
        ].copy()

    # --------------------------------------------------------
    # 5. Apply optional rating filters
    # --------------------------------------------------------

    if min_rating is not None:

        results = results[
            results["review_rating"] >= min_rating
        ].copy()

    if max_rating is not None:

        results = results[
            results["review_rating"] <= max_rating
        ].copy()

    if results.empty:
        return results.drop(
            columns=["_retrieval_order", "_product_key"],
            errors="ignore"
        )

    # --------------------------------------------------------
    # 6. Select the final review sample
    # --------------------------------------------------------

    if product_ids is None:

        # For general RAG questions, preserve semantic ranking.
        selected = results.head(top_k).copy()

    else:

        # For Hybrid questions, retrieve the most relevant
        # review from each product before selecting additional
        # reviews from the same product.
        #
        # This improves product coverage while preserving the
        # semantic order within each selection round.

        results["_product_rank"] = (
            results.groupby("_product_key", sort=False)
            .cumcount()
        )

        selected = (
            results.sort_values(
                by=["_product_rank", "_retrieval_order"],
                kind="stable"
            )
            .head(top_k)
            .copy()
        )

    # --------------------------------------------------------
    # 7. Remove temporary columns
    # --------------------------------------------------------

    selected = selected.drop(
        columns=[
            "_retrieval_order",
            "_product_key",
            "_product_rank"
        ],
        errors="ignore"
    )

    return selected.reset_index(drop=True)
