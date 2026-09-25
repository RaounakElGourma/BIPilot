DATABASE_SCHEMA = """
Database: Sephora Product & Customer Review Analytics

TABLE: products

Columns:
- product_id TEXT: unique product identifier
- product_name TEXT: product name
- brand_name TEXT: brand name
- loves_count INTEGER: number of users who loved the product
- product_rating REAL: average Sephora product rating
- review_count REAL: total number of product reviews
- price_usd REAL: product price in USD
- limited_edition INTEGER: 1 if limited edition, otherwise 0
- is_new INTEGER: 1 if the product is new, otherwise 0
- online_only INTEGER: 1 if available only online
- out_of_stock INTEGER: 1 if currently out of stock
- sephora_exclusive INTEGER: 1 if Sephora exclusive
- primary_category TEXT: main product category
- secondary_category TEXT: secondary category
- tertiary_category TEXT: detailed product category


TABLE: reviews

Columns:
- author_id TEXT: reviewer identifier
- product_id TEXT: product being reviewed
- review_rating INTEGER: rating given by the customer from 1 to 5
- is_recommended REAL: 1 if recommended, 0 otherwise
- helpfulness REAL: helpfulness score
- total_feedback_count INTEGER
- total_neg_feedback_count INTEGER
- total_pos_feedback_count INTEGER
- submission_time TEXT: review submission date
- review_text TEXT: customer review text
- review_title TEXT: review title
- skin_tone TEXT
- eye_color TEXT
- skin_type TEXT
- hair_color TEXT


RELATIONSHIP:

products.product_id = reviews.product_id
"""