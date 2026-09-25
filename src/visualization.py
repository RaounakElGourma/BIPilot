import plotly.express as px


def create_chart(data, question=""):

    if data is None or data.empty:
        return None, None

    df = data.copy()
    question = question.lower()

    # ----------------------------------
    # 1. PRICE CHART
    # ----------------------------------

    if (
        "product_name" in df.columns
        and "price_usd" in df.columns
        and any(word in question for word in ["price", "expensive", "cheap"])
    ):

        chart_data = (
            df.dropna(subset=["price_usd"])
            .sort_values("price_usd", ascending=True)
            .tail(10)
        )

        fig = px.bar(
            chart_data,
            x="price_usd",
            y="product_name",
            orientation="h",
            labels={
                "price_usd": "Price (USD)",
                "product_name": "Product"
            },
            hover_data=[
                col for col in ["brand_name", "product_rating"]
                if col in chart_data.columns
            ]
        )

        fig.update_layout(
            yaxis_title="",
            xaxis_title="Price (USD)"
        )

        return "Price Comparison", fig


    # ----------------------------------
    # 2. RATING CHART
    # ----------------------------------

    if (
        "product_name" in df.columns
        and "product_rating" in df.columns
        and any(word in question for word in ["rating", "rated", "best"])
    ):

        chart_data = (
            df.dropna(subset=["product_rating"])
            .sort_values("product_rating", ascending=True)
            .tail(10)
        )

        fig = px.bar(
            chart_data,
            x="product_rating",
            y="product_name",
            orientation="h",
            labels={
                "product_rating": "Rating",
                "product_name": "Product"
            }
        )

        fig.update_layout(
            yaxis_title="",
            xaxis_title="Rating"
        )

        return "Rating Comparison", fig


    # ----------------------------------
    # 3. AGGREGATED BUSINESS RESULTS
    # ----------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    aggregation_columns = [
        col for col in numeric_columns
        if any(keyword in col.lower() for keyword in [
            "count",
            "average",
            "avg",
            "total",
            "sum"
        ])
    ]

    category_columns = [
        col for col in [
            "brand_name",
            "primary_category",
            "secondary_category",
            "tertiary_category"
        ]
        if col in df.columns
    ]

    if (
        "product_name" not in df.columns
        and aggregation_columns
        and category_columns
        and df[aggregation_columns[0]].nunique() > 1
    ):

        value_column = aggregation_columns[0]
        category_column = category_columns[0]

        chart_data = (
            df.dropna(subset=[value_column])
            .sort_values(value_column, ascending=True)
            .tail(10)
        )

        fig = px.bar(
            chart_data,
            x=value_column,
            y=category_column,
            orientation="h",
            labels={
                value_column: value_column.replace("_", " ").title(),
                category_column: ""
            }
            )

        
        fig.update_layout(
            height=500,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=40
        )
    )

        return "Data Comparison", fig


    return None, None