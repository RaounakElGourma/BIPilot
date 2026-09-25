import html
import re

import pandas as pd
import streamlit as st

from src.bipilot import ask_bipilot
from src.database import execute_query
from src.visualization import create_chart


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BIPilot",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Each record holds a question and its full result (including data and reviews).
# This is session-only history: no database writes or extra Gemini calls on replay.
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "active_analysis" not in st.session_state:
    st.session_state.active_analysis = None

if "next_analysis_id" not in st.session_state:
    st.session_state.next_analysis_id = 1


# ============================================================
# CSV EXPORT HELPERS
# ============================================================

def complete_export_query(sql: str, question: str, preview_data: pd.DataFrame):
    """Return an unbounded query ONLY for a previewed list with a known total.

    Never remove a user-requested top-N ranking or a LIMIT embedded in
    a subquery. Queries we cannot safely expand use the displayed rows.
    """
    if not sql or preview_data is None or preview_data.empty:
        return None
    if "total_count" not in preview_data.columns:
        return None

    try:
        total = int(preview_data["total_count"].iloc[0])
    except (TypeError, ValueError, OverflowError):
        return None
    if total <= len(preview_data):
        return None

    # A requested 'top 10' must still export 10, not every product.
    explicit_top_n = re.search(
        r"\b(?:top|first|last|bottom)\s+\d+\b"
        r"|\b\d+\s+(?:most|least|best|worst|highest|lowest|expensive|cheapest)\b"
        r"|\b(?:show|list|give)\s+(?:me\s+)?(?:the\s+)?\d+\s+"
        r"(?:products|items|brands|reviews)\b",
        question,
        flags=re.IGNORECASE,
    )
    if explicit_top_n:
        return None

    # Supported case: a single final `LIMIT 10` (optionally followed by ;).
    # We deliberately do not rewrite LIMIT/OFFSET, nested limits, or complex
    # expressions. In such cases, export only the displayed rows.
    limited = re.search(r"\s+LIMIT\s+\d+\s*;?\s*$", sql, re.IGNORECASE)
    if not limited:
        return None

    unbounded = sql[:limited.start()].strip().rstrip(";").strip()
    if not re.match(r"^(?:SELECT|WITH)\b", unbounded, flags=re.IGNORECASE):
        return None
    if ";" in unbounded:  # Multiple statements: never re-execute them.
        return None
    return unbounded + ";"


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Create a CSV compatible with Excel, without UI helper columns."""
    clean = frame.drop(columns=["total_count"], errors="ignore").copy()
    # Product names / reviews are external text. Prevent spreadsheet software
    # from interpreting a text cell as a formula when opening the CSV.
    for column in clean.select_dtypes(include=["object", "string"]).columns:
        clean[column] = clean[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
            else value
        )
    return clean.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# GLOBAL STYLE
# ============================================================

st.html(
    """
    <style>

    /* ==========================================
       APP BACKGROUND
    ========================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 8% 5%,
                rgba(124, 92, 255, 0.22),
                transparent 28%
            ),
            radial-gradient(
                circle at 92% 12%,
                rgba(255, 79, 154, 0.13),
                transparent 26%
            ),
            radial-gradient(
                circle at 60% 100%,
                rgba(106, 55, 180, 0.12),
                transparent 28%
            ),
            #110B18;

        color: #F8F6FB;
    }


    /* ==========================================
       MAIN CONTENT
    ========================================== */

    .block-container {
        max-width: 1500px;
        padding-top: 2.4rem;
        padding-bottom: 5rem;
    }

    html, body, [class*="css"] {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    p {
        line-height: 1.7;
    }


    /* ==========================================
       HIDE DEFAULT STREAMLIT UI
    ========================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ==========================================
       INPUT
    ========================================== */

    div[data-testid="stChatInput"] {
        background: #1D1426 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 16px !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #F8F6FB !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #91869B !important;
    }

    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] svg {
        display: none;
    }

    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]::after {
        content: "→";
        font-size: 18px;
    }

    div[data-testid="stTextInput"] input {
        background: #1D1426 !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 14px !important;

        color: #F8F6FB !important;

        padding: 19px 18px !important;
        font-size: 16px !important;

        transition: all 0.2s ease;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8E8499 !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #8B6CFF !important;

        box-shadow:
            0 0 0 3px rgba(139, 108, 255, 0.14) !important;
    }


    /* ==========================================
       PRIMARY BUTTON — ANALYZE
    ========================================== */

    div.stButton > button[kind="primary"] {
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        min-height: 36px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        background: #8B5CF6 !important;
        border: none !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow:
            0 4px 12px rgba(139, 92, 246, 0.25);
    }

    div.stButton > button[kind="primary"]:hover {
        background: #9D6CFF !important;
        transform: scale(1.05);
        color: white !important;
    }

    /* ==========================================
       DATAFRAME
    ========================================== */

    div[data-testid="stDataFrame"] {
        border:
            1px solid rgba(255,255,255,0.08);

        border-radius: 14px;

        overflow: hidden;

        background: #1A1222;
    }


    /* ==========================================
       EXPANDERS
    ========================================== */

    div[data-testid="stExpander"] {
        background: #191121;

        border:
            1px solid rgba(255,255,255,0.08);

        border-radius: 14px;

        overflow: hidden;

        margin-top: 10px;
    }

    div[data-testid="stExpander"] details summary {
        color: #E8E2ED !important;
    }


    /* ==========================================
       CODE
    ========================================== */

    code {
        color: #D7C7FF !important;
    }


    /* Generated SQL: dark editor matching the rest of BIPilot. */
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] code {
        background: #150F1B !important;
        color: #DDD1EB !important;
        border-radius: 11px !important;
        font-size: 13px !important;
    }

    /* ==========================================
       DIVIDERS
    ========================================== */

    hr {
        border: none !important;

        border-top:
            1px solid rgba(255,255,255,0.07) !important;

        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }


    /* ==========================================
       SPINNER
    ========================================== */

    div[data-testid="stSpinner"] {
        color: #C9BFFF;
    }


    /* ==========================================
       ALERTS
    ========================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ==========================================
       AI INSIGHT CARD
    ========================================== */

    .st-key-insight_card {

        background:
            linear-gradient(
                135deg,
                #251932 0%,
                #1D1528 100%
            );

        border:
            1px solid rgba(168, 85, 247, 0.22);

        border-radius: 18px;

        padding: 10px 28px 26px 28px;

        margin-top: 25px;
        margin-bottom: 30px;

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.12);
    }


    /* AI Insight text */

    .st-key-insight_card p,
    .st-key-insight_card li {

        color: #D8D0E0;

        font-size: 15px;

        line-height: 1.8;
    }


    /* Bold text */

    .st-key-insight_card strong {

        color: #F5F0FA;

        font-weight: 650;
    }


    /* Headings */

    .st-key-insight_card h2,
    .st-key-insight_card h3 {

        color: #F8F5FA;

        margin-top: 22px;
        margin-bottom: 12px;
    }


    /* Lists */

    .st-key-insight_card ul {

        padding-left: 24px;

    }

    /* ==========================================
       SIDEBAR / SESSION HISTORY
    ========================================== */

    section[data-testid="stSidebar"] {
        background: #1A1222 !important;
        border-right: 1px solid rgba(255,255,255,0.075) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.4rem;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 17px !important;
        font-weight: 650 !important;
        letter-spacing: -0.3px;
        color: #F8F6FB !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #9F93AC !important;
        font-size: 12px !important;
        line-height: 1.5 !important;
    }

    /* History entries are slim, one-line labels; the full question
       remains accessible in the native button tooltip. */
    section[data-testid="stSidebar"] [class*="st-key-history_"] button {
        display: flex !important;
        width: 100% !important;
        min-height: 39px !important;
        padding: 9px 11px !important;
        border-radius: 10px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: #C7BBD1 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        box-shadow: none !important;
        transition: background .16s ease, color .16s ease;
    }

    section[data-testid="stSidebar"] [class*="st-key-history_"] button p {
        display: block !important;
        width: 100% !important;
        min-width: 0 !important;
        margin: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        font-size: 12.5px !important;
        line-height: 1.3 !important;
        font-weight: 450 !important;
        text-align: left !important;
    }

    section[data-testid="stSidebar"] [class*="st-key-history_"] button:hover {
        background: #30213D !important;
        color: #FFFFFF !important;
    }

    /* Give New analysis a distinct, quiet action style. */
    section[data-testid="stSidebar"] .st-key-new_analysis button {
        min-height: 39px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(177, 140, 255, .25) !important;
        background: rgba(140, 91, 220, .10) !important;
        color: #E6D9FF !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .st-key-new_analysis button:hover {
        background: rgba(140, 91, 220, .19) !important;
        border-color: rgba(177, 140, 255, .42) !important;
    }

    /* Compact result table and quiet metadata label. */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,.10) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background: #17111D !important;
    }
    div[data-testid="stDataFrame"] + div {
        color: #AA9DB7;
    }

    </style>
    """
)


# ============================================================
# CSS-ONLY ICONS
# ============================================================

ICONS = {

    "logo": """
        <div style="
            width:24px;
            height:24px;
            display:flex;
            align-items:flex-end;
            justify-content:center;
            gap:3px;
        ">
            <span style="
                width:4px;
                height:10px;
                background:#9A7CFF;
                border-radius:4px;
            "></span>

            <span style="
                width:4px;
                height:18px;
                background:#B162FF;
                border-radius:4px;
            "></span>

            <span style="
                width:4px;
                height:14px;
                background:#E057AA;
                border-radius:4px;
            "></span>

            <span style="
                width:4px;
                height:22px;
                background:#FF659F;
                border-radius:4px;
            "></span>
        </div>
    """,

    "sparkles": """
        <div style="
            width:21px;
            height:21px;
            position:relative;
        ">
            <span style="
                position:absolute;
                width:11px;
                height:11px;
                left:1px;
                top:1px;
                border-radius:3px;
                background:#9A7CFF;
                transform:rotate(45deg);
            "></span>

            <span style="
                position:absolute;
                width:6px;
                height:6px;
                right:1px;
                bottom:1px;
                border-radius:2px;
                background:#F15CA9;
                transform:rotate(45deg);
            "></span>
        </div>
    """,

    "chart": """
        <div style="
            width:21px;
            height:21px;
            display:flex;
            align-items:flex-end;
            justify-content:center;
            gap:3px;
        ">
            <span style="
                width:3px;
                height:8px;
                border-radius:3px;
                background:#956FFF;
            "></span>

            <span style="
                width:3px;
                height:16px;
                border-radius:3px;
                background:#B55BEA;
            "></span>

            <span style="
                width:3px;
                height:12px;
                border-radius:3px;
                background:#D454BE;
            "></span>

            <span style="
                width:3px;
                height:20px;
                border-radius:3px;
                background:#F2559F;
            "></span>
        </div>
    """,

    "table": """
        <div style="
            width:19px;
            height:19px;
            display:grid;
            grid-template-columns:1fr 1fr;
            grid-template-rows:1fr 1fr;
            gap:3px;
        ">
            <span style="
                background:#9A7CFF;
                border-radius:3px;
            "></span>

            <span style="
                background:#B962E9;
                border-radius:3px;
            "></span>

            <span style="
                background:#D85ABF;
                border-radius:3px;
            "></span>

            <span style="
                background:#F45A9D;
                border-radius:3px;
            "></span>
        </div>
    """
}


# ============================================================
# REUSABLE SECTION TITLE
# ============================================================

def section_title(icon_name: str, title: str):

    safe_title = html.escape(
        str(title)
    )

    st.html(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            margin-top:26px;
            margin-bottom:18px;
        ">

            <div style="
                width:42px;
                height:42px;

                border-radius:12px;

                display:flex;
                align-items:center;
                justify-content:center;

                background:
                    linear-gradient(
                        145deg,
                        rgba(124,92,255,0.20),
                        rgba(255,79,154,0.12)
                    );

                border:
                    1px solid rgba(255,255,255,0.07);

                flex-shrink:0;
            ">
                {ICONS[icon_name]}
            </div>

            <div style="
                color:#F7F3FA;
                font-size:25px;
                font-weight:700;
                letter-spacing:-0.5px;
            ">
                {safe_title}
            </div>

        </div>
        """
    )


# ============================================================
# ROUTE BADGE
# ============================================================

def route_badge(route: str):

    safe_route = html.escape(
        str(route)
    )

    st.html(
        f"""
        <div style="
            margin-top:22px;
            margin-bottom:8px;
        ">

            <span style="
                display:inline-flex;
                align-items:center;

                padding:7px 13px;

                border-radius:999px;

                color:#E8DFFF;

                background:
                    linear-gradient(
                        135deg,
                        rgba(124,92,255,0.18),
                        rgba(255,79,154,0.10)
                    );

                border:
                    1px solid rgba(155,120,255,0.20);

                font-size:11px;
                font-weight:750;
                letter-spacing:0.8px;
            ">
                {safe_route} ANALYSIS
            </span>

        </div>
        """
    )


# ============================================================
# QUESTION HISTORY (collapsible Streamlit sidebar)
# ============================================================

with st.sidebar:
    st.markdown("### History")
    st.caption("Your previous analyses · this session")

    if st.button("＋ New analysis", key="new_analysis", use_container_width=True):
        st.session_state.active_analysis = None
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    if not st.session_state.analysis_history:
        st.caption("Your questions will appear here after your first analysis.")
    else:
        for record in reversed(st.session_state.analysis_history):
            full_question = record["question"]
            active = st.session_state.active_analysis
            is_active = active is not None and active["id"] == record["id"]
            # CSS keeps the label on one line; native tooltip shows it all.
            if is_active:
                st.html(
                    f"""<style>
                    section[data-testid="stSidebar"] .st-key-history_{record['id']} button {{
                        background: rgba(149, 97, 237, .18) !important;
                        border-color: rgba(177, 140, 255, .30) !important;
                        color: #FFFFFF !important;
                    }}
                    </style>"""
                )

            if st.button(
                full_question,
                key=f"history_{record['id']}",
                use_container_width=True,
                help=full_question,
            ):
                # Reuse the previously generated answer, SQL, data, and reviews.
                st.session_state.active_analysis = record
                st.session_state.pending_question = None
                st.rerun()


# ============================================================
# HEADER / HERO
# ============================================================

st.html(
    f"""
    <div style="
        position:relative;

        padding:
            34px 36px 30px 36px;

        margin-bottom:28px;

        border-radius:22px;

        overflow:hidden;

        background:
            linear-gradient(
                145deg,
                rgba(34,22,47,0.96),
                rgba(23,15,32,0.97)
            );

        border:
            1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.25);
    ">

        <div style="
            position:absolute;
            width:260px;
            height:260px;
            border-radius:50%;

            right:-70px;
            top:-120px;

            background:
                radial-gradient(
                    circle,
                    rgba(255,79,154,0.19),
                    transparent 68%
                );
        "></div>

        <div style="
            position:absolute;
            width:280px;
            height:280px;
            border-radius:50%;

            left:38%;
            bottom:-210px;

            background:
                radial-gradient(
                    circle,
                    rgba(124,92,255,0.23),
                    transparent 70%
                );
        "></div>


        <div style="
            display:flex;
            align-items:center;
            gap:15px;
            position:relative;
            z-index:2;
        ">

            <div style="
                width:52px;
                height:52px;

                display:flex;
                align-items:center;
                justify-content:center;

                border-radius:15px;

                background:
                    linear-gradient(
                        145deg,
                        rgba(124,92,255,0.21),
                        rgba(255,79,154,0.14)
                    );

                border:
                    1px solid rgba(255,255,255,0.08);
            ">
                {ICONS["logo"]}
            </div>


            <div>
                <div style="
                    font-size:39px;
                    line-height:1;
                    font-weight:780;
                    letter-spacing:-1.4px;
                    color:#FCF9FD;
                ">
                    BIPilot
                </div>

                <div style="
                    margin-top:8px;

                    font-size:12px;
                    font-weight:700;
                    letter-spacing:1.2px;

                    color:#A692BC;
                ">
                    BUSINESS INTELLIGENCE COPILOT
                </div>
            </div>

        </div>


        <div style="
            max-width:760px;
            margin-top:20px;

            color:#B9AEC2;
            font-size:16px;
            line-height:1.6;

            position:relative;
            z-index:2;
        ">
            Ask your data. Explore product performance. Understand what customers really think.
        </div>


        <div style="
            display:flex;
            align-items:center;
            gap:18px;
            margin-top:22px;
            flex-wrap:wrap;

            color:#AFA2B8;
            font-size:13px;
            font-weight:600;

            position:relative;
            z-index:2;
        ">

            <span style="display:flex; align-items:center; gap:7px;">
                <span style="
                    width:6px;
                    height:6px;
                    border-radius:50%;
                    background:#9B6CFF;
                "></span>
                Query your data
            </span>

            <span style="display:flex; align-items:center; gap:7px;">
                <span style="
                    width:6px;
                    height:6px;
                    border-radius:50%;
                    background:#D95FA7;
                "></span>
                Understand reviews
            </span>

            <span style="display:flex; align-items:center; gap:7px;">
                <span style="
                    width:6px;
                    height:6px;
                    border-radius:50%;
                    background:#C084FC;
                "></span>
                Combine both
            </span>

        </div>

    </div>
    """
)


# ============================================================
# ASK SECTION
# ============================================================

st.html(
    """
    <div style="
        font-size:12px;
        font-weight:750;
        letter-spacing:1px;
        color:#94869F;

        margin-bottom:10px;
    ">
        TRY A QUESTION
    </div>
    """
)


# ============================================================
# SUGGESTION BUTTONS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    if st.button(
        "Most expensive products",
        key="suggestion_1",
        use_container_width=True
    ):

        st.session_state.pending_question = (
            "Show me the 10 most expensive products."
        )


with col2:

    if st.button(
        "Customer complaints about moisturizers",
        key="suggestion_2",
        use_container_width=True
    ):

        st.session_state.pending_question = (
            "Why do customers complain about moisturizers?"
        )


with col3:

    if st.button(
        "Highly rated products with complaints",
        key="suggestion_3",
        use_container_width=True
    ):

        st.session_state.pending_question = (
            "Which highly rated skincare products still receive complaints?"
        )


st.write("")


# ============================================================
# QUESTION INPUT
# ============================================================

typed_question = st.chat_input(
    "Ask about products, brands, ratings, prices, customer feedback..."
)

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    analyze = True

elif typed_question:
    question = typed_question
    analyze = True

else:
    question = None
    analyze = False

# ============================================================
# RUN BIPILOT
# ============================================================

# A new question invokes the pipeline once; selecting history only renders cached results.
if analyze or st.session_state.active_analysis is not None:

    if analyze and not question.strip():

        st.warning(
            "Enter a business question first."
        )

    else:

        try:

            if analyze:
                with st.spinner(
                    "Analyzing product data and customer feedback..."
                ):
                    result = ask_bipilot(question)

                # Store the complete result so earlier views need no new model calls.
                record = {
                    "id": st.session_state.next_analysis_id,
                    "question": question,
                    "result": result,
                }
                st.session_state.next_analysis_id += 1
                st.session_state.analysis_history.append(record)
                st.session_state.active_analysis = record

                # Refresh immediately so the sidebar displays the newly saved
                # question. On the next run, the cached result below is rendered
                # without calling ask_bipilot a second time.
                st.rerun()
            else:
                # The currently selected analysis was already computed.
                record = st.session_state.active_analysis
                question = record["question"]
                result = record["result"]


            # =================================================
            # ROUTE
            # =================================================

            route = result.get(
                "route",
                "UNKNOWN"
            )

            # =============================================
            # DISPLAY USER QUESTION
            # =============================================

            st.markdown(
                f"""
                <div style="
                    display: flex;
                    justify-content: flex-end;
                    margin: 15px 0 20px 0;
                ">
                    <div style="
                        background: #382647;
                        color: #F8F4FC;
                        padding: 15px 22px;
                        border-radius: 20px 20px 6px 20px;
                        max-width: 75%;
                        width: fit-content;
                        font-size: 16px;
                        font-weight: 400;
                        line-height: 1.6;
                        overflow-wrap: anywhere;
                    ">
                        {html.escape(question)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            route_badge(
                route
            )


            # =================================================
            # AI INSIGHT
            # =================================================

            answer = result.get("answer")

            if answer:

                with st.container(key="insight_card"):

                    section_title(
                        "sparkles",
                        "AI Insight"
                    )

                    safe_answer = answer.replace("$", r"\$")

                    st.markdown(safe_answer)


            # =================================================
            # DATA
            # =================================================

            data = result.get(
                "data"
            )

            if (
                data is not None
                and isinstance(data, pd.DataFrame)
                and not data.empty
            ):

                # =============================================
                # CHART
                # =============================================

                chart_title, chart = create_chart(
                    data,
                    question
                )

                if chart is not None:

                    st.divider()

                    section_title(
                        "chart",
                        chart_title
                    )

                    # -----------------------------------------
                    # Identify bar charts
                    # -----------------------------------------

                    bar_traces = [
                        trace
                        for trace in chart.data
                        if trace.type == "bar"
                    ]

                    # -----------------------------------------
                    # Fix overlapping names and sort products
                    # -----------------------------------------

                    if (
                        len(bar_traces) == 1
                        and bar_traces[0].orientation == "h"
                    ):

                        trace = bar_traces[0]

                        # Associate each product with its value
                        products_and_values = list(
                            zip(trace.y, trace.x)
                        )

                        # Sort products from highest to lowest value
                        products_and_values.sort(
                            key=lambda item: float(item[1]),
                            reverse=True
                        )

                        # Extract sorted product names and values
                        original_names = [
                            str(name)
                            for name, value in products_and_values
                        ]

                        sorted_values = [
                            value
                            for name, value in products_and_values
                        ]

                        # Give each product a unique position
                        positions = list(range(len(original_names)))

                        trace.x = sorted_values
                        trace.y = positions

                        # Shorten long names
                        short_names = [
                            name[:42] + "..."
                            if len(name) > 42
                            else name
                            for name in original_names
                        ]

                        # Show complete names on hover
                        trace.hovertext = original_names

                        trace.hovertemplate = (
                            "<b>%{hovertext}</b>"
                            "<br>Value: %{x}"
                            "<extra></extra>"
                        )

                        # Display highest values at the top
                        chart.update_yaxes(
                            tickmode="array",
                            tickvals=positions,
                            ticktext=short_names,
                            autorange="reversed",
                            automargin=True
                        )

                    # -----------------------------------------
                    # Chart dimensions
                    # -----------------------------------------

                    number_of_rows = len(data)

                    chart_height = max(
                        340,
                        min(650, number_of_rows * 43 + 85)
                    )

                    # -----------------------------------------
                    # BIPilot chart theme
                    # -----------------------------------------

                    chart.update_layout(

                        height=chart_height,

                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",

                        font=dict(
                            family="Arial, sans-serif",
                            color="#C8BED0",
                            size=12
                        ),

                        margin=dict(
                            l=20,
                            r=75,
                            t=20,
                            b=45
                        ),

                        bargap=0.38,

                        showlegend=False,

                        xaxis=dict(

                            showgrid=True,

                            gridcolor="rgba(255,255,255,0.045)",

                            zeroline=False,

                            showline=False,

                            tickfont=dict(
                                color="#9F93AC",
                                size=11
                            ),

                            title_font=dict(
                                color="#B9AEC2",
                                size=12
                            ),

                            automargin=True
                        ),

                        hoverlabel=dict(
                            bgcolor="#251932",

                            bordercolor="#A855F7",

                            font=dict(
                                color="#FFFFFF",
                                size=12
                            )
                        )
                    )

                    chart.update_yaxes(
                        showgrid=False,
                        zeroline=False,
                        tickfont=dict(
                            color="#C8BED0",
                            size=11
                        ),
                        automargin=True
                    )

                    # -----------------------------------------
                    # Bar styling
                    # -----------------------------------------

                    for trace in chart.data:

                        if trace.type != "bar":
                            continue

                        # Purple bars
                        trace.marker.color = "#A855F7"

                        # Remove bar borders
                        trace.marker.line.width = 0

                        # Make bars thinner
                        trace.width = 0.58

                        # Show values next to bars
                        if trace.orientation == "h":

                            # Ratings need decimal values
                            if "rating" in chart_title.lower():

                                trace.texttemplate = "%{x:.2f}"

                            else:

                                trace.texttemplate = "%{x:,.0f}"

                        else:

                            if "rating" in chart_title.lower():

                                trace.texttemplate = "%{y:.2f}"

                            else:

                                trace.texttemplate = "%{y:,.0f}"

                        trace.textposition = "outside"

                        trace.textfont = dict(
                            color="#D8D0E0",
                            size=11
                        )

                        trace.cliponaxis = False

                    # -----------------------------------------
                    # Display chart
                    # -----------------------------------------

                    st.plotly_chart(
                        chart,
                        use_container_width=True,
                        config={
                            "displaylogo": False,
                            "responsive": True
                        }
                    )


                # =============================================
                # RESULTS TABLE
                # =============================================

                st.divider()

                section_title(
                    "table",
                    "Results"
                )

                display_data = data.copy()

                # Display the total number of matching records
                if "total_count" in data.columns and not data.empty:

                    total = int(data["total_count"].iloc[0])

                    displayed = len(data)

                    st.caption(
                        f"{total:,} matching products  ·  "
                        f"Showing {displayed:,} of {total:,}"
                    )


                # ---------------------------------------------
                # ROUND VALUES
                # ---------------------------------------------

                if "product_rating" in display_data.columns:

                    display_data[
                        "product_rating"
                    ] = display_data[
                        "product_rating"
                    ].round(2)


                if "average_rating" in display_data.columns:

                    display_data[
                        "average_rating"
                    ] = display_data[
                        "average_rating"
                    ].round(2)


                if "price_usd" in display_data.columns:

                    display_data[
                        "price_usd"
                    ] = display_data[
                        "price_usd"
                    ].round(2)


                # ---------------------------------------------
                # HIDE INTERNAL TOTAL_COUNT
                # ---------------------------------------------

                if "total_count" in display_data.columns:

                    display_data = display_data.drop(
                        columns=[
                            "total_count"
                        ]
                    )


                st.dataframe(
                    display_data,
                    use_container_width=True,
                    hide_index=True,
                    height=min(430, max(150, 44 + len(display_data) * 38))
                )

                # ---------------------------------------------
                # CSV EXPORT: PREVIEW VS. COMPLETE MATCHING DATA
                # ---------------------------------------------
                export_sql = complete_export_query(
                    result.get("sql"), question, data
                )
                export_key = f"csv_export_{record['id']}"
                file_name = f"bipilot_analysis_{record['id']}.csv"

                if export_sql:
                    # SQL is re-executed ONLY after the user requests the full
                    # export. The dashboard stays at 10 rows. No Gemini call.
                    total_rows = int(data["total_count"].iloc[0])
                    st.caption(
                        f"The dashboard shows {len(data):,} rows; "
                        f"the full CSV contains {total_rows:,} matching rows."
                    )
                    if export_key not in st.session_state:
                        if st.button(
                            "Prepare full CSV",
                            key=f"prepare_csv_{record['id']}",
                        ):
                            try:
                                with st.spinner("Preparing the complete CSV..."):
                                    export_data = execute_query(export_sql)
                                    # Check for silent query limits or changes
                                    # before labeling the file as 'complete'.
                                    if len(export_data) != total_rows:
                                        raise ValueError(
                                            f"Expected {total_rows:,} rows but got "
                                            f"{len(export_data):,}. Full export aborted."
                                        )
                                    st.session_state[export_key] = csv_bytes(export_data)
                            except Exception as export_error:
                                st.error(f"Could not prepare the full CSV: {export_error}")
                    if export_key in st.session_state:
                        st.download_button(
                            "Download full CSV",
                            data=st.session_state[export_key],
                            file_name=file_name,
                            mime="text/csv",
                            key=f"download_csv_{record['id']}",
                        )
                else:
                    # Explicit top-N questions, aggregate queries and queries
                    # whose LIMIT cannot be expanded safely export the actual
                    # returned rows, without making misleading 'all' claims.
                    st.download_button(
                        "Download CSV (displayed results)",
                        data=csv_bytes(display_data),
                        file_name=file_name,
                        mime="text/csv",
                        key=f"download_preview_{record['id']}",
                    )


            # =================================================
            # GENERATED SQL
            # =================================================

            sql = result.get(
                "sql"
            )

            if sql:

                st.write("")

                with st.expander(
                    "Generated SQL",
                    expanded=False
                ):

                    st.code(
                        sql,
                        language="sql"
                    )


            # =================================================
            # CUSTOMER EVIDENCE
            # =================================================

            sources = result.get(
                "sources"
            )

            if (
                sources is not None
                and isinstance(sources, pd.DataFrame)
                and not sources.empty
            ):

                with st.expander(
                    "Customer evidence",
                    expanded=False
                ):

                    for position, (_, row) in enumerate(
                        sources.iterrows()
                    ):

                        product_name = row.get(
                            "product_name",
                            "Unknown product"
                        )

                        brand_name = row.get(
                            "brand_name",
                            ""
                        )

                        review_rating = row.get(
                            "review_rating",
                            None
                        )

                        review_text = row.get(
                            "review_text",
                            ""
                        )


                        safe_product = html.escape(
                            str(product_name)
                        )

                        safe_brand = html.escape(
                            str(brand_name)
                        )

                        safe_review = html.escape(
                            str(review_text)
                        )


                        # -------------------------------------
                        # PRODUCT
                        # -------------------------------------

                        st.html(
                            f"""
                            <div style="
                                font-size:17px;
                                font-weight:700;
                                color:#F3EEF6;

                                margin-top:4px;
                                margin-bottom:6px;
                            ">
                                {safe_product}
                            </div>
                            """
                        )


                        # -------------------------------------
                        # BRAND
                        # -------------------------------------

                        if brand_name:

                            st.html(
                                f"""
                                <div style="
                                    font-size:13px;
                                    color:#998EA2;

                                    margin-bottom:8px;
                                ">
                                    {safe_brand}
                                </div>
                                """
                            )


                        # -------------------------------------
                        # RATING
                        # -------------------------------------

                        if review_rating is not None:

                            st.html(
                                f"""
                                <div style="
                                    display:inline-block;

                                    margin-bottom:12px;

                                    padding:5px 9px;

                                    border-radius:8px;

                                    background:
                                        rgba(124,92,255,0.10);

                                    color:#BCAEFF;

                                    font-size:12px;
                                    font-weight:650;
                                ">
                                    Customer rating · {review_rating}/5
                                </div>
                                """
                            )


                        # -------------------------------------
                        # REVIEW
                        # -------------------------------------

                        if review_text:

                            st.html(
                                f"""
                                <div style="
                                    padding:15px 17px;

                                    border-radius:
                                        0 11px 11px 0;

                                    border-left:
                                        3px solid #8B63F7;

                                    background:
                                        rgba(255,255,255,0.035);

                                    color:#C8BECF;

                                    font-size:14px;
                                    line-height:1.75;
                                ">
                                    {safe_review}
                                </div>
                                """
                            )


                        # -------------------------------------
                        # REVIEW DIVIDER
                        # -------------------------------------

                        if position < len(sources) - 1:

                            st.html(
                                """
                                <div style="
                                    height:1px;

                                    background:
                                        rgba(255,255,255,0.07);

                                    margin:25px 0;
                                ">
                                </div>
                                """
                            )


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as error:

            st.error(
                "BIPilot couldn't complete this analysis."
            )

            with st.expander(
                "Technical details",
                expanded=False
            ):

                st.code(
                    str(error)
                )