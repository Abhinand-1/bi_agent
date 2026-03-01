import streamlit as st
import pandas as pd
import json
from datetime import datetime
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Monday BI Agent", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------
# TRACE LOGGING
# ----------------------------
def log_trace(message):
    if "trace" not in st.session_state:
        st.session_state.trace = []
    st.session_state.trace.append(f"→ {message}")

# ----------------------------
# LOAD LOCAL CSV
# ----------------------------
@st.cache_data
def load_deals():
    return pd.read_csv("Cleaned_Deals.csv")

@st.cache_data
def load_work_orders():
    return pd.read_csv("Cleaned_Work_Orders.csv")

# ----------------------------
# CLEANING
# ----------------------------
def clean_deals(df):
    log_trace("Cleaning Deals data")

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )

    # FIXED LINE
    df["deal_value"] = df["deal_value"].replace(r"[\$,]", "", regex=True)
    df["deal_value"] = pd.to_numeric(df["deal_value"], errors="coerce")

    df["tentative_close_date"] = pd.to_datetime(
        df["tentative_close_date"], errors="coerce"
    )

    probability_map = {"high": 0.8, "medium": 0.5, "low": 0.2}
    df["closure_probability"] = df["closure_probability"].astype(str).str.lower()
    df["prob_numeric"] = df["closure_probability"].map(probability_map)

    df["deal_status"] = df["deal_status"].astype(str).str.lower()

    return df


def clean_work_orders(df):
    log_trace("Cleaning Work Orders data")

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)   # FIXED
    )

    if "execution_status" in df.columns:
        df["execution_status"] = df["execution_status"].astype(str).str.lower()

    return df


# ----------------------------
# INTENT DETECTION
# ----------------------------
def detect_intent(query):
    q = query.lower()

    if "won" in q and "execut" in q:
        return "won_execution_gap"

    if "pipeline" in q or "forecast" in q:
        return "pipeline_analysis"

    return "pipeline_analysis"


# ----------------------------
# QUARTER FILTER
# ----------------------------
def filter_current_quarter(df):
    today = datetime.today()
    current_quarter = (today.month - 1) // 3 + 1
    current_year = today.year

    df = df[
        (df["tentative_close_date"].dt.year == current_year) &
        (((df["tentative_close_date"].dt.month - 1) // 3 + 1) == current_quarter)
    ]

    return df


# ----------------------------
# METRICS
# ----------------------------
def pipeline_by_sector(df, sector):
    log_trace(f"Analyzing pipeline for sector: {sector}")

    df_sector = df[df["sector"] == sector]

    total_value = df_sector["deal_value"].sum()

    weighted_value = (
        df_sector["deal_value"] *
        df_sector["prob_numeric"].fillna(0.3)
    ).sum()

    missing_prob = df_sector["prob_numeric"].isna().sum()

    confidence = round(
        100 - (missing_prob / max(len(df_sector), 1)) * 100, 2
    )

    conversion_ratio = round(
        (weighted_value / max(total_value, 1)) * 100, 2
    )

    return {
        "sector": sector,
        "total_pipeline": round(total_value, 2),
        "weighted_pipeline": round(weighted_value, 2),
        "deal_count": int(len(df_sector)),
        "missing_probability_count": int(missing_prob),
        "forecast_confidence_percent": confidence,
        "weighted_conversion_ratio_percent": conversion_ratio
    }


def won_not_executed(deals, work_orders):
    log_trace("Checking for won deals not executed")

    if "execution_status" not in work_orders.columns:
        return {
            "error": "execution_status column missing in work_orders"
        }

    won_deals = deals[deals["deal_status"] == "won"]

    completed = work_orders[
        work_orders["execution_status"] == "completed"
    ]["deal_name"].unique()

    not_executed = won_deals[
        ~won_deals["deal_name"].isin(completed)
    ]

    return {
        "total_won_deals": int(len(won_deals)),
        "not_executed_count": int(len(not_executed)),
        "not_executed_deals": not_executed["deal_name"].tolist()
    }


# ----------------------------
# LLM GENERATION
# ----------------------------
def generate_insight(metrics, context_type):

    log_trace("Generating executive insight")

    if context_type == "pipeline":
        instruction = """
        Explain revenue outlook, risk level, forecast confidence,
        and interpret weighted conversion ratio.
        """
    else:
        instruction = """
        Explain operational execution gap, revenue realization risk,
        and provide a concise recommendation.
        """

    prompt = f"""
    You are a business intelligence advisor.
    Use ONLY the metrics provided.
    Do NOT introduce external commentary.

    {instruction}

    Metrics:
    {json.dumps(metrics, indent=2)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ----------------------------
# UI
# ----------------------------
st.title("📊 Monday.com Business Intelligence Agent (CSV Dev Mode)")

if "trace" not in st.session_state:
    st.session_state.trace = []

query = st.text_input("Ask a founder-level question:")

if query:

    st.session_state.trace = []

    deals = clean_deals(load_deals())
    work_orders = clean_work_orders(load_work_orders())

    intent = detect_intent(query)

    if intent == "pipeline_analysis":

        deals_q = filter_current_quarter(deals)

        sector_list = deals["sector"].dropna().unique()
        selected_sector = None

        for sector in sector_list:
            if sector.lower() in query.lower():
                selected_sector = sector

        if not selected_sector:
            selected_sector = sector_list[0]

        metrics = pipeline_by_sector(deals_q, selected_sector)
        insight = generate_insight(metrics, "pipeline")

        st.subheader("Executive Insight")
        st.write(insight)

        st.subheader("Sector Pipeline Chart")
        st.bar_chart(
            deals_q.groupby("sector")["deal_value"].sum()
        )

    elif intent == "won_execution_gap":

        metrics = won_not_executed(deals, work_orders)
        insight = generate_insight(metrics, "execution")

        st.subheader("Executive Insight")
        st.write(insight)

    st.subheader("Tool Trace")
    for step in st.session_state.trace:
        st.write(step)
