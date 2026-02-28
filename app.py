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
# LOAD LOCAL CSV (DEV MODE)
# ----------------------------
@st.cache_data
def load_deals():
    return pd.read_csv("Cleaned_Deals.csv")

@st.cache_data
def load_work_orders():
    return pd.read_csv("Cleaned_Work_Orders.csv")


# ----------------------------
# CLEANING LAYER
# ----------------------------
def clean_deals(df):

    log_trace("Cleaning Deals data")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )

    # Clean deal_value
    df["deal_value"] = (
        df["deal_value"]
        .replace("[\$,]", "", regex=True)
    )
    df["deal_value"] = pd.to_numeric(df["deal_value"], errors="coerce")

    # Convert dates
    df["tentative_close_date"] = pd.to_datetime(
        df["tentative_close_date"], errors="coerce"
    )

    # Normalize probability
    probability_map = {
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2
    }

    df["closure_probability"] = df["closure_probability"].astype(str).str.lower()
    df["prob_numeric"] = df["closure_probability"].map(probability_map)

    return df


def clean_work_orders(df):

    log_trace("Cleaning Work Orders data")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )

    return df


# ----------------------------
# QUARTER FILTERING
# ----------------------------
def filter_current_quarter(df):

    log_trace("Filtering for current quarter")

    today = datetime.today()
    current_quarter = (today.month - 1) // 3 + 1
    current_year = today.year

    df = df[
        (df["tentative_close_date"].dt.year == current_year) &
        (((df["tentative_close_date"].dt.month - 1) // 3 + 1) == current_quarter)
    ]

    return df


# ----------------------------
# METRICS ENGINE
# ----------------------------
def pipeline_by_sector(df, sector):

    log_trace(f"Filtering deals for sector: {sector}")

    df_sector = df[df["sector"] == sector]

    total_value = df_sector["deal_value"].sum()

    weighted_value = (
        df_sector["deal_value"] *
        df_sector["prob_numeric"].fillna(0.3)
    ).sum()

    missing_prob = df_sector["prob_numeric"].isna().sum()

    confidence_score = round(
        100 - (missing_prob / max(len(df_sector), 1)) * 100,
        2
    )

    return {
        "sector": sector,
        "total_pipeline": round(total_value, 2),
        "weighted_pipeline": round(weighted_value, 2),
        "deal_count": int(len(df_sector)),
        "missing_probability_count": int(missing_prob),
        "forecast_confidence_percent": confidence_score
    }


def sector_distribution(df):

    log_trace("Computing sector distribution")

    return (
        df.groupby("sector")["deal_value"]
        .sum()
        .sort_values(ascending=False)
    )


# ----------------------------
# LLM INSIGHT GENERATOR
# ----------------------------
def generate_insight(metrics):

    log_trace("Generating executive insight via LLM")

    prompt = f"""
    You are a business intelligence advisor.

    Use ONLY the provided metrics.
    Do NOT introduce industry commentary.
    Do NOT assume external information.
    Base all reasoning strictly on the numbers below.

    Provide:
    1. Revenue outlook
    2. Risk assessment
    3. Forecast confidence
    4. Concise recommendation

    Metrics:
    {json.dumps(metrics, indent=2)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📊 Monday.com Business Intelligence Agent (CSV Dev Mode)")

if "trace" not in st.session_state:
    st.session_state.trace = []

query = st.text_input("Ask a founder-level question:")

if query:

    st.session_state.trace = []

    deals = clean_deals(load_deals())
    work_orders = clean_work_orders(load_work_orders())

    # Apply quarter filtering
    deals_q = filter_current_quarter(deals)

    # Extract sector from query
    sector_list = deals["sector"].dropna().unique()
    selected_sector = None

    for sector in sector_list:
        if sector.lower() in query.lower():
            selected_sector = sector

    if not selected_sector:
        selected_sector = sector_list[0]

    metrics = pipeline_by_sector(deals_q, selected_sector)
    insight = generate_insight(metrics)

    # ----------------------------
    # DISPLAY
    # ----------------------------
    st.subheader("Executive Insight")
    st.write(insight)

    st.subheader("Sector Pipeline Chart")
    st.caption("Total pipeline value by sector")
    st.bar_chart(sector_distribution(deals_q))

    st.subheader("Tool Trace")
    for step in st.session_state.trace:
        st.write(step)
