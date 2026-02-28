import streamlit as st
import pandas as pd
import json
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

    # Normalize column names
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

    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
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

    if "execution_status" in df.columns:
        df["execution_status"] = df["execution_status"].fillna("unknown")

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

    return {
        "sector": sector,
        "total_pipeline": round(total_value, 2),
        "weighted_pipeline": round(weighted_value, 2),
        "deal_count": len(df_sector),
        "missing_probability": int(df_sector["prob_numeric"].isna().sum())
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
    You are a business intelligence advisor to a founder.

    Based on the metrics below, provide a concise executive summary.
    Include revenue outlook, risk areas, and forecast confidence.

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

    # Load & clean data
    deals = clean_deals(load_deals())
    work_orders = clean_work_orders(load_work_orders())

    # Sector extraction
    sector_list = deals["sector"].dropna().unique()
    selected_sector = None

    for sector in sector_list:
        if sector.lower() in query.lower():
            selected_sector = sector

    if not selected_sector:
        selected_sector = sector_list[0]

    # Compute metrics
    metrics = pipeline_by_sector(deals, selected_sector)

    # Generate insight
    insight = generate_insight(metrics)

    # Display Results
    st.subheader("Executive Insight")
    st.write(insight)

    st.subheader("Sector Pipeline Chart")
    chart_data = sector_distribution(deals)
    st.bar_chart(chart_data)

    st.subheader("Tool Trace")
    for step in st.session_state.trace:
        st.write(step)
