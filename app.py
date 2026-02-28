import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Monday BI Agent", layout="wide")

# Load OpenAI key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ----------------------------
# TRACE LOGGING
# ----------------------------
def log_trace(message):
    st.session_state.trace.append(f"→ {message}")


# ----------------------------
# LOAD LOCAL CSV (DEV MODE)
# ----------------------------
@st.cache_data
def load_deals():
    log_trace("Loading Deals CSV")
    return pd.read_csv("Cleaned_Deals.csv")


@st.cache_data
def load_work_orders():
    log_trace("Loading Work Orders CSV")
    return pd.read_csv("Cleaned_Work_Orders.csv")


# ----------------------------
# CLEANING LAYER
# ----------------------------
def clean_deals(df):
    log_trace("Cleaning Deals data")

    df["Masked Deal value"] = (
        df["Masked Deal value"]
        .replace("[\$,]", "", regex=True)
    )
    df["Masked Deal value"] = pd.to_numeric(
        df["Masked Deal value"], errors="coerce"
    )

    df["Tentative Close Date"] = pd.to_datetime(
        df["Tentative Close Date"], errors="coerce"
    )

    probability_map = {
        "High": 0.8,
        "Medium": 0.5,
        "Low": 0.2
    }

    df["prob_numeric"] = df["Closure Probability"].map(probability_map)

    return df


def clean_work_orders(df):
    log_trace("Cleaning Work Orders data")

    df["Data Delivery Date"] = pd.to_datetime(
        df["Data Delivery Date"], errors="coerce"
    )

    df["Execution Status"] = df["Execution Status"].fillna("Unknown")

    return df


# ----------------------------
# METRICS ENGINE
# ----------------------------
def pipeline_by_sector(df, sector):
    log_trace(f"Filtering deals for sector: {sector}")

    df_sector = df[df["Sector/service"] == sector]

    total_value = df_sector["Masked Deal value"].sum()

    weighted_value = (
        df_sector["Masked Deal value"] *
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
        df.groupby("Sector/service")["Masked Deal value"]
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

    # Simple sector extraction (basic version)
    sector_list = deals["Sector/service"].dropna().unique()
    selected_sector = None

    for sector in sector_list:
        if sector.lower() in query.lower():
            selected_sector = sector

    if not selected_sector:
        selected_sector = sector_list[0]

    # Compute metrics
    metrics = pipeline_by_sector(deals, selected_sector)

    # Generate LLM response
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
