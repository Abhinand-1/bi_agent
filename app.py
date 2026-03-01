import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import matplotlib.pyplot as plt
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Monday BI Agent", layout="wide")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MONDAY_API_KEY = st.secrets["MONDAY_API_KEY"]
DEALS_BOARD_ID = st.secrets["DEALS_BOARD_ID"]
WORK_ORDERS_BOARD_ID = st.secrets["WORK_ORDERS_BOARD_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)

MONDAY_URL = "https://api.monday.com/v2"
MONDAY_HEADERS = {
    "Authorization": MONDAY_API_KEY,
    "Content-Type": "application/json"
}

# ----------------------------
# TRACE LOGGING
# ----------------------------
def log_trace(message):
    if "trace" not in st.session_state:
        st.session_state.trace = []
    st.session_state.trace.append(f"→ {message}")

# ----------------------------
# MONDAY API FETCH (LIVE)
# ----------------------------
def fetch_board_data(board_id):

    log_trace(f"Calling Monday API for board {board_id}")

    query = f"""
    {{
      boards(ids: {board_id}) {{
        items_page(limit: 500) {{
          items {{
            name
            column_values {{
              text
              column {{
                title
              }}
            }}
          }}
        }}
      }}
    }}
    """

    try:
        response = requests.post(
            MONDAY_URL,
            json={"query": query},
            headers=MONDAY_HEADERS
        )
        response.raise_for_status()
        data = response.json()

        items = data["data"]["boards"][0]["items_page"]["items"]

        records = []
        for item in items:
            row = {"item_name": item["name"]}
            for col in item["column_values"]:
                row[col["column"]["title"]] = col["text"]
            records.append(row)

        log_trace(f"Fetched {len(records)} records from board {board_id}")

        return pd.DataFrame(records)

    except Exception as e:
        log_trace(f"Monday API error: {str(e)}")
        return pd.DataFrame()

# ----------------------------
# CLEANING
# ----------------------------
def normalize_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df


def clean_deals(df):

    log_trace("Cleaning Deals data")

    df = normalize_columns(df)

    # 🔥 Improved Deal Value Parsing
    if "masked_deal_value" in df.columns:
        df["deal_value"] = (
            df["masked_deal_value"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)  # remove currency + text
        )
        df["deal_value"] = pd.to_numeric(df["deal_value"], errors="coerce")
    else:
        df["deal_value"] = 0

    # Dates
    if "tentative_close_date" in df.columns:
        df["tentative_close_date"] = pd.to_datetime(
            df["tentative_close_date"], errors="coerce"
        )

    # Probability
    probability_map = {"high": 0.8, "medium": 0.5, "low": 0.2}
    if "closure_probability" in df.columns:
        df["closure_probability"] = (
            df["closure_probability"]
            .astype(str)
            .str.lower()
            .str.strip()
        )
        df["prob_numeric"] = df["closure_probability"].map(probability_map)
    else:
        df["prob_numeric"] = None

    # Deal Status
    if "deal_status" in df.columns:
        df["deal_status"] = (
            df["deal_status"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    # Sector normalization
    if "sectorservice" in df.columns:
        df["sectorservice"] = (
            df["sectorservice"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    elif "sector" in df.columns:
        df["sector"] = (
            df["sector"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df
    # Probability
    probability_map = {"high": 0.8, "medium": 0.5, "low": 0.2}
    if "closure_probability" in df.columns:
        df["closure_probability"] = df["closure_probability"].astype(str).str.lower().str.strip()
        df["prob_numeric"] = df["closure_probability"].map(probability_map)
    else:
        df["prob_numeric"] = None

    # Deal Status
    if "deal_status" in df.columns:
        df["deal_status"] = df["deal_status"].astype(str).str.lower().str.strip()

    # Sector normalization
    if "sectorservice" in df.columns:
        df["sectorservice"] = df["sectorservice"].astype(str).str.strip().str.lower()
    elif "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip().str.lower()

    return df


def clean_work_orders(df):

    log_trace("Cleaning Work Orders data")

    df = normalize_columns(df)

    execution_cols = [c for c in df.columns if "execut" in c or "status" in c]

    if execution_cols:
        df["execution_status"] = df[execution_cols[0]].astype(str).str.lower().str.strip()
    else:
        df["execution_status"] = None

    return df


# ----------------------------
# LLM INTENT PARSING
# ----------------------------
def parse_query_with_llm(query):

    log_trace("Parsing user intent via LLM")

    prompt = f"""
    Extract structured intent from this founder question.

    Respond ONLY with valid JSON.
    No explanation. No markdown.

    Format:
    {{
        "intent": "pipeline" or "execution_gap",
        "sector": string or null,
        "timeframe": "quarter" or "month" or "year" or null
    }}

    Question:
    {query}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_content = response.choices[0].message.content.strip()

        # Remove markdown if model adds it
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]

        parsed = json.loads(raw_content)

        log_trace(f"Parsed intent: {parsed}")

        return parsed

    except Exception as e:

        log_trace(f"LLM parsing error: {str(e)}")

        # SAFE FALLBACK
        return {
            "intent": "pipeline",
            "sector": None,
            "timeframe": "quarter"
        }


# ----------------------------
# TIME FILTER
# ----------------------------
def filter_timeframe(df, timeframe):

    if "tentative_close_date" not in df.columns or timeframe is None:
        return df

    log_trace(f"Filtering data for timeframe: {timeframe}")

    today = datetime.today()

    if timeframe == "quarter":
        current_quarter = (today.month - 1) // 3 + 1
        filtered = df[
            (df["tentative_close_date"].dt.year == today.year) &
            (((df["tentative_close_date"].dt.month - 1) // 3 + 1) == current_quarter)
        ]

    elif timeframe == "month":
        filtered = df[
            (df["tentative_close_date"].dt.year == today.year) &
            (df["tentative_close_date"].dt.month == today.month)
        ]

    elif timeframe == "year":
        filtered = df[
            df["tentative_close_date"].dt.year == today.year
        ]

    else:
        filtered = df

    log_trace(f"Records after timeframe filter: {len(filtered)}")

    return filtered


# ----------------------------
# METRICS
# ----------------------------
def pipeline_metrics(df, sector):

    log_trace(f"Computing pipeline metrics for sector: {sector}")

    sector_col = "sectorservice" if "sectorservice" in df.columns else "sector"

    if sector_col not in df.columns:
        return {"error": "Sector column missing"}

    # Filter by sector if provided
sector_col = None
if "sectorservice" in deals_filtered.columns:
    sector_col = "sectorservice"
elif "sector" in deals_filtered.columns:
    sector_col = "sector"

if sector_col:

    chart_data = (
        deals_filtered
        .groupby(sector_col)["deal_value"]
        .sum()
        .reset_index()
    )

    total_chart_value = chart_data["deal_value"].sum()

    if total_chart_value > 0:
        st.subheader("Pipeline by Sector")
        st.bar_chart(chart_data.set_index(sector_col))
    elif len(chart_data) > 0:
        st.info("Deals exist but financial values are zero or missing. Chart suppressed.")
    else:
        st.info("No deals available for visualization in selected timeframe.")

    # ✅ FIX: Proper zero-deal handling
    if deal_count == 0:
        confidence = None
    else:
        confidence = round(
            100 - (missing_prob / deal_count) * 100,
            2
        )

    return {
        "sector": sector,
        "deal_count": int(deal_count),
        "total_pipeline": round(float(total_value), 2),
        "weighted_pipeline": round(float(weighted_value), 2),
        "forecast_confidence_percent": confidence,
        "data_quality": {
            "missing_probability": int(missing_prob),
            "missing_close_dates": int(missing_dates)
        }
    }


def execution_gap_metrics(deals, work_orders):

    log_trace("Analyzing won vs execution gap")

    won_deals = deals[deals["deal_status"] == "won"]

    completed = work_orders[
        work_orders["execution_status"] == "completed"
    ]["item_name"].unique()

    not_executed = won_deals[
        ~won_deals["item_name"].isin(completed)
    ]

    return {
        "total_won_deals": int(len(won_deals)),
        "not_executed_count": int(len(not_executed))
    }


# ----------------------------
# LLM INSIGHT
# ----------------------------
def generate_insight(metrics):

    log_trace("Generating executive insight via LLM")

    prompt = f"""
    You are a business intelligence system.

    Use ONLY the metrics provided.
    Do NOT speculate about market trends.
    Do NOT assume external factors.
    If deal_count is 0, clearly state there is no active pipeline in the selected timeframe.

    Provide:
    1. Executive Summary (strictly data-based)
    2. Risk Assessment (based only on available metrics)
    3. One Operational Recommendation (internal action, not market speculation)

    Metrics:
    {json.dumps(metrics, indent=2)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

# ----------------------------
# UI
# ----------------------------
st.title("Monday.com Business Intelligence Agent")

if "trace" not in st.session_state:
    st.session_state.trace = []

query = st.text_input("Ask a founder-level question:")

if query:

    st.session_state.trace = []

    deals_raw = fetch_board_data(DEALS_BOARD_ID)
    work_orders_raw = fetch_board_data(WORK_ORDERS_BOARD_ID)

    deals = clean_deals(deals_raw)
    work_orders = clean_work_orders(work_orders_raw)

    parsed = parse_query_with_llm(query)

    st.subheader("Agent Reasoning")
    st.json(parsed)

    if parsed["intent"] == "pipeline":

        deals_filtered = filter_timeframe(deals, parsed["timeframe"])

        metrics = pipeline_metrics(deals_filtered, parsed["sector"])

        insight = generate_insight(metrics)

        st.subheader("Executive Insight")
        st.write(insight)

        if "sectorservice" in deals_filtered.columns:
            chart_data = deals_filtered.groupby("sectorservice")["deal_value"].sum()
            st.bar_chart(chart_data)

    else:

        metrics = execution_gap_metrics(deals, work_orders)
        insight = generate_insight(metrics)

        st.subheader("Executive Insight")
        st.write(insight)

        completed = metrics["total_won_deals"] - metrics["not_executed_count"]
        fig, ax = plt.subplots()
        ax.pie(
            [completed, metrics["not_executed_count"]],
            labels=["Completed", "Not Executed"],
            autopct="%1.1f%%"
        )
        st.pyplot(fig)

    st.subheader("Tool Trace")
    for step in st.session_state.trace:
        st.write(step)
