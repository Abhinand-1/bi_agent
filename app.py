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
# MONDAY API FETCH
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

    response = requests.post(
        MONDAY_URL,
        json={"query": query},
        headers=MONDAY_HEADERS
    )

    data = response.json()
    items = data["data"]["boards"][0]["items_page"]["items"]

    records = []
    for item in items:
        row = {"item_name": item["name"]}
        for col in item["column_values"]:
            row[col["column"]["title"]] = col["text"]
        records.append(row)

    return pd.DataFrame(records)

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

    if "masked_deal_value" in df.columns:
        df["deal_value"] = df["masked_deal_value"].replace(r"[\$,]", "", regex=True)
        df["deal_value"] = pd.to_numeric(df["deal_value"], errors="coerce")
    else:
        df["deal_value"] = 0

    if "tentative_close_date" in df.columns:
        df["tentative_close_date"] = pd.to_datetime(
            df["tentative_close_date"], errors="coerce"
        )

    probability_map = {"high": 0.8, "medium": 0.5, "low": 0.2}

    if "closure_probability" in df.columns:
        df["closure_probability"] = df["closure_probability"].astype(str).str.lower()
        df["prob_numeric"] = df["closure_probability"].map(probability_map)
    else:
        df["prob_numeric"] = None

    if "deal_status" in df.columns:
        df["deal_status"] = df["deal_status"].astype(str).str.lower()

    return df


def clean_work_orders(df):

    log_trace("Cleaning Work Orders data")

    df = normalize_columns(df)

    execution_cols = [c for c in df.columns if "execut" in c or "status" in c]

    if execution_cols:
        df["execution_status"] = df[execution_cols[0]].astype(str).str.lower()
    else:
        df["execution_status"] = None

    return df

# ----------------------------
# INTENT DETECTION
# ----------------------------
def parse_query_with_llm(query):

    log_trace("Parsing user intent via LLM")

    prompt = f"""
    Extract structured intent from this founder question.

    Return JSON with:
    - intent (pipeline or execution_gap)
    - sector (if mentioned)
    - timeframe (quarter, month, year, none)

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content
    return json.loads(content)

# ----------------------------
# QUARTER FILTER
# ----------------------------
def filter_current_quarter(df):

    if "tentative_close_date" not in df.columns:
        return df

    log_trace("Filtering for current quarter")

    today = datetime.today()
    current_quarter = (today.month - 1) // 3 + 1
    current_year = today.year

    return df[
        (df["tentative_close_date"].dt.year == current_year) &
        (((df["tentative_close_date"].dt.month - 1) // 3 + 1) == current_quarter)
    ]

# ----------------------------
# METRICS
# ----------------------------
def pipeline_metrics(df, sector):

    log_trace(f"Computing pipeline metrics for sector: {sector}")

    if "sectorservice" in df.columns:
        sector_col = "sectorservice"
    elif "sector" in df.columns:
        sector_col = "sector"
    else:
        return {}

    df_sector = df[df[sector_col] == sector]

    total_value = df_sector["deal_value"].sum()
    weighted_value = (
        df_sector["deal_value"] *
        df_sector["prob_numeric"].fillna(0.3)
    ).sum()

    missing_prob = df_sector["prob_numeric"].isna().sum()

    confidence = round(
        100 - (missing_prob / max(len(df_sector), 1)) * 100,
        2
    )

    return {
        "sector": sector,
        "total_pipeline": round(float(total_value), 2),
        "weighted_pipeline": round(float(weighted_value), 2),
        "deal_count": int(len(df_sector)),
        "forecast_confidence_percent": confidence
    }


def execution_gap_metrics(deals, work_orders):

    log_trace("Analyzing won vs execution gap")

    if "deal_status" not in deals.columns:
        return {"error": "deal_status column missing"}

    won_deals = deals[deals["deal_status"] == "won"]

    if "execution_status" not in work_orders.columns or work_orders["execution_status"].isnull().all():
        return {
            "total_won_deals": int(len(won_deals)),
            "not_executed_count": "Unknown"
        }

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
# LLM
# ----------------------------
def generate_insight(metrics, mode):

    log_trace("Generating executive insight via LLM")

    if mode == "pipeline":
        instruction = """
        Explain revenue outlook, risk level, and forecast confidence.
        Base reasoning strictly on the metrics.
        """
    else:
        instruction = """
        Explain operational execution gap and revenue realization risk.
        Provide a concise recommendation.
        """

    prompt = f"""
    You are a business intelligence advisor.
    Use ONLY the metrics below.
    Do NOT assume external information.

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
st.title("Monday.com Business Intelligence Agent ")

if "trace" not in st.session_state:
    st.session_state.trace = []

query = st.text_input("Ask a founder-level question:")

if query:

    st.session_state.trace = []

    deals_raw = fetch_board_data(DEALS_BOARD_ID)
    work_orders_raw = fetch_board_data(WORK_ORDERS_BOARD_ID)

    deals = clean_deals(deals_raw)
    work_orders = clean_work_orders(work_orders_raw)

    intent = detect_intent(query)

    if intent == "pipeline":

        deals_q = filter_current_quarter(deals)

        if "sectorservice" in deals_q.columns:
            sector_col = "sectorservice"
        elif "sector" in deals_q.columns:
            sector_col = "sector"
        else:
            sector_col = None

        if sector_col:
            sector_list = deals_q[sector_col].dropna().unique()
        else:
            sector_list = []

        selected_sector = None
        for sector in sector_list:
            if sector.lower() in query.lower():
                selected_sector = sector

        if not selected_sector and len(sector_list) > 0:
            selected_sector = sector_list[0]

        metrics = pipeline_metrics(deals_q, selected_sector)
        insight = generate_insight(metrics, "pipeline")

        st.subheader("Executive Insight")
        st.write(insight)

        # ---- Chart ----
        if sector_col:
            st.subheader("Pipeline by Sector (Current Quarter)")
            chart_data = deals_q.groupby(sector_col)["deal_value"].sum()
            st.bar_chart(chart_data)

    else:

        metrics = execution_gap_metrics(deals, work_orders)
        insight = generate_insight(metrics, "execution")

        st.subheader("Executive Insight")
        st.write(insight)

        # ---- Pie Chart ----
        if isinstance(metrics.get("not_executed_count"), int):
            completed = metrics["total_won_deals"] - metrics["not_executed_count"]
            not_exec = metrics["not_executed_count"]

            fig, ax = plt.subplots()
            ax.pie(
                [completed, not_exec],
                labels=["Completed", "Not Executed"],
                autopct="%1.1f%%"
            )
            st.subheader("Execution Status Overview")
            st.pyplot(fig)

    st.subheader("Tool Trace")
    for step in st.session_state.trace:
        st.write(step)
