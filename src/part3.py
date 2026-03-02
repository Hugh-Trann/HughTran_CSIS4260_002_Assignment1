import pandas as pd
import streamlit as st


OUT_DIR = "outputs"
PRED_PATH = f"{OUT_DIR}/predictions.csv"
OVERALL_PATH = f"{OUT_DIR}/overall_metrics.csv"
BY_TICKER_PATH = f"{OUT_DIR}/metrics_by_ticker.csv"

DATE_COL = "date"
TICKER_COL = "name"
CLOSE_COL = "close"


st.set_page_config(page_title="HughTran", layout="wide")

st.title("STOCK PREDICTION")

# Load data
@st.cache_data
def load_data():
    preds = pd.read_csv(PRED_PATH)
    overall = pd.read_csv(OVERALL_PATH)
    by_ticker = pd.read_csv(BY_TICKER_PATH)
    # convert date column to datetime format
    preds[DATE_COL] = pd.to_datetime(preds[DATE_COL])
    return preds, overall, by_ticker

preds, overall, by_ticker = load_data()

# --- Overall of model trainings ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Overall Performance by Models")
    st.dataframe(overall, use_container_width=True)

with col2:
    st.markdown("### Metrics by Ticker (Top rows by RF_MAE)") # 
    st.dataframe(by_ticker.head(20), use_container_width=True)

st.divider()

# --- Ticker charts ---

# Search ticker bar
st.sidebar.header("SEARCH")

all_tickers = sorted(preds[TICKER_COL].unique().tolist())
search = st.sidebar.text_input("Search ticker", "")

filtered = [t for t in all_tickers if search.lower() in str(t).lower()]
if not filtered:
    st.sidebar.warning("No match, try again.")
    st.stop()

# Filter to one ticker
selected = st.sidebar.selectbox("Select ticker", filtered)
df_t = preds[preds[TICKER_COL] == selected].sort_values(DATE_COL).copy()

# display the selected ticker
st.subheader(f"The selected ticker: {selected}")

# Prediction chart
st.markdown("## Price Predictions")

chart_df = df_t[[DATE_COL, "Actual_NextDayClose", "Pred_LR", "Pred_RF"]].set_index(DATE_COL)
st.line_chart(chart_df, use_container_width=True)

st.divider()

# Indicator charts
st.markdown("## Technical Indicators")

st.markdown("### EMA")
ema_df = df_t[[DATE_COL, CLOSE_COL, "EMA_12", "EMA_26"]].set_index(DATE_COL)
st.line_chart(ema_df, use_container_width=True)

st.markdown("### MACD")
macd_df = df_t[[DATE_COL, "MACD", "MACD_Signal", "MACD_Hist"]].set_index(DATE_COL)
st.line_chart(macd_df, use_container_width=True)

st.markdown("### RSI")
rsi_df = df_t[[DATE_COL, "RSI_14"]].set_index(DATE_COL)
st.line_chart(rsi_df, use_container_width=True)

st.divider()

st.markdown("## Data Preview (with Indicators)")
st.dataframe(df_t.head(50), use_container_width=True)