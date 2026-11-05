"""
Streamlit dashboard — NegativeHours.

Deployed on Hugging Face Spaces (free tier).
Reads predictions from the Hopsworks predictions feature group and outcomes from
the labels feature group, joined on delivery_utc.

Two views:
  TOMORROW FORECAST  (available once the forecast for D is written)
    Hourly bar chart: P(negative) per delivery hour, colour-coded by risk level.
    Green < 20% · Yellow 20–50% · Red > 50%

  PAST ACCURACY  (available from 13:00 Berlin on D-1, when prices count as published)
    Forecast vs actual for the last 30 days.
    PR-AUC and F1 trend lines.
"""

import streamlit as st

# TODO: implement load_predictions(days: int = 30) -> pd.DataFrame
#       reads from Hopsworks predictions feature group
# TODO: implement render_tomorrow_forecast(df: pd.DataFrame) -> None
#       Altair / Plotly bar chart with colour encoding
# TODO: implement render_accuracy_history(df: pd.DataFrame) -> None
#       line chart: PR-AUC and F1 per day; confusion breakdown

st.set_page_config(page_title="NegativeHours", page_icon="⚡", layout="wide")
st.title("NegativeHours — DE-LU Negative Price Forecast")
st.caption("Predicting which hours of tomorrow will have a negative day-ahead price.")
st.info("Dashboard under construction. Come back soon.")
