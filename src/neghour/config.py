"""
Central configuration for NegativeHours.

Reads settings from environment variables (see .env.example).
Defines the 10:00 Berlin cut-off that separates features from labels.
"""

from __future__ import annotations

import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

BERLIN_TZ = ZoneInfo("Europe/Berlin")
CUTOFF_HOUR: int = int(os.getenv("CUTOFF_HOUR", "10"))  # 10:00 Berlin on D-1

# SMARD (Bundesnetzagentur)
SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
SMARD_PRICE_FILTER = 4169  # DE-LU day-ahead price
SMARD_REGION = "DE-LU"
SMARD_SOLAR_FILTER = 4068
SMARD_WIND_ONSHORE_FILTER = 4067
SMARD_WIND_OFFSHORE_FILTER = 1225
SMARD_LOAD_FILTER = 410

# Open-Meteo
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Fixed grid points (latitude, longitude). Each variable averages the points
# that matter for it. See README, "Spatial coverage".
WEATHER_POINTS: dict[str, tuple[float, float]] = {
    "n_coast": (54.4, 8.7),
    "ne": (53.0, 13.0),
    "centre": (51.2, 10.5),
    "south": (48.6, 11.0),
}
WIND_POINTS = ("n_coast", "ne", "centre")
SOLAR_POINTS = ("ne", "centre", "south")
TEMPERATURE_POINTS = ("n_coast", "ne", "centre", "south")

# Residual-load proxy: approximate installed capacity at the end of 2024, and
# the wind speed at which a turbine reaches full output. Fixed on purpose:
# nothing is fitted, so nothing can leak from later periods.
WIND_CAPACITY_MW = 73_000
SOLAR_CAPACITY_MW = 100_000
WIND_RATED_SPEED_MS = 12.0
FULL_SUN_W_M2 = 1000.0

# Population by state, 2025, based on census 2022. Source: Statistische Aemter
# des Bundes und der Laender, statistikportal.de, "Flaeche und Bevoelkerung".
STATE_POPULATION: dict[str, int] = {
    "BW": 11_238_761,
    "BY": 13_245_503,
    "BE": 3_700_577,
    "BB": 2_551_889,
    "HB": 706_915,
    "HH": 1_869_473,
    "HE": 6_278_594,
    "MV": 1_573_685,
    "NI": 8_000_826,
    "NW": 17_986_214,
    "RP": 4_123_022,
    "SL": 1_007_342,
    "SN": 4_026_599,
    "ST": 2_120_252,
    "SH": 2_958_519,
    "TH": 2_078_946,
}

# Energy-Charts (fallback)
ENERGY_CHARTS_BASE_URL = "https://api.energy-charts.info"

# Storage
STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "hopsworks")

# Hopsworks
HOPSWORKS_HOST: str = os.getenv("HOPSWORKS_HOST", "")
HOPSWORKS_PROJECT: str = os.getenv("HOPSWORKS_PROJECT", "neghour")
HOPSWORKS_API_KEY: str = os.getenv("HOPSWORKS_API_KEY", "")

FEATURE_GROUP_NAME = "neghour_features"
FEATURE_GROUP_VERSION = 1
LABEL_GROUP_NAME = "neghour_labels"
LABEL_GROUP_VERSION = 1
PREDICTION_GROUP_NAME = "neghour_predictions"
PREDICTION_GROUP_VERSION = 1

# MLflow / DagsHub
DAGSHUB_REPO: str = os.getenv("DAGSHUB_REPO", "CorsinAI/MLOPS-Project-HSLU-HS-26")
DAGSHUB_TOKEN: str = os.getenv("DAGSHUB_TOKEN", "")

# Model registry
MODEL_NAME = "neghour_lgbm"

# Success criteria (proposal §1)
TARGET_PR_AUC = 0.75
TARGET_F1_MARGIN = 0.10  # LightGBM F1 must exceed best baseline by at least this
