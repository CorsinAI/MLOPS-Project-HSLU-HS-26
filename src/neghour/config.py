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

# Open-Meteo
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORY_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Energy-Charts (fallback)
ENERGY_CHARTS_BASE_URL = "https://api.energy-charts.info"

# Hopsworks
HOPSWORKS_PROJECT: str = os.getenv("HOPSWORKS_PROJECT", "neghour")
HOPSWORKS_API_KEY: str = os.getenv("HOPSWORKS_API_KEY", "")

FEATURE_GROUP_NAME = "neghour_features"
FEATURE_GROUP_VERSION = 1
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

# TODO: add Grid coordinates for Open-Meteo (centre of DE grid)
DE_GRID_LAT = 51.2
DE_GRID_LON = 10.5
