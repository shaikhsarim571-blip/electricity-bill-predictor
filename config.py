import os

NEPRA_TARIFF_SLABS = {
    "0-100": {"rate": 18.50, "category": "Protected (Low-Income)"},
    "101-200": {"rate": 20.50, "category": "Normal"},
    "201-300": {"rate": 24.75, "category": "High"},
    "301-500": {"rate": 28.90, "category": "Very High"},
    "500+": {"rate": 32.50, "category": "Industrial"}
}

PROTECTED_STATUS_THRESHOLD = 100
PEAK_HOURS = (9, 11, 14, 17, 20)
OFF_PEAK_HOURS = (0, 6, 12, 23)

CONSUMER_SEGMENTS = {
    "Low-Income/Protected": {"consumption_range": (0, 100), "behavior": "minimal"},
    "Middle-Class/Slab-Breacher": {"consumption_range": (101, 300), "behavior": "moderate"},
    "Heavy Commercial/AC-Heavy": {"consumption_range": (301, 500), "behavior": "heavy"},
    "Industrial/Extreme": {"consumption_range": (500, 10000), "behavior": "industrial"}
}

OPTIMAL_CLUSTERS = 4

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1
}

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0
}

OUTLIER_IQR_MULTIPLIER = 1.5
NORMALIZATION_METHOD = "minmax"
MISSING_DATA_STRATEGY = "seasonal_median"
TEST_SIZE = 0.2
FORECAST_HORIZON = 3

THEME_PRIMARY = "#0e1117"
THEME_ACCENT_BLUE = "#00D9FF"
THEME_ACCENT_GREEN = "#39FF14"
THEME_SUCCESS = "#00FF00"
THEME_WARNING = "#FFD700"
THEME_ERROR = "#FF4444"

COLOR_PALETTE = {
    "cluster_0": "#00D9FF",
    "cluster_1": "#39FF14",
    "cluster_2": "#FFD700",
    "cluster_3": "#FF4444",
    "forecast": "#00D9FF",
    "actual": "#39FF14"
}

LLM_PROVIDER = "groq"
GROQ_MODEL = "mixtral-8x7b-32768"
TEMPERATURE = 0.7
MAX_TOKENS = 500

TEMPORAL_FEATURES = [
    "month", "quarter", "season", "is_peak_hour", 
    "day_of_week", "is_weekend"
]

WEATHER_FEATURES = [
    "temperature", "humidity", "wind_speed", "rainfall"
]

DATA_PATH = "data/"
MODEL_PATH = "models/"
LOG_LEVEL = "INFO"
