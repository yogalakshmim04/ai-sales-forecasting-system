# Project paths and constants
from pathlib import Path

# Project root (AI-Sales-Forecasting-System/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EXTERNAL_DIR = BASE_DIR / "data" / "external"

CLEAN_DATA_PATH = PROCESSED_DIR / "cleaned_data.csv"
DAILY_SALES_PATH = PROCESSED_DIR / "daily_sales.csv"
WEEKLY_SALES_PATH = PROCESSED_DIR / "weekly_sales.csv"

# Model and database paths
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "database" / "ecommerce.db"

# ML settings
RANDOM_STATE = 42
TEST_DAYS = 28      # last 28 days = daily test set
TEST_WEEKS = 6      # last 6 weeks = weekly test set