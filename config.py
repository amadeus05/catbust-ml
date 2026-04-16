from pathlib import Path

# --- ОСНОВНЫЕ ---
DB_PATH = "./data/market_data.db"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "XLM/USDT",
    "ADA/USDT",
    "TRX/USDT",
    "XMR/USDT",
]

TIMEFRAME = "1h"

# --- DATA LOADING ---
START_DATE = "2023-01-01"
END_DATE = "2026-04-15"
BINANCE_LIMIT = 1500
BINANCE_SLEEP = 0.3

# --- FEATURE ENGINEERING ---
ADX_LENGTH = 14
HURST_WINDOW = 100

# Мультисимвольный baseline: ядро + несколько условно полезных фич
FEATURE_COLUMNS = (
    "EMA_50",
    "EMA_200",
    "Dist_to_Resistance",
    "vol_ratio_12_24",
    "vol_24",
    "channel_width_50",
    "channel_width_20_vs_mean",
    "atr_expansion_14_28",
    "Dist_to_Support",
    "ADX_14",
    "ret_24",
    "hurst_rs",
    "atr_pct",
)

FEATURE_SET_VERSION = 101

# --- TARGET / LABEL SIMULATION ---
HORIZON = 12
TP_PCT = 0.030
SL_PCT = 0.015

# Издержки, встроенные в таргет
TAKER_COM = 0.0004
SLIPPAGE = 0.0003

# --- BACKTEST / SIGNALS ---
CONFIDENCE_THRESHOLD = 0.0
TRADE_MODE = "both"   # "short_only" | "long_only" | "both"

# Калибровка порогов на validation-фолде
CALIBRATION_CANDIDATE_TOP_FRACS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20)
CALIBRATION_MIN_COUNT = 40
CALIBRATION_MIN_MEAN_TRUE = 0.0
CALIBRATION_DEFAULT_LONG_THRESHOLD = float("inf")
CALIBRATION_DEFAULT_SHORT_THRESHOLD = float("inf")

# --- REGIME FILTER ---
# Для baseline выключен, но оставлен как переключатель
ENABLE_REGIME_FILTER = False
REGIME_MIN_HURST = 0.52
REGIME_MIN_ADX = 18.0
REGIME_MIN_CHANNEL_WIDTH_RATIO = 0.80
REGIME_MAX_CHANNEL_WIDTH_RATIO = 2.75

# --- WALK-FORWARD ---
WF_TRAIN_DAYS = 365
WF_TEST_DAYS = 30
WF_STEP_DAYS = 30
WF_MAX_FOLDS = 50
WF_MIN_TRAIN_ROWS = 1500
WF_VERBOSE_TRADES = True

# --- PATHS ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)