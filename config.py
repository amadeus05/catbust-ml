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
HTF_TIMEFRAME = "4h"

# Слабые / отключённые для обучения (etl может всё равно писать колонки в БД).
WEAK_FEATURES_DISABLED = (
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_6",
    "ret_12",
    "Trend",
    "vol_6",
    "HTF_Trend",
    "vol_zscore_20",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_pos_in_bar",
    "range_to_atr",
    "range_pct",
    "body_pct",
    "range_expand_20",
    "range_x_relvol",
    "vol_ratio_6_24",
    "atr_ratio_7_28",
    "SMA_10",
    "SMA_20",
    "SMA_200",
    "dist_to_high_20",
    "dist_to_low_20",
    "dist_to_high_50",
    "channel_pos_20",
    "channel_width_20",
    "vol_12",
    "atr_ratio_14_28",
    "beta_btc_60",
    "corr_btc_60",
    "beta_btc_120",
    "corr_btc_120",
    "mkt_corr_btc_eth_60",
    "dist_to_low_50",
    "bb_zscore_20",
    "rel_volume_20",
)

# --- Глобальный контекст (ETL умеет, но в текущей основной версии выключено) ---
BTC_ANCHOR_SYMBOL = "BTC/USDT"
ETH_ANCHOR_SYMBOL = "ETH/USDT"
BETA_BTC_SHORT = 60
BETA_BTC_LONG = 120
MKT_CORR_BTC_ETH_WINDOW = 60
ENABLE_GLOBAL_CONTEXT_ETL = False

# =============================================================================
# Наборы фич — только этот список в train / bt (профили core / core_plus совпадают)
# =============================================================================
FEATURE_COLUMNS_ACTIVE = (
    "EMA_50",
    "vol_24",
    "hurst_rs",
    "EMA_200",
    "Dist_to_Resistance",
    "channel_width_50",
    "channel_width_20_vs_mean",
    "ret_24",
    "vol_ratio_12_24",
    "ADX_14",
    "atr_pct",
    "Dist_to_Support",
    "atr_expansion_14_28",
)

FEATURE_COLUMNS_CORE = FEATURE_COLUMNS_ACTIVE
FEATURE_COLUMNS_CORE_PLUS = FEATURE_COLUMNS_ACTIVE

FEATURE_COLUMNS_CORE_V3 = FEATURE_COLUMNS_ACTIVE
FEATURE_COLUMNS_CORE_PLUS_V3 = FEATURE_COLUMNS_ACTIVE

# Активный профиль
FEATURE_PROFILE = "core_plus"

_FEATURE_PROFILE_MAP = {
    "core": FEATURE_COLUMNS_CORE,
    "core_plus": FEATURE_COLUMNS_CORE_PLUS,
    "core_v3": FEATURE_COLUMNS_CORE_V3,
    "core_plus_v3": FEATURE_COLUMNS_CORE_PLUS_V3,
}
FEATURE_COLUMNS = _FEATURE_PROFILE_MAP.get(FEATURE_PROFILE, FEATURE_COLUMNS_CORE_PLUS)
FEATURE_SET_VERSION = 2

# --- DATA LOADING ---
START_DATE = "2023-01-01"
END_DATE = None
BINANCE_LIMIT = 1500
BINANCE_SLEEP = 0.3

ENABLE_PROD_TRAINING = False

# --- Режим рынка (etl: ADX, Hurst) ---
ADX_LENGTH = 14
HURST_WINDOW = 100

# --- ML LABELING ---
HORIZON = 12
TP_PCT = 0.030
SL_PCT = 0.015
MFE_THRESHOLD = 0.003

# --- TRADING ---
CONFIDENCE_THRESHOLD = 0.01

# Режим торговли в bt.py: "short_only" | "long_only" | "both"
TRADE_MODE = "short_only"

# Калибровка порогов по каждому fold на внутреннем validation
CALIBRATION_CANDIDATE_TOP_FRACS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20)
CALIBRATION_MIN_COUNT = 40
CALIBRATION_MIN_MEAN_TRUE = 0.0
CALIBRATION_DEFAULT_LONG_THRESHOLD = float("inf")
CALIBRATION_DEFAULT_SHORT_THRESHOLD = float("inf")

# --- Regime filter в bt.py ---
ENABLE_REGIME_FILTER = False  # Отключено: анализ показал, что фильтры ухудшают результаты
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
