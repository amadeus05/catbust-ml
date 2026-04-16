import logging
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"


def _hurst_rs_window(log_returns: np.ndarray) -> float:
    x = np.asarray(log_returns, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < max(32, HURST_WINDOW // 2):
        return np.nan

    mu = np.mean(x)
    y = np.cumsum(x - mu)
    r = np.max(y) - np.min(y)
    s = np.std(x, ddof=1)

    if s < 1e-12:
        return np.nan

    h = np.log((r / s) + 1e-12) / np.log(n)
    return float(np.clip(h, 0.0, 1.0))


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT,
            timeframe TEXT,
            open_time INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            quote_volume REAL,
            PRIMARY KEY (symbol, timeframe, open_time)
        )
        """
    )
    conn.commit()
    return conn


def fetch_data(conn, symbol, timeframe):
    api_symbol = symbol.replace("/", "")

    cur = conn.cursor()
    cur.execute(
        "SELECT MAX(open_time) FROM candles WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    )
    last_ts = cur.fetchone()[0]

    if last_ts:
        start_ts = last_ts + 1
    else:
        start_ts = int(datetime.fromisoformat(START_DATE).timestamp() * 1000)

    end_ts = int(datetime.fromisoformat(END_DATE).timestamp() * 1000) if END_DATE else None

    if end_ts and start_ts >= end_ts:
        logger.info(f"[{symbol}-{timeframe}] Данные уже загружены до {END_DATE}")
        return 0

    total_loaded = 0

    while True:
        params = {
            "symbol": api_symbol,
            "interval": timeframe,
            "startTime": start_ts,
            "limit": BINANCE_LIMIT,
        }
        if end_ts:
            params["endTime"] = end_ts

        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error(f"Ошибка загрузки {symbol}-{timeframe}: {e}")
            break

        if not data:
            break

        rows = []
        for k in data:
            current_ts = k[0]
            rows.append(
                (
                    symbol,
                    timeframe,
                    current_ts,
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                    float(k[7]),
                )
            )
            start_ts = current_ts + 1

        cur.executemany("INSERT OR IGNORE INTO candles VALUES (?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        total_loaded += len(rows)

        logger.info(
            f"[{symbol}-{timeframe}] Загружено {total_loaded} свечей, "
            f"до {datetime.fromtimestamp((start_ts - 1) / 1000)}"
        )

        if len(data) < BINANCE_LIMIT:
            break

        if end_ts and start_ts >= end_ts:
            logger.info(f"[{symbol}-{timeframe}] Достигнута дата окончания {END_DATE}")
            break

        time.sleep(BINANCE_SLEEP)

    return total_loaded


def load_from_db(conn, symbol, timeframe):
    df = pd.read_sql_query(
        """
        SELECT open_time as timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol=? AND timeframe=?
        ORDER BY open_time
        """,
        conn,
        params=(symbol, timeframe),
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close_safe = df["close"].replace(0, 1e-9)
    ret_1 = df["close"].pct_change(1)

    df["ret_24"] = df["close"].pct_change(24)

    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["vol_24"] = ret_1.rolling(24).std()
    vol_12 = ret_1.rolling(12).std()
    df["vol_ratio_12_24"] = vol_12 / df["vol_24"].replace(0, 1e-9)

    adx_df = df.ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=ADX_LENGTH,
    )
    if adx_df is not None:
        adx_col = f"ADX_{ADX_LENGTH}"
        if adx_col not in adx_df.columns:
            candidates = [c for c in adx_df.columns if c.startswith("ADX_") and not c.startswith("ADXR")]
            adx_col = candidates[0] if candidates else adx_df.columns[0]
        df[f"ADX_{ADX_LENGTH}"] = adx_df[adx_col]
    else:
        df[f"ADX_{ADX_LENGTH}"] = np.nan

    log_ret = np.log(df["close"] / df["close"].shift(1)).replace([np.inf, -np.inf], np.nan)
    df["hurst_rs"] = log_ret.rolling(HURST_WINDOW, min_periods=HURST_WINDOW).apply(
        _hurst_rs_window,
        raw=True,
    )

    atr_14 = df.ta.atr(length=14).replace(0, 1e-9)
    atr_28 = df.ta.atr(length=28).replace(0, 1e-9)
    df["atr_pct"] = atr_14 / close_safe
    df["atr_expansion_14_28"] = atr_14 / atr_28

    roll_high_20 = df["high"].rolling(20).max().shift(1)
    roll_low_20 = df["low"].rolling(20).min().shift(1)
    roll_high_50 = df["high"].rolling(50).max().shift(1)
    roll_low_50 = df["low"].rolling(50).min().shift(1)

    channel_width_20 = (roll_high_20 - roll_low_20) / close_safe
    df["channel_width_20_vs_mean"] = (
        channel_width_20 / channel_width_20.rolling(50).mean().replace(0, 1e-9)
    )
    df["channel_width_50"] = (roll_high_50 - roll_low_50) / close_safe

    resistance = df["high"].rolling(50, min_periods=1).max().shift(1)
    support = df["low"].rolling(50, min_periods=1).min().shift(1)

    df["Dist_to_Resistance"] = (resistance - df["close"]) / atr_14
    df["Dist_to_Support"] = (df["close"] - support) / atr_14

    df.dropna(inplace=True)
    return df


def _calc_long_trade_return_np(entry_price, opens, highs, lows, closes):
    tp_price = entry_price * (1 + TP_PCT)
    sl_price = entry_price * (1 - SL_PCT)

    for k in range(len(opens)):
        bar_open = opens[k]
        bar_high = highs[k]
        bar_low = lows[k]

        hit_sl = bar_low <= sl_price
        hit_tp = bar_high >= tp_price

        if hit_sl and hit_tp:
            exit_price = (bar_open if bar_open < sl_price else sl_price) * (1 - SLIPPAGE)
            raw_ret = (exit_price - entry_price) / entry_price
            return raw_ret - 2 * TAKER_COM

        if hit_sl:
            exit_price = (bar_open if bar_open < sl_price else sl_price) * (1 - SLIPPAGE)
            raw_ret = (exit_price - entry_price) / entry_price
            return raw_ret - 2 * TAKER_COM

        if hit_tp:
            exit_price = tp_price * (1 - SLIPPAGE)
            raw_ret = (exit_price - entry_price) / entry_price
            return raw_ret - 2 * TAKER_COM

    exit_price = closes[-1] * (1 - SLIPPAGE)
    raw_ret = (exit_price - entry_price) / entry_price
    return raw_ret - 2 * TAKER_COM


def _calc_short_trade_return_np(entry_price, opens, highs, lows, closes):
    tp_price = entry_price * (1 - TP_PCT)
    sl_price = entry_price * (1 + SL_PCT)

    for k in range(len(opens)):
        bar_open = opens[k]
        bar_high = highs[k]
        bar_low = lows[k]

        hit_sl = bar_high >= sl_price
        hit_tp = bar_low <= tp_price

        if hit_sl and hit_tp:
            exit_price = (bar_open if bar_open > sl_price else sl_price) * (1 + SLIPPAGE)
            raw_ret = (entry_price - exit_price) / entry_price
            return raw_ret - 2 * TAKER_COM

        if hit_sl:
            exit_price = (bar_open if bar_open > sl_price else sl_price) * (1 + SLIPPAGE)
            raw_ret = (entry_price - exit_price) / entry_price
            return raw_ret - 2 * TAKER_COM

        if hit_tp:
            exit_price = tp_price * (1 + SLIPPAGE)
            raw_ret = (entry_price - exit_price) / entry_price
            return raw_ret - 2 * TAKER_COM

    exit_price = closes[-1] * (1 + SLIPPAGE)
    raw_ret = (entry_price - exit_price) / entry_price
    return raw_ret - 2 * TAKER_COM


def add_trade_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    n = len(df)
    usable = max(0, n - HORIZON - 1)

    long_targets = np.full(n, np.nan, dtype=np.float32)
    short_targets = np.full(n, np.nan, dtype=np.float32)

    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)

    logger.info(f"▶️ Начинаю расчет trade-return таргетов: rows={n}, usable={usable}")

    for i in range(usable):
        entry_bar_idx = i + 1
        horizon_end_idx = entry_bar_idx + HORIZON

        entry_open = opens[entry_bar_idx]

        long_entry_price = entry_open * (1 + SLIPPAGE)
        short_entry_price = entry_open * (1 - SLIPPAGE)

        window_opens = opens[entry_bar_idx:horizon_end_idx]
        window_highs = highs[entry_bar_idx:horizon_end_idx]
        window_lows = lows[entry_bar_idx:horizon_end_idx]
        window_closes = closes[entry_bar_idx:horizon_end_idx]

        long_targets[i] = _calc_long_trade_return_np(
            long_entry_price, window_opens, window_highs, window_lows, window_closes
        )
        short_targets[i] = _calc_short_trade_return_np(
            short_entry_price, window_opens, window_highs, window_lows, window_closes
        )

        if i % 5000 == 0 and i > 0:
            logger.info(f"   progress: {i}/{usable} ({i / usable * 100:.1f}%)")

    df["Target_Long_Return"] = long_targets
    df["Target_Short_Return"] = short_targets

    logger.info(f"✅ Trade-return таргеты посчитаны: usable={usable} / total={n}")
    return df


def save_processed(df: pd.DataFrame, symbol: str):
    conn = sqlite3.connect(DB_PATH)
    table_name = symbol.replace("/", "_") + "_features"
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"💾 {symbol} features сохранены ({len(df)} строк)")


def main():
    conn = init_db()

    for symbol in SYMBOLS:
        logger.info(f"Loading {symbol} {TIMEFRAME} from {START_DATE}...")
        loaded = fetch_data(conn, symbol, TIMEFRAME)
        logger.info(f"{symbol} {TIMEFRAME}: {loaded} new candles")

        df = load_from_db(conn, symbol, TIMEFRAME)
        if len(df) == 0:
            logger.warning(f"{symbol}: no data in DB")
            continue

        logger.info(f"{symbol}: {TIMEFRAME}={len(df)} свечей")

        df = add_features(df)
        df = add_trade_return_targets(df)
        save_processed(df, symbol)

    conn.close()


if __name__ == "__main__":
    main()