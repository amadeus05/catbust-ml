"""
scan.py — Сканер всех монет из config.py
Качает 1500 свечей, формирует фичи (etl.py), прогоняет модели, выводит анализ.
Можно указать --datetime "2025-12-01 12:00" для анализа на исторический момент.
"""

import argparse
import sys
import pickle
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from catboost import CatBoostClassifier, CatBoostRegressor
from config import *
from etl import add_features, add_htf_features, TF_MS

# ─── BINANCE API ─────────────────────────────────────────────────────────────

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"


def fetch_candles(symbol: str, timeframe: str, limit: int = 1500,
                  end_ms: int | None = None) -> pd.DataFrame:
    """
    Скачать последние `limit` свечей напрямую с Binance Futures API.
    Если end_ms задан — берём свечи, заканчивающиеся до этого момента.
    """
    api_symbol = symbol.replace("/", "")
    params = {
        "symbol": api_symbol,
        "interval": timeframe,
        "limit": limit,
    }
    if end_ms is not None:
        params["endTime"] = end_ms

    r = requests.get(BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if not data:
        return pd.DataFrame()

    rows = []
    for k in data:
        rows.append({
            "timestamp": pd.Timestamp(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })

    df = pd.DataFrame(rows)
    return df


# ─── МОДЕЛИ ──────────────────────────────────────────────────────────────────

def load_models():
    """Загрузить сохранённые CatBoost‑модели и список фич."""
    clf = CatBoostClassifier()
    clf.load_model(str(MODELS_DIR / "catboost_model.cbm"))

    reg_long = CatBoostRegressor()
    reg_long.load_model(str(MODELS_DIR / "mfe_long_model.cbm"))

    reg_short = CatBoostRegressor()
    reg_short.load_model(str(MODELS_DIR / "mfe_short_model.cbm"))

    with open(MODELS_DIR / "features.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return clf, reg_long, reg_short, feature_names


# ─── АНАЛИЗ ОДНОЙ МОНЕТЫ ─────────────────────────────────────────────────────

def analyse_symbol(symbol: str, clf, reg_long, reg_short, feature_names,
                   end_ms: int | None = None) -> dict | None:
    """
    Полный цикл для одной монеты:
      1) качаем свечи (1h + 4h)
      2) строим фичи
      3) предсказываем направление + MFE
    Возвращает dict с результатами или None при ошибке.
    """
    try:
        # Качаем данные
        df = fetch_candles(symbol, TIMEFRAME, limit=1500, end_ms=end_ms)
        if df.empty or len(df) < 250:
            return {"symbol": symbol, "error": f"Мало свечей {TIMEFRAME}: {len(df)}"}

        htf_df = fetch_candles(symbol, HTF_TIMEFRAME, limit=1500, end_ms=end_ms)
        if htf_df.empty or len(htf_df) < 60:
            return {"symbol": symbol, "error": f"Мало свечей {HTF_TIMEFRAME}: {len(htf_df)}"}

        time.sleep(0.25)  # Пауза между запросами

        # Формируем фичи
        df = add_features(df)
        df = add_htf_features(df, htf_df)

        if df.empty:
            return {"symbol": symbol, "error": "DataFrame пуст после add_features"}

        # Берём последнюю строку для предсказания
        last = df.iloc[-1]
        last_ts = last["timestamp"]
        last_close = last["close"]

        # Проверяем наличие всех нужных фич
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            return {"symbol": symbol, "error": f"Нет фич: {missing[:5]}"}

        feat_row = last[feature_names]
        if feat_row.isna().any():
            nan_cols = [c for c in feature_names if pd.isna(last[c])]
            return {"symbol": symbol, "error": f"NaN в фичах: {nan_cols[:5]}"}

        X = feat_row.to_frame().T

        # Предсказание
        probs = clf.predict_proba(X)[0]
        # class map: 0=Short, 1=Neutral, 2=Long
        p_short = float(probs[0])
        p_neutral = float(probs[1])
        p_long = float(probs[2])

        mfe_long_val = max(0.0, min(float(reg_long.predict(X)[0]), 1.0))
        mfe_short_val = max(0.0, min(float(reg_short.predict(X)[0]), 1.0))

        # Определяем направление
        if p_long > CONFIDENCE_THRESHOLD:
            direction = "LONG"
            signal_prob = p_long
            mfe_relevant = mfe_long_val
        elif p_short > CONFIDENCE_THRESHOLD:
            direction = "SHORT"
            signal_prob = p_short
            mfe_relevant = mfe_short_val
        else:
            direction = "NEUTRAL"
            signal_prob = p_neutral
            mfe_relevant = max(mfe_long_val, mfe_short_val)

        # MFE фильтр
        mfe_pass = mfe_relevant >= MFE_THRESHOLD if direction != "NEUTRAL" else False

        trend = last.get("Trend", None)
        htf_trend = last.get("HTF_Trend", None)

        return {
            "symbol": symbol,
            "error": None,
            "timestamp": last_ts,
            "close": last_close,
            "direction": direction,
            "p_long": p_long,
            "p_short": p_short,
            "p_neutral": p_neutral,
            "signal_prob": signal_prob,
            "mfe_long": mfe_long_val,
            "mfe_short": mfe_short_val,
            "mfe_relevant": mfe_relevant,
            "mfe_pass": mfe_pass,
            "ema_200": last.get("EMA_200"),
            "trend": trend,
            "htf_trend": htf_trend,
            "dist_to_resistance": last.get("Dist_to_Resistance"),
            "dist_to_support": last.get("Dist_to_Support"),
            "candles_count": len(df),
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# ─── КРАСИВЫЙ ВЫВОД ──────────────────────────────────────────────────────────

# Цвета ANSI
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    BG_GREEN  = "\033[42m"
    BG_RED    = "\033[41m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"


def direction_badge(direction: str) -> str:
    if direction == "LONG":
        return f"{C.BG_GREEN}{C.WHITE}{C.BOLD}  🟢 LONG   {C.RESET}"
    elif direction == "SHORT":
        return f"{C.BG_RED}{C.WHITE}{C.BOLD}  🔴 SHORT  {C.RESET}"
    else:
        return f"{C.BG_YELLOW}{C.WHITE}{C.BOLD}  ⚪ NEUTRAL {C.RESET}"


def mfe_bar(value: float, threshold: float, width: int = 20) -> str:
    """Прогресс‑бар для MFE."""
    filled = int(min(value / max(threshold * 2, 0.01), 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    color = C.GREEN if value >= threshold else C.RED
    return f"{color}{bar}{C.RESET} {value*100:.2f}%"


def prob_bar(value: float, width: int = 15) -> str:
    """Прогресс‑бар для вероятностей."""
    filled = int(value * width)
    bar = "▓" * filled + "░" * (width - filled)
    if value >= CONFIDENCE_THRESHOLD:
        color = C.GREEN
    elif value >= 0.5:
        color = C.YELLOW
    else:
        color = C.DIM
    return f"{color}{bar}{C.RESET} {value*100:.1f}%"


def trend_icon(val) -> str:
    if val is None or pd.isna(val):
        return "❓"
    return "📈 Бычий" if val == 1 else "📉 Медвежий"


def print_separator():
    print(f"{C.GRAY}{'─' * 72}{C.RESET}")


def print_result(res: dict):
    """Напечатать результат анализа одной монеты."""
    symbol = res["symbol"]

    if res.get("error"):
        print(f"\n  ⚠️  {C.YELLOW}{C.BOLD}{symbol}{C.RESET}  —  {C.RED}{res['error']}{C.RESET}")
        print_separator()
        return

    badge = direction_badge(res["direction"])
    close = res["close"]
    ts = res["timestamp"]

    print(f"\n  {C.BOLD}{C.CYAN}{'═' * 68}{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}  {symbol}{C.RESET}   │  Close: {C.BOLD}${close:,.4f}{C.RESET}  │  🕐 {ts}")
    print(f"  {C.BOLD}{C.CYAN}{'═' * 68}{C.RESET}")

    # Направление
    print(f"\n     {badge}")

    # Вероятности
    print(f"\n  {C.BOLD}📊 Вероятности:{C.RESET}")
    print(f"     🟢 Long    {prob_bar(res['p_long'])}")
    print(f"     ⚪ Neutral {prob_bar(res['p_neutral'])}")
    print(f"     🔴 Short   {prob_bar(res['p_short'])}")

    # MFE
    print(f"\n  {C.BOLD}📐 MFE (Max Favorable Excursion):{C.RESET}")
    print(f"     ⬆️  MFE Long   {mfe_bar(res['mfe_long'], MFE_THRESHOLD)}")
    print(f"     ⬇️  MFE Short  {mfe_bar(res['mfe_short'], MFE_THRESHOLD)}")

    # Пороги
    mfe_icon = "✅" if res["mfe_pass"] else "❌"
    print(f"\n  {C.BOLD}🎯 Пороги (Thresholds):{C.RESET}")
    print(f"     Confidence   : {CONFIDENCE_THRESHOLD*100:.0f}%  │  Сигнал: {res['signal_prob']*100:.1f}%  {'✅' if res['signal_prob'] >= CONFIDENCE_THRESHOLD else '❌'}")
    print(f"     MFE Threshold: {MFE_THRESHOLD*100:.1f}%  │  MFE:    {res['mfe_relevant']*100:.2f}%  {mfe_icon}")
    print(f"     TP: +{TP_PCT*100:.1f}%   │  SL: -{SL_PCT*100:.1f}%   │  Horizon: {HORIZON} свечей")

    # Verdict
    if res["direction"] != "NEUTRAL" and res["mfe_pass"]:
        verdict = f"  {C.GREEN}{C.BOLD}✅ СИГНАЛ АКТИВЕН — {res['direction']}!{C.RESET}"
    elif res["direction"] != "NEUTRAL" and not res["mfe_pass"]:
        verdict = f"  {C.YELLOW}{C.BOLD}⚠️  Направление есть, но MFE < порога — НЕТ ВХОДА{C.RESET}"
    else:
        verdict = f"  {C.DIM}💤 Нет выраженного направления — ОЖИДАНИЕ{C.RESET}"
    print(f"\n  {verdict}")

    # Фичи модели
    print(f"\n  {C.BOLD}🔧 Фичи:{C.RESET}")
    ema_v = res.get("ema_200")
    if ema_v is not None and not pd.isna(ema_v):
        print(f"     EMA 200      : {ema_v:,.4f}")
    print(f"     Тренд {TIMEFRAME}     : {trend_icon(res['trend'])}")
    print(f"     Тренд {HTF_TIMEFRAME}     : {trend_icon(res['htf_trend'])}")
    dtr = res.get("dist_to_resistance")
    dts = res.get("dist_to_support")
    if dtr is not None and not pd.isna(dtr):
        print(f"     Dist→Res (ATR): {dtr:.2f}")
    if dts is not None and not pd.isna(dts):
        print(f"     Dist→Sup (ATR): {dts:.2f}")

    print(f"     {C.DIM}Свечей обработано: {res['candles_count']}{C.RESET}")
    print_separator()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🔍 Сканер монет — анализ направления, MFE и порогов"
    )
    parser.add_argument(
        "--datetime", "-d",
        type=str,
        default=None,
        help='Дата-время для исторического анализа, например: "2025-12-01 12:00"'
    )
    args = parser.parse_args()

    # Определяем момент времени
    if args.datetime:
        try:
            dt = datetime.fromisoformat(args.datetime)
            end_ms = int(dt.timestamp() * 1000)
            time_label = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            print(f"❌ Неправильный формат даты: {args.datetime}")
            print(f"   Используйте: --datetime \"2025-12-01 12:00\"")
            sys.exit(1)
    else:
        end_ms = None
        time_label = datetime.now().strftime("%Y-%m-%d %H:%M") + " (текущий)"

    # ─── Заголовок ───
    print()
    print(f"  {C.BOLD}{C.MAGENTA}╔{'═' * 66}╗{C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}║{C.WHITE}   🔍  CRYPTO SCANNER  —  Анализ направления и MFE              {C.MAGENTA}║{C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}╚{'═' * 66}╝{C.RESET}")
    print()
    print(f"  🕐 Время анализа : {C.BOLD}{time_label}{C.RESET}")
    print(f"  📊 Таймфрейм     : {C.BOLD}{TIMEFRAME}{C.RESET} + {HTF_TIMEFRAME} (HTF)")
    print(f"  🪙 Монеты        : {C.BOLD}{', '.join(SYMBOLS)}{C.RESET}")
    print(f"  🎯 Confidence    : {C.BOLD}{CONFIDENCE_THRESHOLD*100:.0f}%{C.RESET}")
    print(f"  📐 MFE Threshold : {C.BOLD}{MFE_THRESHOLD*100:.1f}%{C.RESET}")
    print(f"  🎰 TP / SL       : {C.BOLD}+{TP_PCT*100:.1f}% / -{SL_PCT*100:.1f}%{C.RESET}")
    print()
    print_separator()

    # Загрузка моделей
    print(f"\n  ⏳ Загрузка моделей...", end="", flush=True)
    try:
        clf, reg_long, reg_short, feature_names = load_models()
        print(f" {C.GREEN}✅  OK ({len(feature_names)} фич){C.RESET}")
    except Exception as e:
        print(f"\n  {C.RED}❌ Ошибка загрузки моделей: {e}{C.RESET}")
        print(f"  {C.DIM}   Сначала запустите train.py для обучения моделей.{C.RESET}")
        sys.exit(1)

    # Сканируем каждую монету
    results = []
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"  ⏳ [{i}/{len(SYMBOLS)}] Анализ {C.BOLD}{symbol}{C.RESET}...", end="", flush=True)
        res = analyse_symbol(symbol, clf, reg_long, reg_short, feature_names, end_ms)
        if res and not res.get("error"):
            print(f" {C.GREEN}✅{C.RESET}")
        elif res and res.get("error"):
            print(f" {C.YELLOW}⚠{C.RESET}")
        results.append(res)

    # Итоговый вывод
    print(f"\n\n  {C.BOLD}{C.CYAN}{'═' * 68}{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}  📋  РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ{C.RESET}")
    print(f"  {C.BOLD}{C.CYAN}{'═' * 68}{C.RESET}")

    for res in results:
        if res:
            print_result(res)

    # Сводная таблица
    active = [r for r in results if r and not r.get("error") and r.get("direction") != "NEUTRAL" and r.get("mfe_pass")]
    waiting = [r for r in results if r and not r.get("error") and (r.get("direction") == "NEUTRAL" or not r.get("mfe_pass"))]
    errors = [r for r in results if r and r.get("error")]

    print(f"\n  {C.BOLD}{'═' * 68}{C.RESET}")
    print(f"  {C.BOLD}  📊  СВОДКА{C.RESET}")
    print(f"  {C.BOLD}{'═' * 68}{C.RESET}")

    if active:
        print(f"\n  {C.GREEN}{C.BOLD}  🚀 Активные сигналы ({len(active)}):{C.RESET}")
        for r in active:
            dir_emoji = "🟢" if r["direction"] == "LONG" else "🔴"
            print(f"     {dir_emoji} {r['symbol']:<12} {r['direction']:<6}  prob={r['signal_prob']*100:.1f}%  MFE={r['mfe_relevant']*100:.2f}%")
    else:
        print(f"\n  {C.DIM}  💤 Активных сигналов нет{C.RESET}")

    if waiting:
        print(f"\n  {C.YELLOW}  ⏳ Ожидание ({len(waiting)}):{C.RESET}")
        for r in waiting:
            reason = "MFE < порога" if r.get("direction") != "NEUTRAL" else "Нет направления"
            print(f"     ⚪ {r['symbol']:<12} {reason}")

    if errors:
        print(f"\n  {C.RED}  ❌ Ошибки ({len(errors)}):{C.RESET}")
        for r in errors:
            print(f"     ⚠️  {r['symbol']:<12} {r['error']}")

    print(f"\n  {C.DIM}{'─' * 68}{C.RESET}")
    print(f"  {C.DIM}  Сканирование завершено: {datetime.now().strftime('%H:%M:%S')}{C.RESET}")
    print()


if __name__ == "__main__":
    main()
