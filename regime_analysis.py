#!/usr/bin/env python3
"""
Анализ режимов (ADX, Hurst, вола) vs направление forward-доходности.
Данные: те же таблицы *_features, что и train.py (после etl.py).
"""
from __future__ import annotations

import sqlite3
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from config import (
        ADX_LENGTH,
        DB_PATH,
        FEATURE_COLUMNS,
        FEATURE_SET_VERSION,
        HORIZON,
        SYMBOLS,
        TIMEFRAME,
    )
except ImportError as e:
    print(f"Ошибка импорта config: {e}")
    sys.exit(1)

# Пороги для подсказок (как в вашем черновике; не обязаны совпадать с bt.py)
ADX_THRESHOLD = 20.0
HURST_THRESHOLD = 0.55
VOL_PERCENTILE = 80

OUTPUT_PNG = Path("regime_analysis.png")


def load_ml_data_from_db() -> pd.DataFrame:
    """Та же схема, что train.load_data_from_db (без импорта train → без CatBoost)."""
    conn = sqlite3.connect(DB_PATH)
    all_data = []
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_features';"
    )
    tables = cursor.fetchall()

    for table_name in tables:
        table = table_name[0]
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        raw_name = table.replace("_features", "")
        df["symbol"] = raw_name.replace("_", "/", 1) if "_" in raw_name else raw_name

        needed = ["Target_Long_Return", "Target_Short_Return"]
        if not all(col in df.columns for col in needed):
            print(f"[!] Пропуск {table}: нет {needed}")
            continue

        missing_feat = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_feat:
            raise ValueError(
                f"Таблица «{table}» устарела (FEATURE_SET_VERSION={FEATURE_SET_VERSION}). "
                f"Нет колонок: {missing_feat[:10]}… Запустите: python etl.py"
            )

        df = df.dropna(
            subset=["timestamp", *FEATURE_COLUMNS, "Target_Long_Return", "Target_Short_Return"]
        ).copy()
        all_data.append(df)

    conn.close()

    if not all_data:
        return pd.DataFrame()

    out = pd.concat(all_data, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    # Только символы из config (если в БД есть лишние таблицы)
    out = out[out["symbol"].isin(SYMBOLS)].reset_index(drop=True)
    return out


def add_forward_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward return close[t+h]/close[t]-1 внутри каждого symbol (без утечки между монетами)."""
    df = df.copy()
    g = df.groupby("symbol", sort=False)
    df["fwd_ret_h"] = g["close"].transform(lambda s: s.shift(-horizon) / s - 1.0)
    return df


def analyze_regime_performance(df: pd.DataFrame, output_png: Path, show_plot: bool) -> None:
    adx_col = f"ADX_{ADX_LENGTH}"
    metrics = [adx_col, "hurst_rs", "vol_24", "atr_pct"]
    existing_metrics = [m for m in metrics if m in df.columns]

    if "fwd_ret_h" not in df.columns:
        print("[!] Нет fwd_ret_h — добавьте add_forward_return.")
        return

    target_col = "fwd_ret_h"
    df = df.dropna(subset=[target_col]).copy()

    signal_mask = df[target_col].abs() > 0.002
    df_signals = df.loc[signal_mask].copy()
    df_signals["is_positive_move"] = df_signals[target_col] > 0

    print("\nСТАТИСТИКА РЕЖИМОВ")
    print("-" * 60)
    print(f"Таймфрейм: {TIMEFRAME}, forward H={HORIZON} баров → fwd_ret_h")
    print(f"Точек после фильтра |fwd_ret_h|>0.2%: {len(df_signals)} / {len(df)}")

    if not existing_metrics:
        print("[!] Нет ADX/Hurst/vol_24/atr_pct в данных.")
        return

    print(f"\n{'Метрика':<18} | {'Среднее (все строки)':<20} | {'Среднее (|fwd|>0.2%)':<20}")
    print("-" * 60)
    for m in existing_metrics:
        mean_all = df[m].mean()
        mean_sig = df_signals[m].mean()
        print(f"{m:<18} | {mean_all:20.6f} | {mean_sig:20.6f}")

    if len(df_signals) > 0 and target_col:
        print("\nСРАВНЕНИЕ: положительный fwd_ret_h vs отрицательный")
        print("-" * 70)
        print(f"{'Метрика':<18} | {'Win Avg':<12} | {'Loss Avg':<12} | {'Diff':<12} | Вердикт")
        print("-" * 70)

        wins = df_signals[df_signals["is_positive_move"]]
        losses = df_signals[~df_signals["is_positive_move"]]

        for m in existing_metrics:
            win_avg = wins[m].mean()
            loss_avg = losses[m].mean()
            diff = win_avg - loss_avg

            verdict = "Neutral"
            if m == adx_col:
                if diff > 2:
                    verdict = "Trend Helps"
                elif diff < -2:
                    verdict = "Chop Kills"
            elif m == "hurst_rs":
                if diff > 0.05:
                    verdict = "Trend Helps"
                elif diff < -0.05:
                    verdict = "MeanRev Helps"

            print(f"{m:<18} | {win_avg:12.6f} | {loss_avg:12.6f} | {diff:+12.6f} | {verdict}")

    n_plots = min(4, len(existing_metrics))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        if i < n_plots:
            m = existing_metrics[i]
            if target_col and len(df_signals) > 0:
                wins = df_signals[df_signals["is_positive_move"]][m]
                losses = df_signals[~df_signals["is_positive_move"]][m]
                if len(wins):
                    wins.hist(bins=50, alpha=0.6, label="Win (fwd>0)", ax=ax, color="green")
                if len(losses):
                    losses.hist(bins=50, alpha=0.6, label="Loss (fwd<=0)", ax=ax, color="red")
            else:
                df_signals[m].hist(bins=50, alpha=0.7, ax=ax, color="blue")
            ax.set_title(f"Distribution of {m}")
            ax.set_xlabel(m)
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"\nГрафик сохранён: {output_png.resolve()}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    # Подсказки по порогам
    print("\nПОДСКАЗКИ ПО ФИЛЬТРАЦИИ (эвристика; сверяйте с bt.ENABLE_REGIME_FILTER)")
    if adx_col in existing_metrics and len(df_signals) > 0:
        w = df_signals[df_signals["is_positive_move"]][adx_col]
        l = df_signals[~df_signals["is_positive_move"]][adx_col]
        if len(w) and len(l):
            if w.mean() > l.mean():
                print(f"   → ADX: выигрышные движения в среднем при более высоком ADX; порог флэта ~ {l.mean():.1f}")
            else:
                print("   → ADX: среднее на выигрышах ниже, чем на проигрышах — проверьте логику/выборку.")

    if "hurst_rs" in existing_metrics and len(df_signals) > 0:
        w = df_signals[df_signals["is_positive_move"]]["hurst_rs"]
        l = df_signals[~df_signals["is_positive_move"]]["hurst_rs"]
        if len(w) and len(l):
            print(f"   → Hurst: win_mean={w.mean():.3f}, loss_mean={l.mean():.3f} (порог тренда часто ~0.5)")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Режимный анализ по данным ETL/train")
    p.add_argument("--output", type=Path, default=OUTPUT_PNG, help="PNG с гистограммами")
    p.add_argument("--show", action="store_true", help="Показать окно matplotlib после сохранения")
    args = p.parse_args()

    print(f"Загрузка из {DB_PATH} (символы из БД; в config SYMBOLS={len(SYMBOLS)} шт.)…")
    df = load_ml_data_from_db()
    if df.empty:
        raise ValueError("Нет данных: проверьте data/market_data.db и таблицы *_features.")

    print("Расчёт fwd_ret_h по символам…")
    df = add_forward_return(df, HORIZON)
    df = df.dropna(subset=["fwd_ret_h"])

    n0 = len(df)
    df = df.dropna()
    print(f"Готово: {len(df)} строк после dropna (отброшено {n0 - len(df)} с NaN в фичах/fwd).")

    analyze_regime_performance(df, args.output, args.show)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)
