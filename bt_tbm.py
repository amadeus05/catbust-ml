#!/usr/bin/env python3
"""
Бэктест walk-forward для TBM (CatBoostClassifier).

prob = P(класс 1): в разметке etl это «верхний барьер раньше нижнего» (лонг-ориентированный сценарий).
LONG:  prob > TBM_LONG_MIN_PROB
SHORT: prob < TBM_SHORT_MAX_PROB (жёстче: класс 0 смешивает нижний барьер и таймаут)

Требуется: etl (tbm_label), train ML_LABEL_MODE=tbm, пороги в config.
"""
from __future__ import annotations

import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import *
from train import get_purge_gap, load_data_from_db, train_single_classifier

from bt import (
    align_dataframes_to_common_index,
    build_walk_forward_folds,
    calculate_unrealized_pnl,
    purge_bars_count,
    regime_allows_trade,
    split_train_for_early_stopping,
)

TAKER_COM = 0.0004
SLIPPAGE = 0.0003
LEVERAGE = 1
RISK_PER_TRADE = 0.01


def load_all_data_tbm(symbols, feature_names):
    conn = sqlite3.connect(DB_PATH)
    all_dfs = {}
    print(f"Загрузка данных для {len(symbols)} монет (TBM, без регрессионных таргетов)...")
    for sym in symbols:
        table_name = f"{sym.replace('/', '_')}_features"
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cols_to_keep = ["timestamp", "open", "high", "low", "close"] + list(feature_names)
            missing = [c for c in cols_to_keep if c not in df.columns]
            if missing:
                print(f"[!] Пропуск {sym}: нет колонок {missing[:6]}...")
                continue
            df = df[cols_to_keep].sort_values("timestamp").reset_index(drop=True)
            all_dfs[sym] = df
        except Exception as e:
            print(f"[!] Ошибка загрузки {sym}: {e}")
    conn.close()
    return all_dfs


def train_models_walk_forward_slice_tbm(df_train: pd.DataFrame, verbose: bool = False):
    features = list(FEATURE_COLUMNS)
    tr, va = split_train_for_early_stopping(df_train)
    if len(va) < 30 or len(tr) < 50:
        tr, va = split_train_for_early_stopping(df_train, frac=0.75)

    X_tr, X_va = tr[features], va[features]
    y_tr = tr["tbm_label"].astype(np.int32)
    y_va = va["tbm_label"].astype(np.int32)

    model, _, metrics, _ = train_single_classifier(X_tr, y_tr, X_va, y_va, verbose=verbose)
    return {
        "tbm_model": model,
        "auc": metrics["auc"],
        "logloss": metrics["logloss"],
    }


def print_fold_train_metrics_tbm(fid: int, res: dict) -> None:
    print(
        f"     TBM  val AUC={res['auc']:.4f} | LogLoss={res['logloss']:.4f}"
    )


def train_all_walk_forward_models_tbm(df_ml, folds, common_timestamps, purge_gap):
    trained_packs = []
    print("\n" + "=" * 72)
    print("ФАЗА 1 — обучение TBM (CatBoostClassifier) по фолдам walk-forward")
    print("=" * 72)

    for fold in folds:
        fid = fold["fold_id"]
        train_start_ts = fold["train_start_ts"]
        first_test_ts = fold["first_test_ts"]
        train_cutoff = first_test_ts - purge_gap

        train_mask = (df_ml["timestamp"] >= train_start_ts) & (df_ml["timestamp"] <= train_cutoff)
        df_train = df_ml.loc[train_mask].copy()

        if len(df_train) < WF_MIN_TRAIN_ROWS:
            print(f"\n[!] Фолд {fid}: train {len(df_train)} < {WF_MIN_TRAIN_ROWS} — пропуск")
            continue

        print(f"\n--- Фолд {fid} | train: {train_start_ts} … {train_cutoff} | rows={len(df_train)} ---")
        print(f"    OOS-период: {fold['first_test_ts']} … {fold['last_test_ts']}")

        try:
            res = train_models_walk_forward_slice_tbm(df_train, verbose=False)
        except Exception as e:
            print(f"[!] Фолд {fid}: ошибка обучения: {e}")
            continue

        print_fold_train_metrics_tbm(fid, res)
        trained_packs.append(
            {
                "fold_id": fid,
                "tbm_model": res["tbm_model"],
                "test_ts_slice": common_timestamps[fold["test_start_idx"] : fold["test_end_idx"]],
                "test_start_ts": fold["first_test_ts"],
                "test_end_ts": fold["last_test_ts"],
                "res": res,
                "n_train": len(df_train),
            }
        )

    if trained_packs:
        print("\n" + "=" * 72)
        print("Сводка val по фолдам (TBM)")
        print("=" * 72)
        for p in trained_packs:
            r = p["res"]
            print(f"  fold {p['fold_id']}: AUC={r['auc']:.4f}  LogLoss={r['logloss']:.4f}  train_n={p['n_train']}")
        print("=" * 72)

    return trained_packs


def build_continuous_oos_context_tbm(trained_packs, all_dfs_bt, common_timestamps):
    required_oos_cols = ["open", "high", "low", "close"] + list(FEATURE_COLUMNS)
    all_aligned = align_dataframes_to_common_index(
        all_dfs_bt, common_timestamps, required_cols=required_oos_cols
    )
    if all_aligned is None:
        return None, None

    fold_by_ts = {}
    for pack in trained_packs:
        for ts in pack["test_ts_slice"]:
            fold_by_ts[ts] = pack["fold_id"]

    valid_ts = [ts for ts in common_timestamps if ts in fold_by_ts]
    if len(valid_ts) < 2:
        return None, None

    filtered_aligned = {}
    ts_index = pd.DatetimeIndex(valid_ts)
    for sym, df in all_aligned.items():
        tmp = (
            df.set_index("timestamp")
            .loc[ts_index]
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        filtered_aligned[sym] = tmp

    pack_by_fold = {p["fold_id"]: p for p in trained_packs}
    return filtered_aligned, {
        "valid_ts": valid_ts,
        "fold_by_ts": fold_by_ts,
        "pack_by_fold": pack_by_fold,
    }


def run_oos_simulation_tbm(
    trained_packs,
    all_dfs_bt,
    feature_names,
    common_timestamps,
    initial_balance,
    verbose_trades,
    long_min_prob: float,
    short_max_prob: float,
):
    lo = float(long_min_prob)
    shi = float(short_max_prob)
    print("\n" + "=" * 72)
    print(
        f"ФАЗА 2 — OOS TBM | LONG: prob > {lo:.2f} | SHORT: prob < {shi:.2f} "
        f"(prob = P(верхний барьер раньше нижнего))"
    )
    print("=" * 72)

    aligned, ctx = build_continuous_oos_context_tbm(trained_packs, all_dfs_bt, common_timestamps)
    if aligned is None or ctx is None:
        print("[!] Не удалось собрать OOS-ленту.")
        return {
            "balance": initial_balance,
            "all_trades": [],
            "equity_all_ts": [],
            "equity_all_val": [],
            "max_dd_global": 0.0,
            "fold_stats": {},
        }

    test_timestamps = ctx["valid_ts"]
    fold_by_ts = ctx["fold_by_ts"]
    pack_by_fold = ctx["pack_by_fold"]

    balance = float(initial_balance)
    positions = {sym: None for sym in aligned}
    trades = []
    equity_curve = []
    equity_timestamps = []
    peak_balance = balance
    max_drawdown = 0.0
    used_margin = 0.0
    fold_stats = {}
    current_fold_id = None

    for i in range(len(test_timestamps) - 1):
        current_ts = test_timestamps[i]
        next_ts = test_timestamps[i + 1]

        fold_id = fold_by_ts[current_ts]
        pack = pack_by_fold[fold_id]
        tbm_model = pack["tbm_model"]

        if current_fold_id != fold_id:
            if current_fold_id is not None and current_fold_id in fold_stats:
                prev = fold_stats[current_fold_id]
                wr = (prev["wins"] / prev["trades"] * 100) if prev["trades"] > 0 else 0.0
                print(
                    f"   итог fold {current_fold_id}: trades={prev['trades']} | WR={wr:.1f}% | "
                    f"W/L={prev['wins']}/{prev['losses']} | L/S={prev['longs']}/{prev['shorts']} | "
                    f"balance={balance:.2f}$"
                )

            current_fold_id = fold_id
            if fold_id not in fold_stats:
                fold_stats[fold_id] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "longs": 0,
                    "shorts": 0,
                    "start_balance": balance,
                }

            print(
                f"\n↔ SWITCH TBM pack -> fold {fold_id} | "
                f"OOS {pack['test_start_ts']} … {pack['test_end_ts']} | balance={balance:.2f}$"
            )

        current_equity = balance
        for sym, pos in positions.items():
            if pos is not None:
                current_equity += calculate_unrealized_pnl(pos, aligned[sym].iloc[i]["close"])

        equity_curve.append(current_equity)
        equity_timestamps.append(current_ts)

        if current_equity > peak_balance:
            peak_balance = current_equity
        current_dd = (peak_balance - current_equity) / peak_balance * 100 if peak_balance > 0 else 0.0
        if current_dd > max_drawdown:
            max_drawdown = current_dd

        for sym, df in aligned.items():
            next_row = df.iloc[i + 1]
            next_open = next_row["open"]
            next_high = next_row["high"]
            next_low = next_row["low"]

            if positions[sym] is not None:
                pos = positions[sym]
                entry_price = pos["entry"]
                direction = pos["dir"]
                position_notional = pos["size"]
                exit_signal = False
                exit_price = 0.0
                reason = ""

                if direction == 1:
                    stop_price = entry_price * (1 - SL_PCT)
                    take_price = entry_price * (1 + TP_PCT)
                    if next_low <= stop_price and next_high >= take_price:
                        exit_price = (next_open if next_open < stop_price else stop_price) * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "SL"
                    elif next_low <= stop_price:
                        exit_price = (next_open if next_open < stop_price else stop_price) * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "SL"
                    elif next_high >= take_price:
                        exit_price = take_price * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "TP"
                else:
                    stop_price = entry_price * (1 + SL_PCT)
                    take_price = entry_price * (1 - TP_PCT)
                    if next_high >= stop_price and next_low <= take_price:
                        exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "SL"
                    elif next_high >= stop_price:
                        exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "SL"
                    elif next_low <= take_price:
                        exit_price = take_price * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "TP"

                if exit_signal:
                    if direction == 1:
                        raw_pnl = (exit_price - entry_price) / entry_price
                    else:
                        raw_pnl = (entry_price - exit_price) / entry_price
                    commission = position_notional * (TAKER_COM + TAKER_COM)
                    pnl_clean = raw_pnl - (TAKER_COM + TAKER_COM)
                    trade_profit = position_notional * pnl_clean
                    used_margin -= pos["margin"]
                    if used_margin < 0:
                        used_margin = 0.0
                    balance += trade_profit
                    trades.append(
                        {
                            "sym": sym,
                            "dir": direction,
                            "pnl_pct": pnl_clean,
                            "pnl_abs": trade_profit,
                            "ts": next_ts,
                            "fold_id": fold_id,
                        }
                    )
                    fold_stats[fold_id]["trades"] += 1
                    if pnl_clean > 0:
                        fold_stats[fold_id]["wins"] += 1
                    else:
                        fold_stats[fold_id]["losses"] += 1
                    if direction == 1:
                        fold_stats[fold_id]["longs"] += 1
                    else:
                        fold_stats[fold_id]["shorts"] += 1
                    positions[sym] = None
                    if verbose_trades:
                        print(
                            f"[{next_ts}] {sym}: {reason} | PnL: {pnl_clean*100:.2f}% | "
                            f"Com: {commission:.2f}$ | Bal: {balance:.2f}"
                        )
                    continue

            if positions[sym] is None:
                current_row = df.iloc[i]
                current_features = df.iloc[[i]][feature_names]
                # Класс 1 = касание верхнего барьера раньше нижнего (метка «лонг-успех» в TBM).
                prob = float(tbm_model.predict_proba(current_features)[0, 1])

                long_ok = prob > lo and regime_allows_trade(current_row, side=1)
                # Шорт: низкая prob — модель не ожидает «верх первым»; порог short_max_prob < long_min из-за класса 0 (таймауты).
                short_ok = prob < shi and regime_allows_trade(current_row, side=-1)

                signal = 0
                if TRADE_MODE == "long_only":
                    if long_ok:
                        signal = 1
                elif TRADE_MODE == "short_only":
                    if short_ok:
                        signal = -1
                else:
                    if long_ok and short_ok:
                        dist_long = prob - lo
                        dist_short = shi - prob
                        signal = 1 if dist_long >= dist_short else -1
                    elif long_ok:
                        signal = 1
                    elif short_ok:
                        signal = -1

                if signal != 0:
                    risk_capital = balance * RISK_PER_TRADE
                    position_notional = risk_capital / SL_PCT
                    position_notional = min(position_notional, balance * LEVERAGE)
                    required_margin = position_notional / LEVERAGE
                    available_balance = balance - used_margin
                    if required_margin > available_balance:
                        required_margin = available_balance
                        position_notional = required_margin * LEVERAGE
                    if position_notional < 10:
                        continue
                    used_margin += required_margin
                    if signal == 1:
                        entry_price = next_open * (1 + SLIPPAGE)
                    else:
                        entry_price = next_open * (1 - SLIPPAGE)
                    positions[sym] = {
                        "dir": signal,
                        "entry": entry_price,
                        "size": position_notional,
                        "margin": required_margin,
                        "ts_open": next_ts,
                        "prob_tp": prob,
                        "fold_id": fold_id,
                    }
                    dir_str = "LONG" if signal == 1 else "SHORT"
                    if verbose_trades:
                        print(
                            f"[{next_ts}] {sym}: OPEN {dir_str} prob(class1)={prob:.3f} "
                            f"(long>{lo:.2f}, short< {shi:.2f}) | "
                            f"entry {entry_price:.4f} | size {position_notional:.1f}$"
                        )

    if len(test_timestamps) > 0:
        last_ts = test_timestamps[-1]
        last_fold_id = fold_by_ts[last_ts]
        for sym, pos in list(positions.items()):
            if pos is None:
                continue
            last_close = aligned[sym].iloc[-1]["close"]
            entry_price = pos["entry"]
            direction = pos["dir"]
            position_notional = pos["size"]
            if direction == 1:
                raw_pnl = (last_close * (1 - SLIPPAGE) - entry_price) / entry_price
            else:
                raw_pnl = (entry_price - last_close * (1 + SLIPPAGE)) / entry_price
            pnl_clean = raw_pnl - (TAKER_COM * 2)
            trade_profit = position_notional * pnl_clean
            used_margin -= pos["margin"]
            if used_margin < 0:
                used_margin = 0.0
            balance += trade_profit
            trades.append(
                {
                    "sym": sym,
                    "dir": direction,
                    "pnl_pct": pnl_clean,
                    "pnl_abs": trade_profit,
                    "ts": last_ts,
                    "fold_id": last_fold_id,
                }
            )
            fold_stats[last_fold_id]["trades"] += 1
            if pnl_clean > 0:
                fold_stats[last_fold_id]["wins"] += 1
            else:
                fold_stats[last_fold_id]["losses"] += 1
            if direction == 1:
                fold_stats[last_fold_id]["longs"] += 1
            else:
                fold_stats[last_fold_id]["shorts"] += 1
            positions[sym] = None
            if verbose_trades:
                print(f"[{last_ts}] {sym}: FORCE EXIT | PnL: {pnl_clean*100:.2f}% | Bal: {balance:.2f}")

        final_equity = balance
        equity_curve.append(final_equity)
        equity_timestamps.append(last_ts)
        if final_equity > peak_balance:
            peak_balance = final_equity
        current_dd = (peak_balance - final_equity) / peak_balance * 100 if peak_balance > 0 else 0.0
        if current_dd > max_drawdown:
            max_drawdown = current_dd

    if current_fold_id is not None and current_fold_id in fold_stats:
        prev = fold_stats[current_fold_id]
        wr = (prev["wins"] / prev["trades"] * 100) if prev["trades"] > 0 else 0.0
        print(
            f"   итог fold {current_fold_id}: trades={prev['trades']} | WR={wr:.1f}% | "
            f"balance={balance:.2f}$"
        )

    return {
        "balance": balance,
        "all_trades": trades,
        "equity_all_ts": equity_timestamps,
        "equity_all_val": equity_curve,
        "max_dd_global": max_drawdown,
        "fold_stats": fold_stats,
    }


def backtest_tbm():
    if globals().get("ML_LABEL_MODE", "trade_return") != "tbm":
        print("[!] В config задайте ML_LABEL_MODE = \"tbm\" и пересоберите данные (etl + train).")
        return

    long_min = float(
        globals().get("TBM_LONG_MIN_PROB", globals().get("TBM_CONFIDENCE_THRESHOLD", 0.55))
    )
    short_max = float(globals().get("TBM_SHORT_MAX_PROB", 0.35))
    print("Walk-forward + OOS (TBM): загрузка...")
    purge_gap = get_purge_gap()
    print(f"Purge gap: {purge_gap} ({purge_bars_count(purge_gap)} баров)")

    df_ml = load_data_from_db()
    if df_ml.empty or "tbm_label" not in df_ml.columns:
        print("[!] Нет данных или колонки tbm_label — запустите etl.py")
        return

    df_ml["timestamp"] = pd.to_datetime(df_ml["timestamp"])
    df_ml = df_ml.sort_values("timestamp").reset_index(drop=True)

    feature_names = list(FEATURE_COLUMNS)
    all_dfs_bt = load_all_data_tbm(SYMBOLS, feature_names)
    if not all_dfs_bt:
        print("[!] Нет данных для бэктеста")
        return

    common_timestamps = sorted(set.intersection(*(set(df["timestamp"]) for df in all_dfs_bt.values())))
    if not common_timestamps:
        print("[!] Нет общих timestamp")
        return

    folds = build_walk_forward_folds(common_timestamps, purge_gap)
    if not folds:
        return

    print(
        f"\nЗапланировано фолдов: {len(folds)} | "
        f"TBM LONG prob>{long_min:.2f} | SHORT prob<{short_max:.2f}"
    )

    trained_packs = train_all_walk_forward_models_tbm(df_ml, folds, common_timestamps, purge_gap)
    if not trained_packs:
        print("[!] Нет обученных фолдов.")
        return

    initial_balance = 100.0
    sim = run_oos_simulation_tbm(
        trained_packs,
        all_dfs_bt,
        feature_names,
        common_timestamps,
        initial_balance,
        verbose_trades=WF_VERBOSE_TRADES,
        long_min_prob=long_min,
        short_max_prob=short_max,
    )

    balance = sim["balance"]
    all_trades = sim["all_trades"]
    equity_all_ts = sim["equity_all_ts"]
    equity_all_val = sim["equity_all_val"]
    max_dd_global = sim["max_dd_global"]

    monthly_global = {}
    for t in all_trades:
        mk = pd.Timestamp(t["ts"]).strftime("%Y-%m")
        if mk not in monthly_global:
            monthly_global[mk] = {"pnl_abs": 0.0, "trades": 0, "wins": 0}
        monthly_global[mk]["trades"] += 1
        monthly_global[mk]["pnl_abs"] += t["pnl_abs"]
        if t["pnl_pct"] > 0:
            monthly_global[mk]["wins"] += 1

    print("\n" + "=" * 50)
    print("WALK-FORWARD TBM: ИТОГ OOS")
    print("=" * 50)
    print(f"{'Месяц':<10} | {'Сделок':<8} | {'WinRate':<8} | {'PnL $':<14}")
    print("-" * 50)

    total_pnl_abs = 0.0
    total_trades = 0
    total_wins = 0
    for m in sorted(monthly_global.keys()):
        stats = monthly_global[m]
        count = stats["trades"]
        wins = stats["wins"]
        pnl_abs = stats["pnl_abs"]
        wr = (wins / count * 100) if count > 0 else 0.0
        total_pnl_abs += pnl_abs
        total_trades += count
        total_wins += wins
        print(f"{m:<10} | {count:<8} | {wr:<7.1f}% | {pnl_abs:+.2f}$")

    print("-" * 50)
    final_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return_pct = ((balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0
    print(f"ИТОГО      | {total_trades:<8} | {final_wr:.1f}%     | {total_return_pct:+.2f}% ({total_pnl_abs:+.2f}$)")
    print(f"Конечный баланс: {balance:.2f}$ (старт {initial_balance}$)")
    print(f"Макс. просадка: {max_dd_global:.2f}%")

    if len(equity_all_val) > 1 and all_trades:
        rets = np.array([t["pnl_abs"] for t in all_trades])
        gp = rets[rets > 0].sum()
        gl = abs(rets[rets < 0].sum())
        pf = gp / gl if gl > 0 else float("inf")
        print(f"\nProfit Factor: {pf:.2f}")

    if len(equity_all_val) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_all_ts, equity_all_val, "b-", label="TBM OOS Equity")
        plt.axhline(y=initial_balance, color="gray", linestyle="--")
        plt.title(
            f"TBM WF OOS | long>{long_min} short<{short_max} | "
            f"trades={total_trades} | DD~{max_dd_global:.1f}%"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("equity_curve_tbm.png", dpi=150)
        plt.show()
        print("\nСохранено: equity_curve_tbm.png")


if __name__ == "__main__":
    backtest_tbm()
