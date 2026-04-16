import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *

from train import (
    load_data_from_db,
    get_purge_gap,
    FEATURE_COLUMNS,
    train_single_regressor,
    calibrate_side_threshold,
)

# === ФЬЮЧЕРСЫ ===
TAKER_COM = 0.0004
SLIPPAGE = 0.0003
LEVERAGE = 1
RISK_PER_TRADE = 0.01

# Порог теперь калибруется на каждом fold по внутреннему validation.
DEFAULT_LONG_THRESHOLD = CALIBRATION_DEFAULT_LONG_THRESHOLD
DEFAULT_SHORT_THRESHOLD = CALIBRATION_DEFAULT_SHORT_THRESHOLD

TF_TO_HOURS = {"1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}


def bars_per_day_tf():
    h = TF_TO_HOURS.get(TIMEFRAME, 1)
    return max(1, int(round(24 / h)))


def purge_bars_count(purge_gap: pd.Timedelta) -> int:
    bar_h = TF_TO_HOURS.get(TIMEFRAME, 1)
    hours = purge_gap.total_seconds() / 3600
    return max(1, int(np.ceil(hours / bar_h)))


def load_all_data(symbols, feature_names):
    conn = sqlite3.connect(DB_PATH)
    all_dfs = {}

    print(f"Загрузка данных для {len(symbols)} монет...")
    for sym in symbols:
        table_name = f"{sym.replace('/', '_')}_features"
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                cols_to_keep = [
                    "timestamp", "open", "high", "low", "close",
                    "Target_Long_Return", "Target_Short_Return"
                ] + list(feature_names)

                missing_cols = [c for c in cols_to_keep if c not in df.columns]
                if missing_cols:
                    print(f"⚠️ Пропуск {sym}: отсутствуют колонки {missing_cols}")
                    continue

                df = df[cols_to_keep].sort_values("timestamp").reset_index(drop=True)
                all_dfs[sym] = df
            else:
                print(f"⚠️ Пропуск {sym}: нет колонки timestamp")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {sym}: {e}")

    conn.close()
    return all_dfs


def align_dataframes_to_common_index(all_dfs, timestamps, required_cols=None):
    """
    Безопасное выравнивание по общей временной оси.
    Dropna делаем только по действительно нужным колонкам.
    """
    aligned = {}
    ts_index = pd.DatetimeIndex(timestamps)

    for sym, df in all_dfs.items():
        aligned_df = (
            df.set_index("timestamp")
            .sort_index()
            .reindex(ts_index)
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )

        if required_cols is None:
            cols_check = [c for c in aligned_df.columns if c != "timestamp"]
        else:
            cols_check = [c for c in required_cols if c in aligned_df.columns]

        before = len(aligned_df)
        aligned_df = aligned_df.dropna(subset=cols_check)
        after = len(aligned_df)

        if after != len(ts_index):
            print(f"⚠️ {sym}: после reindex/dropna осталось {after} из {len(ts_index)} timestamp")
            return None

        aligned[sym] = aligned_df

    return aligned


def calculate_unrealized_pnl(pos, mark_price):
    entry_price = pos["entry"]
    direction = pos["dir"]
    position_notional = pos["size"]

    if direction == 1:
        raw_pnl = (mark_price - entry_price) / entry_price
    else:
        raw_pnl = (entry_price - mark_price) / entry_price

    return position_notional * raw_pnl


def build_walk_forward_folds(common_timestamps, purge_gap: pd.Timedelta):
    n = len(common_timestamps)
    bpd = bars_per_day_tf()
    train_bars = max(1, int(WF_TRAIN_DAYS * bpd))
    test_bars = max(1, int(WF_TEST_DAYS * bpd))
    step_bars = max(1, int(WF_STEP_DAYS * bpd))
    pb = purge_bars_count(purge_gap)
    need = train_bars + pb + test_bars

    if n < need:
        print(f"❌ Мало общих баров: {n} < нужно {need} (train+purge+test).")
        return []

    folds = []
    fold_start = 0
    while fold_start + need <= n:
        test_start_idx = fold_start + train_bars + pb
        test_end_idx = test_start_idx + test_bars

        folds.append(
            {
                "fold_id": len(folds),
                "fold_start_idx": fold_start,
                "train_bars": train_bars,
                "purge_bars": pb,
                "test_bars": test_bars,
                "train_start_ts": common_timestamps[fold_start],
                "first_test_ts": common_timestamps[test_start_idx],
                "last_test_ts": common_timestamps[test_end_idx - 1],
                "test_start_idx": test_start_idx,
                "test_end_idx": test_end_idx,
            }
        )
        fold_start += step_bars
        if len(folds) >= WF_MAX_FOLDS:
            break

    return folds


def split_train_for_early_stopping(df_train: pd.DataFrame, frac=0.85):
    df_train = df_train.sort_values("timestamp").reset_index(drop=True)
    n = len(df_train)
    if n < 150:
        cut = max(1, n // 2)
    else:
        cut = max(1, int(n * frac))
    if n - cut < 40:
        cut = n - max(40, n // 10)
    cut = max(1, min(cut, n - 40))
    return df_train.iloc[:cut].copy(), df_train.iloc[cut:].copy()


def evaluate_regression_fold(y_true, y_pred, top_frac=0.10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return {
            "spearman": np.nan,
            "mean_true_top_pred": np.nan,
            "uplift_top_pred": np.nan,
        }

    try:
        from scipy.stats import spearmanr
        sp_corr, _ = spearmanr(y_true, y_pred)
    except Exception:
        sp_corr = np.nan

    n = len(y_true)
    k = max(1, int(n * top_frac))
    top_idx = np.argsort(y_pred)[-k:]
    mean_true_top_pred = float(np.mean(y_true[top_idx])) if len(top_idx) else np.nan

    baseline_mean_true = float(np.mean(y_true)) if n > 0 else np.nan
    if baseline_mean_true == 0 or np.isnan(baseline_mean_true):
        uplift_top_pred = np.nan
    else:
        uplift_top_pred = mean_true_top_pred / baseline_mean_true

    return {
        "spearman": sp_corr,
        "mean_true_top_pred": mean_true_top_pred,
        "uplift_top_pred": uplift_top_pred,
    }




def regime_allows_trade(row, side: int) -> bool:
    """
    side: 1 = long, -1 = short
    Используем найденные сильные режимные фичи как gate.
    """
    if not ENABLE_REGIME_FILTER:
        return True

    close = float(row.get("close", np.nan))
    ema50 = float(row.get("EMA_50", np.nan)) if "EMA_50" in row else np.nan
    ema200 = float(row.get("EMA_200", np.nan)) if "EMA_200" in row else np.nan
    hurst = float(row.get("hurst_rs", np.nan)) if "hurst_rs" in row else np.nan
    adx_col = f"ADX_{ADX_LENGTH}"
    adx = float(row.get(adx_col, np.nan)) if adx_col in row else np.nan
    cw = float(row.get("channel_width_20_vs_mean", np.nan)) if "channel_width_20_vs_mean" in row else np.nan

    if np.isnan(close) or np.isnan(ema50) or np.isnan(ema200):
        return True

    if side == 1:
        ok = close > ema50 and ema50 > ema200
    else:
        ok = close < ema50 and ema50 < ema200

    if not np.isnan(hurst):
        ok = ok and (hurst >= REGIME_MIN_HURST)
    if not np.isnan(adx):
        ok = ok and (adx >= REGIME_MIN_ADX)
    if not np.isnan(cw):
        ok = ok and (REGIME_MIN_CHANNEL_WIDTH_RATIO <= cw <= REGIME_MAX_CHANNEL_WIDTH_RATIO)

    return bool(ok)

def train_models_walk_forward_slice(df_train: pd.DataFrame, verbose=False):
    features = list(FEATURE_COLUMNS)
    tr, va = split_train_for_early_stopping(df_train)

    if len(va) < 30 or len(tr) < 50:
        tr, va = split_train_for_early_stopping(df_train, frac=0.75)

    X_tr = tr[features]
    X_va = va[features]

    y_long_tr = tr["Target_Long_Return"]
    y_long_va = va["Target_Long_Return"]

    y_short_tr = tr["Target_Short_Return"]
    y_short_va = va["Target_Short_Return"]

    long_model, long_pred, _ = train_single_regressor(
        X_tr, y_long_tr, X_va, y_long_va, verbose=verbose
    )
    short_model, short_pred, _ = train_single_regressor(
        X_tr, y_short_tr, X_va, y_short_va, verbose=verbose
    )

    long_metrics = evaluate_regression_fold(y_long_va, long_pred, top_frac=0.10)
    short_metrics = evaluate_regression_fold(y_short_va, short_pred, top_frac=0.10)
    long_thr_info = calibrate_side_threshold(y_long_va, long_pred, side_name="LONG")
    short_thr_info = calibrate_side_threshold(y_short_va, short_pred, side_name="SHORT")

    return {
        "long_model": long_model,
        "short_model": short_model,
        "long_metrics": long_metrics,
        "short_metrics": short_metrics,
        "long_thr_info": long_thr_info,
        "short_thr_info": short_thr_info,
    }


def _fmt_metric(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{x:.{nd}f}"


def print_fold_train_metrics(fid, res):
    lm = res["long_metrics"]
    sm = res["short_metrics"]
    lt = res["long_thr_info"]
    st = res["short_thr_info"]
    print(
        f"     LONG Spearman={_fmt_metric(lm['spearman'])} uplift={_fmt_metric(lm['uplift_top_pred'], 2)} top_mean={_fmt_metric(lm['mean_true_top_pred'], 4)} | "
        f"SHORT Spearman={_fmt_metric(sm['spearman'])} uplift={_fmt_metric(sm['uplift_top_pred'], 2)} top_mean={_fmt_metric(sm['mean_true_top_pred'], 4)}"
    )
    print(
        f"     THR  LONG={_fmt_metric(lt['threshold'] * 100, 4)}% top={_fmt_metric(lt['top_frac'] * 100 if pd.notna(lt['top_frac']) else np.nan, 1)}% cnt={lt['count']} | "
        f"SHORT={_fmt_metric(st['threshold'] * 100, 4)}% top={_fmt_metric(st['top_frac'] * 100 if pd.notna(st['top_frac']) else np.nan, 1)}% cnt={st['count']}"
    )


def print_wf_train_summary_table(trained_packs):
    if not trained_packs:
        print("\n⚠️ Нет успешно обученных фолдов.")
        return

    print("\n" + "=" * 120)
    print("СВОДКА ОБУЧЕНИЯ (метрики CatBoost на внутреннем val; OOS ещё не смотрели)")
    print("=" * 120)

    line = (
        f"{'#':>3} {'train_n':>8} "
        f"{'L_Sp':>8} {'L_up':>7} {'L_top':>9} "
        f"{'S_Sp':>8} {'S_up':>7} {'S_top':>9}"
    )
    print(line)
    print("-" * len(line))

    for p in trained_packs:
        r = p["res"]
        lm = r["long_metrics"]
        sm = r["short_metrics"]
        print(
            f"{p['fold_id']:>3} {p['n_train']:>8} "
            f"{_fmt_metric(lm['spearman']):>8} {_fmt_metric(lm['uplift_top_pred'], 2):>7} {_fmt_metric(lm['mean_true_top_pred'], 4):>9} "
            f"{_fmt_metric(sm['spearman']):>8} {_fmt_metric(sm['uplift_top_pred'], 2):>7} {_fmt_metric(sm['mean_true_top_pred'], 4):>9}"
        )

    print("=" * 120)


def train_all_walk_forward_models(df_ml, folds, common_timestamps, purge_gap):
    trained_packs = []
    print("\n" + "=" * 72)
    print("ФАЗА 1 — обучение моделей по всем фолдам walk-forward")
    print("=" * 72)

    for fold in folds:
        fid = fold["fold_id"]
        train_start_ts = fold["train_start_ts"]
        first_test_ts = fold["first_test_ts"]
        train_cutoff = first_test_ts - purge_gap

        train_mask = (df_ml["timestamp"] >= train_start_ts) & (df_ml["timestamp"] <= train_cutoff)
        df_train = df_ml.loc[train_mask].copy()

        if len(df_train) < WF_MIN_TRAIN_ROWS:
            print(f"\n⚠️ Фолд {fid}: train {len(df_train)} < {WF_MIN_TRAIN_ROWS} — пропуск")
            continue

        print(f"\n--- Фолд {fid} | train: {train_start_ts} … {train_cutoff} | rows={len(df_train)} ---")
        print(f"    OOS-период: {fold['first_test_ts']} … {fold['last_test_ts']}")

        try:
            res = train_models_walk_forward_slice(df_train, verbose=False)
        except Exception as e:
            print(f"❌ Фолд {fid}: ошибка обучения: {e}")
            continue

        print_fold_train_metrics(fid, res)
        trained_packs.append(
            {
                "fold_id": fid,
                "long_model": res["long_model"],
                "short_model": res["short_model"],
                "test_ts_slice": common_timestamps[fold["test_start_idx"]:fold["test_end_idx"]],
                "test_start_ts": fold["first_test_ts"],
                "test_end_ts": fold["last_test_ts"],
                "long_thr_info": res["long_thr_info"],
                "short_thr_info": res["short_thr_info"],
                "res": res,
                "n_train": len(df_train),
            }
        )

    print_wf_train_summary_table(trained_packs)
    return trained_packs


def build_continuous_oos_context(trained_packs, all_dfs_bt, common_timestamps):
    """
    Готовим непрерывную OOS-ленту:
    для каждого timestamp знаем, какой пакет моделей активен.

    Важно:
    для OOS-симуляции нам НЕ нужны target-колонки,
    поэтому dropna делаем только по OHLC + feature columns.
    """
    required_oos_cols = ["open", "high", "low", "close"] + list(FEATURE_COLUMNS)

    all_aligned = align_dataframes_to_common_index(
        all_dfs_bt,
        common_timestamps,
        required_cols=required_oos_cols
    )
    if all_aligned is None:
        return None, None

    fold_by_ts = {}
    for pack in trained_packs:
        fid = pack["fold_id"]
        for ts in pack["test_ts_slice"]:
            fold_by_ts[ts] = fid

    valid_ts = [ts for ts in common_timestamps if ts in fold_by_ts]
    if len(valid_ts) < 2:
        return None, None

    # После первого align часть timestamp могла бы отпасть, но здесь уже
    # все df гарантированно имеют одинаковую полную ось.
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


def run_oos_simulation_continuous(trained_packs, all_dfs_bt, feature_names, common_timestamps, initial_balance, verbose_trades):
    """
    Фаза 2: единый непрерывный поток OOS-сделок.
    Модели переключаются по timestamp, но лог не рвется на отдельные simulate по фолдам.
    """
    print("\n" + "=" * 72)
    print("ФАЗА 2 — OOS-симуляция (непрерывный поток сделок, без train)")
    print("=" * 72)

    aligned, ctx = build_continuous_oos_context(trained_packs, all_dfs_bt, common_timestamps)
    if aligned is None or ctx is None:
        print("❌ Не удалось собрать непрерывную OOS-ленту.")
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
        long_model = pack["long_model"]
        short_model = pack["short_model"]

        if current_fold_id != fold_id:
            # Перед переключением печатаем итог предыдущего фолда
            if current_fold_id is not None and current_fold_id in fold_stats:
                prev = fold_stats[current_fold_id]
                wr = (prev["wins"] / prev["trades"] * 100) if prev["trades"] > 0 else 0.0
                print(
                    f"   итог fold {current_fold_id}: "
                    f"trades={prev['trades']} | WR={wr:.1f}% | "
                    f"W/L={prev['wins']}/{prev['losses']} | "
                    f"L/S={prev['longs']}/{prev['shorts']} | "
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
                f"\n↔ SWITCH MODEL PACK -> fold {fold_id} | "
                f"OOS {pack['test_start_ts']} … {pack['test_end_ts']} | "
                f"balance={balance:.2f}$"
            )

        current_equity = balance
        for sym, pos in positions.items():
            if pos is not None:
                current_close = aligned[sym].iloc[i]["close"]
                current_equity += calculate_unrealized_pnl(pos, current_close)

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
                        reason = "❌ SL"
                    elif next_low <= stop_price:
                        exit_price = (next_open if next_open < stop_price else stop_price) * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL"
                    elif next_high >= take_price:
                        exit_price = take_price * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "✅ TP"

                else:
                    stop_price = entry_price * (1 + SL_PCT)
                    take_price = entry_price * (1 - TP_PCT)

                    if next_high >= stop_price and next_low <= take_price:
                        exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL"
                    elif next_high >= stop_price:
                        exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL"
                    elif next_low <= take_price:
                        exit_price = take_price * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "✅ TP"

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
                            f"[{next_ts}] {sym}: {reason} | "
                            f"PnL: {pnl_clean*100:.2f}% | Com: {commission:.2f}$ | Bal: {balance:.2f}"
                        )
                    continue

            if positions[sym] is None:
                current_row = df.iloc[i]
                current_features = df.iloc[[i]][feature_names]
                pred_long = float(long_model.predict(current_features)[0])
                pred_short = float(short_model.predict(current_features)[0])

                long_thr = pack.get("long_thr_info", {}).get("threshold", DEFAULT_LONG_THRESHOLD)
                short_thr = pack.get("short_thr_info", {}).get("threshold", DEFAULT_SHORT_THRESHOLD)

                long_ok = np.isfinite(long_thr) and pred_long >= long_thr and regime_allows_trade(current_row, side=1)
                short_ok = np.isfinite(short_thr) and pred_short >= short_thr and regime_allows_trade(current_row, side=-1)

                signal = 0

                if TRADE_MODE == "long_only":
                    if long_ok:
                        signal = 1
                elif TRADE_MODE == "short_only":
                    if short_ok:
                        signal = -1
                else:
                    if long_ok and short_ok:
                        signal = 1 if pred_long >= pred_short else -1
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
                        chosen_pred = pred_long
                    else:
                        entry_price = next_open * (1 - SLIPPAGE)
                        chosen_pred = pred_short

                    positions[sym] = {
                        "dir": signal,
                        "entry": entry_price,
                        "size": position_notional,
                        "margin": required_margin,
                        "ts_open": next_ts,
                        "pred_long": pred_long,
                        "pred_short": pred_short,
                        "pred_chosen": chosen_pred,
                        "fold_id": fold_id,
                    }

                    dir_str = "LONG" if signal == 1 else "SHORT"
                    if verbose_trades:
                        print(
                            f"[{next_ts}] {sym}: 🚀 OPEN {dir_str} "
                            f"(Pred L: {pred_long*100:.4f}%, Pred S: {pred_short*100:.4f}%, "
                            f"Thr L: {long_thr*100 if np.isfinite(long_thr) else float('nan'):.4f}%, Thr S: {short_thr*100 if np.isfinite(short_thr) else float('nan'):.4f}%) "
                            f"at {entry_price:.2f} | Size: {position_notional:.1f}$"
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
                print(f"[{last_ts}] {sym}: 🔚 FORCE EXIT | PnL: {pnl_clean*100:.2f}% | Bal: {balance:.2f}")

        final_equity = balance
        equity_curve.append(final_equity)
        equity_timestamps.append(last_ts)

        if final_equity > peak_balance:
            peak_balance = final_equity

        current_dd = (peak_balance - final_equity) / peak_balance * 100 if peak_balance > 0 else 0.0
        if current_dd > max_drawdown:
            max_drawdown = current_dd

    for fid, st in fold_stats.items():
        st["end_balance"] = balance if fid == current_fold_id else st.get("end_balance", None)

    if current_fold_id is not None and current_fold_id in fold_stats:
        prev = fold_stats[current_fold_id]
        wr = (prev["wins"] / prev["trades"] * 100) if prev["trades"] > 0 else 0.0
        print(
            f"   итог fold {current_fold_id}: "
            f"trades={prev['trades']} | WR={wr:.1f}% | "
            f"W/L={prev['wins']}/{prev['losses']} | "
            f"L/S={prev['longs']}/{prev['shorts']} | "
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


def backtest():
    print("Walk-forward + OOS: загрузка данных...")
    purge_gap = get_purge_gap()
    print(f"Purge gap: {purge_gap} ({purge_bars_count(purge_gap)} баров)")

    df_ml = load_data_from_db()
    if df_ml.empty:
        print("❌ Нет данных load_data_from_db — запустите etl.py")
        return

    df_ml["timestamp"] = pd.to_datetime(df_ml["timestamp"])
    df_ml = df_ml.sort_values("timestamp").reset_index(drop=True)

    feature_names = list(FEATURE_COLUMNS)
    all_dfs_bt = load_all_data(SYMBOLS, feature_names)
    if not all_dfs_bt:
        print("❌ Нет данных для бэктеста")
        return

    common_timestamps = sorted(list(set.intersection(*(set(df["timestamp"]) for df in all_dfs_bt.values()))))
    if not common_timestamps:
        print("❌ Нет общих timestamp")
        return

    folds = build_walk_forward_folds(common_timestamps, purge_gap)
    if not folds:
        return

    print(f"\n🔁 Запланировано фолдов: {len(folds)} | train≈{WF_TRAIN_DAYS}d, OOS≈{WF_TEST_DAYS}d, шаг≈{WF_STEP_DAYS}d")

    trained_packs = train_all_walk_forward_models(df_ml, folds, common_timestamps, purge_gap)
    if not trained_packs:
        print("❌ Нет обученных фолдов — выход.")
        return

    initial_balance = 100.0
    sim = run_oos_simulation_continuous(
        trained_packs,
        all_dfs_bt,
        feature_names,
        common_timestamps,
        initial_balance,
        verbose_trades=WF_VERBOSE_TRADES,
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
    print("WALK-FORWARD: ИТОГО ПО ВСЕМ ФОЛДАМ (только OOS)")
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
    print(f"\nКонечный баланс: {balance:.2f}$ (старт {initial_balance}$)")
    print(f"Макс. просадка (по фолдам): {max_dd_global:.2f}%")

    if len(equity_all_val) > 1:
        equity_series = pd.Series(equity_all_val, index=pd.to_datetime(equity_all_ts))
        daily_equity = equity_series.resample("D").last().ffill()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            total_days = max(1, (daily_equity.index[-1] - daily_equity.index[0]).days)
            cagr = (daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (365 / total_days) - 1
            sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365))
            downside = daily_returns[daily_returns < 0]
            sortino = (
                (daily_returns.mean() / downside.std() * np.sqrt(365))
                if len(downside) > 1 and downside.std() != 0
                else 0.0
            )
            calmar = cagr / (max_dd_global / 100) if max_dd_global > 0 else 0.0
        else:
            cagr = sharpe = sortino = calmar = 0.0

        if all_trades:
            rets = np.array([t["pnl_abs"] for t in all_trades])
            gp = rets[rets > 0].sum()
            gl = abs(rets[rets < 0].sum())
            pf = gp / gl if gl > 0 else float("inf")
        else:
            pf = 0.0

        print("\n" + "=" * 40)
        print("📊 Метрики (склеенная OOS equity)")
        print("=" * 40)
        print(f"Profit Factor:   {pf:.2f}")
        print(f"Sharpe:          {sharpe:.2f}")
        print(f"Sortino:         {sortino:.2f}")
        print(f"Calmar:          {calmar:.2f}")
        print(f"CAGR:            {cagr*100:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(equity_all_ts, equity_all_val, "b-", label="WF OOS Equity")
        plt.axhline(y=initial_balance, color="gray", linestyle="--")
        plt.title(f"Walk-Forward OOS | trades={total_trades} | DD≈{max_dd_global:.1f}%")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("equity_curve.png", dpi=150)
        plt.show()
        print("\n📈 Сохранено: equity_curve.png")


if __name__ == "__main__":
    backtest()