import sqlite3
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from config import *

# === ФИЧИ ===
# Слабые: WEAK_FEATURES_DISABLED. Наборы v2/v3, FEATURE_PROFILE — в config.py (v3: beta/corr vs BTC, mkt_corr BTC↔ETH)

TF_TO_HOURS = {"1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}

# --- Calibration defaults (safe fallbacks if config.py does not define them) ---
CALIBRATION_CANDIDATE_TOP_FRACS = tuple(globals().get("CALIBRATION_CANDIDATE_TOP_FRACS", (0.01, 0.02, 0.03, 0.05, 0.10)))
CALIBRATION_MIN_COUNT = int(globals().get("CALIBRATION_MIN_COUNT", 25))
CALIBRATION_MIN_MEAN_TRUE = float(globals().get("CALIBRATION_MIN_MEAN_TRUE", 0.0))
CALIBRATION_DEFAULT_LONG_THRESHOLD = float(globals().get("CALIBRATION_DEFAULT_LONG_THRESHOLD", 0.0010))
CALIBRATION_DEFAULT_SHORT_THRESHOLD = float(globals().get("CALIBRATION_DEFAULT_SHORT_THRESHOLD", 0.0010))



def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    all_data = []
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_features';")
    tables = cursor.fetchall()

    for table_name in tables:
        table = table_name[0]
        df = pd.read_sql(f"SELECT * FROM {table}", conn)

        raw_name = table.replace("_features", "")
        df["symbol"] = raw_name.replace("_", "/", 1) if "_" in raw_name else raw_name

        needed_cols = ["Target_Long_Return", "Target_Short_Return"]
        if not all(col in df.columns for col in needed_cols):
            print(f"⚠️ Пропуск {table}: нет новых таргетов {needed_cols}")
            continue

        df = df.dropna(
            subset=["timestamp", *FEATURE_COLUMNS, "Target_Long_Return", "Target_Short_Return"]
        ).copy()

        all_data.append(df)

    conn.close()

    if not all_data:
        return pd.DataFrame()

    df_full = pd.concat(all_data, ignore_index=True)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
    df_full = df_full.sort_values("timestamp").reset_index(drop=True)
    return df_full


def get_purge_gap():
    bar_hours = TF_TO_HOURS.get(TIMEFRAME, 1)
    return pd.Timedelta(hours=bar_hours * HORIZON)


def bars_per_day_tf():
    h = TF_TO_HOURS.get(TIMEFRAME, 1)
    return max(1, int(round(24 / h)))


def purge_bars_count(purge_gap: pd.Timedelta) -> int:
    bar_h = TF_TO_HOURS.get(TIMEFRAME, 1)
    hours = purge_gap.total_seconds() / 3600
    return max(1, int(np.ceil(hours / bar_h)))


def build_walk_forward_folds_from_df(df_full, purge_gap):
    """
    Скользящее окно:
      [train] -> [purge] -> [OOS/val]
    Строим по уникальным timestamp общей ML-таблицы.
    """
    common_timestamps = sorted(df_full["timestamp"].drop_duplicates().tolist())
    n = len(common_timestamps)

    bpd = bars_per_day_tf()
    train_bars = max(1, int(WF_TRAIN_DAYS * bpd))
    test_bars = max(1, int(WF_TEST_DAYS * bpd))
    step_bars = max(1, int(WF_STEP_DAYS * bpd))
    pb = purge_bars_count(purge_gap)
    need = train_bars + pb + test_bars

    if n < need:
        print(f"❌ Мало timestamp для walk-forward: {n} < нужно {need}")
        return []

    folds = []
    fold_start = 0

    while fold_start + need <= n:
        train_start_ts = common_timestamps[fold_start]
        first_test_ts = common_timestamps[fold_start + train_bars + pb]
        last_test_ts = common_timestamps[fold_start + train_bars + pb + test_bars - 1]

        folds.append(
            {
                "fold_id": len(folds),
                "train_start_ts": train_start_ts,
                "first_test_ts": first_test_ts,
                "last_test_ts": last_test_ts,
            }
        )

        fold_start += step_bars
        if len(folds) >= WF_MAX_FOLDS:
            break

    return folds


def calibrate_side_threshold(y_true, y_pred, side_name="LONG"):
    """
    Подбираем threshold по внутреннему validation fold.
    Ищем top-frac, где средний реальный return > 0 при достаточном числе наблюдений.
    Если такого окна нет — возвращаем безопасный fallback threshold.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_pred)
    if n == 0:
        return {
            "side": side_name,
            "threshold": float("inf"),
            "top_frac": np.nan,
            "count": 0,
            "mean_true": np.nan,
            "hitrate": np.nan,
            "score": -np.inf,
            "enabled": False,
        }

    records = []
    for frac in CALIBRATION_CANDIDATE_TOP_FRACS:
        k = max(1, int(n * frac))
        idx = np.argsort(y_pred)[-k:]
        if len(idx) == 0:
            continue

        threshold = float(np.min(y_pred[idx]))
        selected = y_true[idx]
        count = int(len(selected))
        mean_true = float(np.mean(selected)) if count else np.nan
        hitrate = float(np.mean(selected > 0)) if count else np.nan
        score = mean_true * np.log1p(count)

        records.append({
            "side": side_name,
            "threshold": threshold,
            "top_frac": float(frac),
            "count": count,
            "mean_true": mean_true,
            "hitrate": hitrate,
            "score": score,
            "enabled": bool(count >= CALIBRATION_MIN_COUNT and mean_true > CALIBRATION_MIN_MEAN_TRUE),
        })

    valid = [r for r in records if r["enabled"]]
    if valid:
        return max(valid, key=lambda r: (r["score"], r["mean_true"], r["count"]))

    fallback = CALIBRATION_DEFAULT_LONG_THRESHOLD if side_name.upper() == "LONG" else CALIBRATION_DEFAULT_SHORT_THRESHOLD
    return {
        "side": side_name,
        "threshold": float(fallback),
        "top_frac": np.nan,
        "count": 0,
        "mean_true": np.nan,
        "hitrate": np.nan,
        "score": -np.inf,
        "enabled": False,
    }


def evaluate_regression(y_true, y_pred, name="TARGET", top_frac=0.10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)

    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan

    try:
        sp_corr, sp_pvalue = spearmanr(y_true, y_pred)
    except Exception:
        sp_corr, sp_pvalue = np.nan, np.nan

    n = len(y_true)
    k = max(1, int(n * top_frac))
    top_idx = np.argsort(y_pred)[-k:]

    mean_true_top_pred = float(np.mean(y_true[top_idx])) if len(top_idx) else np.nan
    hitrate_top_pred = float(np.mean(y_true[top_idx] > 0)) if len(top_idx) else np.nan

    baseline_mean_true = float(np.mean(y_true)) if n > 0 else np.nan
    baseline_hitrate = float(np.mean(y_true > 0)) if n > 0 else np.nan

    if baseline_mean_true == 0 or np.isnan(baseline_mean_true):
        uplift_top_pred = np.nan
    else:
        uplift_top_pred = mean_true_top_pred / baseline_mean_true

    pos_mask = y_pred > 0
    pos_count = int(np.sum(pos_mask))
    mean_true_pred_pos = float(np.mean(y_true[pos_mask])) if pos_count > 0 else np.nan
    hitrate_pred_pos = float(np.mean(y_true[pos_mask] > 0)) if pos_count > 0 else np.nan

    return {
        "name": name,
        "mae": mae,
        "r2": r2,
        "spearman": sp_corr,
        "spearman_pvalue": sp_pvalue,
        "top_frac": top_frac,
        "mean_true_top_pred": mean_true_top_pred,
        "hitrate_top_pred": hitrate_top_pred,
        "baseline_mean_true": baseline_mean_true,
        "baseline_hitrate": baseline_hitrate,
        "uplift_top_pred": uplift_top_pred,
        "pred_pos_count": pos_count,
        "mean_true_pred_pos": mean_true_pred_pos,
        "hitrate_pred_pos": hitrate_pred_pos,
    }


def train_single_regressor(X_train, y_train, X_test, y_test, verbose=False):
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="MAE",
        early_stopping_rounds=200,
        verbose=100 if verbose else 0,
        allow_writing_files=False,
        random_seed=42,
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    preds = model.predict(X_test)
    metrics = evaluate_regression(y_true=y_test, y_pred=preds, top_frac=0.10)

    try:
        importances = model.get_feature_importance()
    except Exception:
        importances = np.zeros(X_train.shape[1], dtype=float)

    return model, preds, metrics, np.asarray(importances, dtype=float)


def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{x:.{nd}f}"


def print_fold_header(fold_id, train_start_ts, train_cutoff, first_test_ts, last_test_ts, n_train, n_test):
    print(f"\n--- Фолд {fold_id} | train: {train_start_ts} … {train_cutoff} | rows={n_train} ---")
    print(f"    OOS/val:      {first_test_ts} … {last_test_ts} | rows={n_test}")


def print_fold_metrics_block(long_metrics, short_metrics, long_thr_info, short_thr_info):
    print(
        "     LONG  "
        f"Sp={_fmt(long_metrics['spearman'], 3)} | "
        f"top_mean={_fmt(long_metrics['mean_true_top_pred'], 4)} | "
        f"pred>0={long_metrics['pred_pos_count']} | "
        f"mean_true(pred>0)={_fmt(long_metrics['mean_true_pred_pos'], 4)}"
    )
    print(
        "     SHORT "
        f"Sp={_fmt(short_metrics['spearman'], 3)} | "
        f"top_mean={_fmt(short_metrics['mean_true_top_pred'], 4)} | "
        f"pred>0={short_metrics['pred_pos_count']} | "
        f"mean_true(pred>0)={_fmt(short_metrics['mean_true_pred_pos'], 4)}"
    )
    print(
        "     THR   "
        f"LONG thr={_fmt(long_thr_info['threshold'] * 100, 4)}% | top={_fmt(long_thr_info['top_frac'] * 100 if pd.notna(long_thr_info['top_frac']) else np.nan, 1)}% | cnt={int(long_thr_info['count'])} | "
        f"SHORT thr={_fmt(short_thr_info['threshold'] * 100, 4)}% | top={_fmt(short_thr_info['top_frac'] * 100 if pd.notna(short_thr_info['top_frac']) else np.nan, 1)}% | cnt={int(short_thr_info['count'])}"
    )


def summarize_feature_stability(feature_names, long_imps_list, short_imps_list):
    """
    Сводка по стабильности фич.
    """
    long_arr = np.vstack(long_imps_list) if long_imps_list else np.zeros((0, len(feature_names)))
    short_arr = np.vstack(short_imps_list) if short_imps_list else np.zeros((0, len(feature_names)))

    rows = []
    for j, feat in enumerate(feature_names):
        long_col = long_arr[:, j] if len(long_arr) else np.array([])
        short_col = short_arr[:, j] if len(short_arr) else np.array([])

        long_mean = float(np.mean(long_col)) if len(long_col) else np.nan
        long_med = float(np.median(long_col)) if len(long_col) else np.nan
        long_std = float(np.std(long_col)) if len(long_col) else np.nan

        short_mean = float(np.mean(short_col)) if len(short_col) else np.nan
        short_med = float(np.median(short_col)) if len(short_col) else np.nan
        short_std = float(np.std(short_col)) if len(short_col) else np.nan

        rows.append(
            {
                "feature": feat,
                "long_mean_imp": long_mean,
                "long_median_imp": long_med,
                "long_std_imp": long_std,
                "short_mean_imp": short_mean,
                "short_median_imp": short_med,
                "short_std_imp": short_std,
            }
        )

    df = pd.DataFrame(rows)

    # top-5/top-10 frequency
    if len(long_arr):
        top5_long = {f: 0 for f in feature_names}
        top10_long = {f: 0 for f in feature_names}
        for row in long_arr:
            order = np.argsort(row)[::-1]
            for idx in order[:5]:
                top5_long[feature_names[idx]] += 1
            for idx in order[:10]:
                top10_long[feature_names[idx]] += 1
        df["long_top5_count"] = df["feature"].map(top5_long)
        df["long_top10_count"] = df["feature"].map(top10_long)
    else:
        df["long_top5_count"] = 0
        df["long_top10_count"] = 0

    if len(short_arr):
        top5_short = {f: 0 for f in feature_names}
        top10_short = {f: 0 for f in feature_names}
        for row in short_arr:
            order = np.argsort(row)[::-1]
            for idx in order[:5]:
                top5_short[feature_names[idx]] += 1
            for idx in order[:10]:
                top10_short[feature_names[idx]] += 1
        df["short_top5_count"] = df["feature"].map(top5_short)
        df["short_top10_count"] = df["feature"].map(top10_short)
    else:
        df["short_top5_count"] = 0
        df["short_top10_count"] = 0

    # combined stability score
    df["combined_mean_imp"] = (
        df["long_mean_imp"].fillna(0.0) + df["short_mean_imp"].fillna(0.0)
    ) / 2.0

    df["combined_top5_count"] = df["long_top5_count"] + df["short_top5_count"]
    df["combined_top10_count"] = df["long_top10_count"] + df["short_top10_count"]

    max_top5 = max(1, int(df["combined_top5_count"].max()))
    max_imp = max(1e-9, float(df["combined_mean_imp"].max()))

    df["stability_score"] = (
        0.6 * (df["combined_top5_count"] / max_top5) +
        0.4 * (df["combined_mean_imp"] / max_imp)
    )

    def verdict(row):
        if row["combined_top5_count"] >= max(6, int(WF_MAX_FOLDS * 0.35)) and row["combined_mean_imp"] > 0:
            return "stable_useful"
        if row["combined_top10_count"] >= max(6, int(WF_MAX_FOLDS * 0.35)):
            return "conditional"
        return "likely_noise"

    df["verdict"] = df.apply(verdict, axis=1)

    df = df.sort_values(
        ["stability_score", "combined_mean_imp", "combined_top5_count"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df


def print_fold_summary_table(df_folds):
    if df_folds.empty:
        print("\n⚠️ Нет успешно обученных фолдов.")
        return

    print("\n" + "=" * 132)
    print("СВОДКА WALK-FORWARD TRAIN/VAL ПО ФОЛДАМ")
    print("=" * 132)

    line = (
        f"{'#':>3} {'train_n':>8} {'test_n':>8} "
        f"{'L_Sp':>8} {'L_top':>9} {'L_>0':>7} {'L_mean+':>9} "
        f"{'S_Sp':>8} {'S_top':>9} {'S_>0':>7} {'S_mean+':>9}"
    )
    print(line)
    print("-" * len(line))

    for _, r in df_folds.iterrows():
        print(
            f"{int(r['fold_id']):>3} "
            f"{int(r['train_n']):>8} "
            f"{int(r['test_n']):>8} "
            f"{_fmt(r['long_spearman'], 3):>8} "
            f"{_fmt(r['long_top_mean'], 4):>9} "
            f"{int(r['long_pred_pos_count']):>7} "
            f"{_fmt(r['long_mean_true_pred_pos'], 4):>9} "
            f"{_fmt(r['short_spearman'], 3):>8} "
            f"{_fmt(r['short_top_mean'], 4):>9} "
            f"{int(r['short_pred_pos_count']):>7} "
            f"{_fmt(r['short_mean_true_pred_pos'], 4):>9}"
        )

    print("=" * len(line))

    print("\nИТОГО ПО ФОЛДАМ:")
    print(
        f"LONG  avg Spearman={_fmt(df_folds['long_spearman'].mean(), 4)} | "
        f"avg top_mean={_fmt(df_folds['long_top_mean'].mean(), 4)} | "
        f"avg mean_true(pred>0)={_fmt(df_folds['long_mean_true_pred_pos'].mean(), 4)}"
    )
    print(
        f"SHORT avg Spearman={_fmt(df_folds['short_spearman'].mean(), 4)} | "
        f"avg top_mean={_fmt(df_folds['short_top_mean'].mean(), 4)} | "
        f"avg mean_true(pred>0)={_fmt(df_folds['short_mean_true_pred_pos'].mean(), 4)}"
    )


def print_feature_summary(df_feat):
    if df_feat.empty:
        print("\n⚠️ Нет данных по важности фич.")
        return

    print("\n" + "=" * 152)
    print("СВОДКА ПО ФИЧАМ: СТАБИЛЬНОСТЬ ИСПОЛЬЗОВАНИЯ ПО ВСЕМ ФОЛДАМ")
    print("=" * 152)

    line = (
        f"{'feature':<22} "
        f"{'L_mean':>9} {'L_med':>9} {'L_top5':>7} {'L_top10':>8} "
        f"{'S_mean':>9} {'S_med':>9} {'S_top5':>7} {'S_top10':>8} "
        f"{'stab':>7} {'verdict':>14}"
    )
    print(line)
    print("-" * len(line))

    for _, r in df_feat.iterrows():
        print(
            f"{r['feature']:<22} "
            f"{_fmt(r['long_mean_imp'], 3):>9} "
            f"{_fmt(r['long_median_imp'], 3):>9} "
            f"{int(r['long_top5_count']):>7} "
            f"{int(r['long_top10_count']):>8} "
            f"{_fmt(r['short_mean_imp'], 3):>9} "
            f"{_fmt(r['short_median_imp'], 3):>9} "
            f"{int(r['short_top5_count']):>7} "
            f"{int(r['short_top10_count']):>8} "
            f"{_fmt(r['stability_score'], 3):>7} "
            f"{r['verdict']:>14}"
        )

    print("=" * len(line))

    stable = df_feat[df_feat["verdict"] == "stable_useful"]["feature"].tolist()
    conditional = df_feat[df_feat["verdict"] == "conditional"]["feature"].tolist()
    noisy = df_feat[df_feat["verdict"] == "likely_noise"]["feature"].tolist()

    print("\nВЕРДИКТ ПО ФИЧАМ:")
    print(f"Stable useful : {', '.join(stable) if stable else '—'}")
    print(f"Conditional   : {', '.join(conditional) if conditional else '—'}")
    print(f"Likely noise  : {', '.join(noisy) if noisy else '—'}")


def train_walk_forward():
    df_full = load_data_from_db()
    if df_full.empty:
        print("❌ Нет данных для обучения. Сначала запусти etl.py")
        return

    missing_feat = [c for c in FEATURE_COLUMNS if c not in df_full.columns]
    if missing_feat:
        print(f"❌ В данных нет фич: {missing_feat}")
        return

    purge_gap = get_purge_gap()
    folds = build_walk_forward_folds_from_df(df_full, purge_gap)
    if not folds:
        return

    print("\n⚙️ РЕЖИМ WALK-FORWARD FEATURE RESEARCH")
    print(f"Полный диапазон: {df_full['timestamp'].min()} -> {df_full['timestamp'].max()}")
    print(f"Purge gap:       {purge_gap}")
    print(f"Фичей:           {len(FEATURE_COLUMNS)}")
    print(f"Фолдов:          {len(folds)}")
    print(f"Train≈{WF_TRAIN_DAYS}d | OOS/val≈{WF_TEST_DAYS}d | step≈{WF_STEP_DAYS}d")

    features = list(FEATURE_COLUMNS)

    fold_rows = []
    long_imps = []
    short_imps = []

    best_fold_models = None
    best_fold_score = -np.inf
    best_fold_id = None

    print("\n" + "=" * 84)
    print("ФАЗА 1 — walk-forward обучение и сбор метрик/важностей")
    print("=" * 84)

    for fold in folds:
        fid = fold["fold_id"]
        train_start_ts = fold["train_start_ts"]
        first_test_ts = fold["first_test_ts"]
        last_test_ts = fold["last_test_ts"]
        train_cutoff = first_test_ts - purge_gap

        train_mask = (df_full["timestamp"] >= train_start_ts) & (df_full["timestamp"] <= train_cutoff)
        test_mask = (df_full["timestamp"] >= first_test_ts) & (df_full["timestamp"] <= last_test_ts)

        df_train = df_full.loc[train_mask].copy().reset_index(drop=True)
        df_test = df_full.loc[test_mask].copy().reset_index(drop=True)

        if len(df_train) < WF_MIN_TRAIN_ROWS or len(df_test) < 100:
            print(f"\n⚠️ Фолд {fid}: train={len(df_train)}, test={len(df_test)} — пропуск")
            continue

        print_fold_header(
            fid, train_start_ts, train_cutoff, first_test_ts, last_test_ts, len(df_train), len(df_test)
        )

        X_train = df_train[features]
        X_test = df_test[features]

        y_long_train = df_train["Target_Long_Return"]
        y_long_test = df_test["Target_Long_Return"]

        y_short_train = df_train["Target_Short_Return"]
        y_short_test = df_test["Target_Short_Return"]

        try:
            long_model, long_preds, long_metrics, long_imp = train_single_regressor(
                X_train, y_long_train, X_test, y_long_test, verbose=False
            )
            short_model, short_preds, short_metrics, short_imp = train_single_regressor(
                X_train, y_short_train, X_test, y_short_test, verbose=False
            )
        except Exception as e:
            print(f"❌ Фолд {fid}: ошибка обучения: {e}")
            continue

        long_thr_info = calibrate_side_threshold(y_long_test.to_numpy(), long_preds, side_name="LONG")
        short_thr_info = calibrate_side_threshold(y_short_test.to_numpy(), short_preds, side_name="SHORT")

        print_fold_metrics_block(long_metrics, short_metrics, long_thr_info, short_thr_info)

        fold_rows.append(
            {
                "fold_id": fid,
                "train_n": len(df_train),
                "test_n": len(df_test),
                "train_start_ts": train_start_ts,
                "train_end_ts": train_cutoff,
                "test_start_ts": first_test_ts,
                "test_end_ts": last_test_ts,
                "long_spearman": long_metrics["spearman"],
                "long_top_mean": long_metrics["mean_true_top_pred"],
                "long_pred_pos_count": long_metrics["pred_pos_count"],
                "long_mean_true_pred_pos": long_metrics["mean_true_pred_pos"],
                "short_spearman": short_metrics["spearman"],
                "short_top_mean": short_metrics["mean_true_top_pred"],
                "short_pred_pos_count": short_metrics["pred_pos_count"],
                "short_mean_true_pred_pos": short_metrics["mean_true_pred_pos"],
                "long_threshold": long_thr_info["threshold"],
                "short_threshold": short_thr_info["threshold"],
                "long_calib_top_frac": long_thr_info["top_frac"],
                "short_calib_top_frac": short_thr_info["top_frac"],
                "long_calib_count": long_thr_info["count"],
                "short_calib_count": short_thr_info["count"],
                "long_calib_mean_true": long_thr_info["mean_true"],
                "short_calib_mean_true": short_thr_info["mean_true"],
                "long_calib_enabled": long_thr_info["enabled"],
                "short_calib_enabled": short_thr_info["enabled"],
            }
        )

        long_imps.append(long_imp)
        short_imps.append(short_imp)

        # === Лучший фолд: Spearman + mean_true_top_pred (топ-10% сигналов), симметрично L/S ===
        long_sp = np.nan_to_num(long_metrics["spearman"], nan=-1.0)
        short_sp = np.nan_to_num(short_metrics["spearman"], nan=-1.0)
        long_top = np.nan_to_num(long_metrics["mean_true_top_pred"], nan=-0.005)
        short_top = np.nan_to_num(short_metrics["mean_true_top_pred"], nan=-0.005)

        # Ограничиваем экстремумы top_mean, чтобы один удачный/неудачный фолд не ломал score
        long_top = np.clip(long_top, -0.01, 0.01)
        short_top = np.clip(short_top, -0.01, 0.01)

        # Веса: spearman по 0.25; top_mean масштабируем коэфф. 25 (стабильнее ручного *60 на среднем)
        fold_score = (
            0.25 * long_sp
            + 0.25 * short_sp
            + 25.0 * long_top
            + 25.0 * short_top
        )

        if fold_score > best_fold_score:
            best_fold_score = fold_score
            best_fold_id = fid
            best_fold_models = (long_model, short_model)

    if not fold_rows:
        print("❌ Нет успешно обученных фолдов.")
        return

    df_folds = pd.DataFrame(fold_rows)
    df_feat = summarize_feature_stability(features, long_imps, short_imps)

    print_fold_summary_table(df_folds)
    print_feature_summary(df_feat)

    # Сохраняем лучшие модели по внутреннему fold score
    if best_fold_models is not None:
        long_model, short_model = best_fold_models
        long_model.save_model(str(MODELS_DIR / "long_return_model.cbm"))
        short_model.save_model(str(MODELS_DIR / "short_return_model.cbm"))

        with open(MODELS_DIR / "features.pkl", "wb") as f:
            pickle.dump(features, f)

        df_folds.to_csv(MODELS_DIR / "wf_fold_metrics.csv", index=False)
        df_feat.to_csv(MODELS_DIR / "wf_feature_stability.csv", index=False)

        print("\n✅ Сохранено:")
        print(f" - long_return_model.cbm")
        print(f" - short_return_model.cbm")
        print(f" - features.pkl")
        print(f" - wf_fold_metrics.csv")
        print(f" - wf_feature_stability.csv")
        print(f"\nЛучшая сохраненная пара моделей: fold {best_fold_id}")
    else:
        print("\n⚠️ Модели не были сохранены: не удалось выбрать лучший фолд.")


if __name__ == "__main__":
    train_walk_forward()