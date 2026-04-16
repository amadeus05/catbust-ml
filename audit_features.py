#!/usr/bin/env python3
"""
Аудит корреляций: новые фичи vs «золотой» core.
Загружает все таблицы *_features из БД (как train.py), строит corr и опционально
сопоставляет с важностями из CSV (экспорт из лога / train).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from config import DB_PATH
except ImportError:
    DB_PATH = "./data/market_data.db"

# Новые фичи (текущий «расширенный» набор из etl при отключённых micro/vol/vwap-блоках)
NEW_FEATURES = [
    "feat_cum_delta_vol",
    "feat_close_position",
    "feat_persistent_trend",
    "feat_vol_efficiency",
    "feat_ret_1",
    "feat_vol_pressure",
    "feat_ret_accel",
    "feat_risk_adj_return",
    "feat_intra_strength",
    "feat_trend_purity",
]

CORE_FEATURES = [
    "hurst_rs",
    "ema50_dist",
    "ema200_dist",
    "ema_spread_50_200",
    "channel_width_50",
    "Dist_to_Resistance",
    "ret_24",
    "vol_24",
]


def _discover_feature_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_features';"
    )
    return [r[0] for r in cur.fetchall()]


def load_data(
    db_path: Path,
    core: list[str],
    new: list[str],
    max_rows: int | None,
) -> pd.DataFrame:
    """Загрузка последних max_rows строк по объединённым таблицам *_features."""
    cols_needed = ["timestamp"] + [c for c in core + new if c not in ("timestamp",)]

    conn = sqlite3.connect(db_path)
    try:
        tables = _discover_feature_tables(conn)
        if not tables:
            return pd.DataFrame()

        chunks: list[pd.DataFrame] = []
        for table in tables:
            cur = conn.cursor()
            cur.execute(f'PRAGMA table_info("{table}")')
            table_cols = {row[1] for row in cur.fetchall()}
            missing = [c for c in cols_needed if c not in table_cols]
            if missing:
                print(f"⚠️ Таблица {table}: пропуск — нет колонок: {missing[:8]}{'…' if len(missing) > 8 else ''}")
                continue

            qcols = ", ".join(f'"{c}"' for c in cols_needed)
            df = pd.read_sql_query(f"SELECT {qcols} FROM {table}", conn)
            chunks.append(df)

        if not chunks:
            return pd.DataFrame()

        out = pd.concat(chunks, ignore_index=True)
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        out = out.sort_values("timestamp").reset_index(drop=True)
        if max_rows is not None and len(out) > max_rows:
            out = out.iloc[-max_rows:].copy()
        return out.dropna(subset=cols_needed[1:])
    finally:
        conn.close()


def load_importance_csv(path: Path) -> pd.Series:
    """
    CSV с колонками feature, importance (или первые две колонки: имя, значение).
    """
    imp = pd.read_csv(path)
    if imp.shape[1] < 2:
        raise ValueError("В CSV нужны минимум 2 колонки: feature, importance")
    c0, c1 = imp.columns[0], imp.columns[1]
    s = pd.Series(imp[c1].values, index=imp[c0].astype(str).str.strip())
    return s


def run_audit(
    df: pd.DataFrame,
    core: list[str],
    new: list[str],
    importance: pd.Series | None,
    heatmap_path: Path | None,
) -> None:
    core_ok = [c for c in core if c in df.columns]
    new_ok = [c for c in new if c in df.columns]
    missing_core = [c for c in core if c not in df.columns]
    missing_new = [c for c in new if c not in df.columns]
    if missing_core:
        print(f"⚠️ Нет core-колонок в данных: {missing_core}")
    if missing_new:
        print(f"⚠️ Нет new-колонок в данных: {missing_new}")
    if not core_ok or not new_ok:
        print("❌ Недостаточно колонок для корреляционного анализа.")
        return

    all_feats = core_ok + new_ok
    corr_matrix = df[all_feats].corr()

    print("\n🔍 АНАЛИЗ КОРРЕЛЯЦИЙ НОВЫХ ФИЧ СО СТАРЫМИ")
    print("-" * 60)

    redundant_count = 0
    rows_out: list[dict] = []

    for new_f in new_ok:
        correlations = corr_matrix.loc[new_f, core_ok].abs()
        max_corr = float(correlations.max())
        best_match = correlations.idxmax()

        if max_corr > 0.85:
            status = "⚠️ REDUNDANT"
            redundant_count += 1
        elif max_corr < 0.3:
            status = "❓ WEAK_LINK"
        else:
            status = "✅ OK"

        imp_val = np.nan
        if importance is not None and new_f in importance.index:
            imp_val = float(importance.loc[new_f])

        rows_out.append(
            {
                "feature": new_f,
                "max_corr_core": max_corr,
                "best_core": best_match,
                "status": status,
                "importance": imp_val,
            }
        )

        imp_str = f" | imp={imp_val:.4f}" if importance is not None and pd.notna(imp_val) else ""
        print(f"{new_f:25} | Max Corr: {max_corr:.3f} (с {best_match:15}) | {status}{imp_str}")

    print("\n" + "=" * 60)
    if redundant_count > 0:
        print(
            f"💡 ВЫВОД: {redundant_count} из {len(new_ok)} новых фич сильно коррелируют с core (|r|>0.85)."
        )
        print("   Действие: рассмотреть исключение дублей из обучения.")
    else:
        print("💡 ВЫВОД: Прямых «дублей» по порогу 0.85 нет.")
        print("   Если в CatBoost фичи всё равно в noise:")
        print("   1) слабая связь с текущим таргетом / горизонтом;")
        print("   2) высокий шум таргета;")
        print("   3) смотреть распределение таргета или смену разметки (TBM).")

    if importance is not None and rows_out:
        print("\n📊 СВОДКА: importance (из CSV) vs max |corr| с core")
        print("-" * 60)
        for r in sorted(rows_out, key=lambda x: -abs(x.get("importance") or 0) if pd.notna(x.get("importance")) else 0):
            imp = r["importance"]
            imps = f"{imp:.5f}" if pd.notna(imp) else "n/a"
            print(f"  {r['feature']:25}  imp={imps:12}  max|corr|={r['max_corr_core']:.3f}  {r['status']}")

    if heatmap_path is not None:
        try:
            import seaborn as sns
        except ImportError:
            sns = None
        fig_w = max(10, len(all_feats) * 0.35)
        fig_h = max(8, len(all_feats) * 0.35)
        plt.figure(figsize=(fig_w, fig_h))
        if sns is not None:
            sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        else:
            plt.imshow(
                corr_matrix.values,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                aspect="auto",
            )
            plt.colorbar()
            plt.xticks(
                range(len(corr_matrix.columns)),
                corr_matrix.columns,
                rotation=90,
                fontsize=7,
            )
            plt.yticks(range(len(corr_matrix.index)), corr_matrix.index, fontsize=7)
        plt.title("Correlation: core + new features")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"\n🖼 Тепловая карта сохранена: {heatmap_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит корреляций фичей")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(DB_PATH),
        help="Путь к SQLite (по умолчанию из config.DB_PATH)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="Взять последние N строк после объединения всех символов (по времени)",
    )
    parser.add_argument(
        "--importance-csv",
        type=Path,
        default=None,
        help="CSV: колонка feature + колонка importance (из лога / экспорта train)",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        default=None,
        help="Сохранить heatmap корреляций в файл (PNG)",
    )
    args = parser.parse_args()

    imp_series: pd.Series | None = None
    if args.importance_csv is not None:
        p = args.importance_csv.resolve()
        if not args.importance_csv.is_file():
            print(f"❌ Файл важностей не найден: {p}")
            print("   Ожидаются две колонки: feature, importance (имена могут быть любыми — берутся первые две).")
            print("   Пример в корне проекта: importances.csv (заглушки — замените на значения из CatBoost).")
            return
        imp_series = load_importance_csv(args.importance_csv)
        print(f"📥 Загружены важности для {len(imp_series)} признаков из {args.importance_csv}")

    print("🔄 Загрузка данных...")
    if not args.db.is_file():
        print(f"❌ БД не найдена: {args.db}")
        return

    df = load_data(args.db, CORE_FEATURES, NEW_FEATURES, args.max_rows)
    if df.empty:
        print("❌ Нет данных. Проверьте путь к БД и наличие таблиц *_features с нужными колонками.")
        return

    print(f"✅ Строк для анализа: {len(df)} | колонок: {len(df.columns)}")
    run_audit(df, CORE_FEATURES, NEW_FEATURES, imp_series, args.heatmap)


if __name__ == "__main__":
    main()
