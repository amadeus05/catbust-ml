from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TradeSimulationResult:
    direction: int
    entry_price: float
    exit_price: float
    return_pct: float
    exit_bar_offset: int
    reason: str


def simulate_trade_return(
    direction: int,
    entry_open: float,
    opens,
    highs,
    lows,
    closes,
    tp_pct: float,
    sl_pct: float,
    slippage: float,
    taker_com: float,
    target_mode: str = "pct",
    entry_atr: float | None = None,
    tp_atr_mult: float | None = None,
    sl_atr_mult: float | None = None,
) -> TradeSimulationResult:
    """
    Simulate one fixed-rule trade from the entry bar through the supplied window.

    The first element in opens/highs/lows/closes is the entry bar. If TP and SL
    are both touched inside one bar, SL wins. The last supplied close is the
    timeout exit.
    """
    direction = int(direction)
    if direction not in (1, -1):
        raise ValueError("direction must be 1 for long or -1 for short")

    opens = np.asarray(opens, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)

    if len(opens) == 0:
        raise ValueError("trade window must contain at least one bar")

    if not (len(opens) == len(highs) == len(lows) == len(closes)):
        raise ValueError("OHLC windows must have the same length")

    if direction == 1:
        entry_price = float(entry_open) * (1.0 + slippage)
    else:
        entry_price = float(entry_open) * (1.0 - slippage)

    target_mode = str(target_mode).lower()
    if target_mode == "pct":
        tp_distance = entry_price * float(tp_pct)
        sl_distance = entry_price * float(sl_pct)
    elif target_mode == "atr":
        if entry_atr is None or tp_atr_mult is None or sl_atr_mult is None:
            raise ValueError("entry_atr, tp_atr_mult, and sl_atr_mult are required for atr mode")
        if not np.isfinite(entry_atr) or entry_atr <= 0:
            raise ValueError("entry_atr must be a positive finite value")

        tp_distance = float(entry_atr) * float(tp_atr_mult)
        sl_distance = float(entry_atr) * float(sl_atr_mult)
    else:
        raise ValueError("target_mode must be 'pct' or 'atr'")

    if direction == 1:
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance

        for k in range(len(opens)):
            bar_open = opens[k]
            hit_sl = lows[k] <= sl_price
            hit_tp = highs[k] >= tp_price

            if hit_sl:
                exit_price = (bar_open if bar_open < sl_price else sl_price) * (1.0 - slippage)
                raw_ret = (exit_price - entry_price) / entry_price
                return TradeSimulationResult(
                    direction=direction,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    return_pct=float(raw_ret - 2.0 * taker_com),
                    exit_bar_offset=k,
                    reason="SL",
                )

            if hit_tp:
                exit_price = tp_price * (1.0 - slippage)
                raw_ret = (exit_price - entry_price) / entry_price
                return TradeSimulationResult(
                    direction=direction,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    return_pct=float(raw_ret - 2.0 * taker_com),
                    exit_bar_offset=k,
                    reason="TP",
                )

        exit_price = closes[-1] * (1.0 - slippage)
        raw_ret = (exit_price - entry_price) / entry_price
        return TradeSimulationResult(
            direction=direction,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            return_pct=float(raw_ret - 2.0 * taker_com),
            exit_bar_offset=len(opens) - 1,
            reason="TIMEOUT",
        )

    tp_price = entry_price - tp_distance
    sl_price = entry_price + sl_distance

    for k in range(len(opens)):
        bar_open = opens[k]
        hit_sl = highs[k] >= sl_price
        hit_tp = lows[k] <= tp_price

        if hit_sl:
            exit_price = (bar_open if bar_open > sl_price else sl_price) * (1.0 + slippage)
            raw_ret = (entry_price - exit_price) / entry_price
            return TradeSimulationResult(
                direction=direction,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                return_pct=float(raw_ret - 2.0 * taker_com),
                exit_bar_offset=k,
                reason="SL",
            )

        if hit_tp:
            exit_price = tp_price * (1.0 + slippage)
            raw_ret = (entry_price - exit_price) / entry_price
            return TradeSimulationResult(
                direction=direction,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                return_pct=float(raw_ret - 2.0 * taker_com),
                exit_bar_offset=k,
                reason="TP",
            )

    exit_price = closes[-1] * (1.0 + slippage)
    raw_ret = (entry_price - exit_price) / entry_price
    return TradeSimulationResult(
        direction=direction,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        return_pct=float(raw_ret - 2.0 * taker_com),
        exit_bar_offset=len(opens) - 1,
        reason="TIMEOUT",
    )
