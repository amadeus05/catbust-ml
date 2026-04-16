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
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)

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

    entry_price = float(entry_open) * (1.0 - slippage)
    tp_price = entry_price * (1.0 - tp_pct)
    sl_price = entry_price * (1.0 + sl_pct)

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
