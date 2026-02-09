import numpy as np


def compute_returns(values):
    returns = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((values[i] - prev) / prev)
    return returns


def max_drawdown(values):
    peak = values[0] if values else 0.0
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def volatility(returns):
    if not returns:
        return 0.0
    return float(np.std(returns))


def sharpe_ratio(returns):
    if not returns:
        return 0.0
    vol = volatility(returns)
    if vol == 0.0:
        return 0.0
    return float(np.mean(returns) / vol * np.sqrt(len(returns)))


def turnover(executed_trades, total_steps):
    if total_steps <= 0:
        return 0.0
    return executed_trades / total_steps


METRICS = [
    "final_value",
    "total_reward",
    "max_drawdown",
    "volatility",
    "sharpe",
    "turnover",
    "action_hold_ratio",
    "action_buy_ratio",
    "action_sell_ratio",
    "executed_trade_ratio",
]
