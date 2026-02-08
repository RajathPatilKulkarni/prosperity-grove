def bull_market(length=20, start=100, step=2):
    return [start + i * step for i in range(length)]


def bear_market(length=20, start=120, step=2):
    return [start - i * step for i in range(length)]


def sideways_market(length=20, price=100):
    return [price for _ in range(length)]


def volatile_market(length=20, base=100, amplitude=5):
    prices = []
    for i in range(length):
        if i % 2 == 0:
            prices.append(base + amplitude)
        else:
            prices.append(base - amplitude)
    return prices


BASE_SCENARIOS = {
    "bull": bull_market,
    "bear": bear_market,
    "sideways": sideways_market,
    "volatile": volatile_market,
}


def regime_schedule(regimes, length=20):
    """
    Build a regime-shift price series by concatenating base regimes.
    regimes: list like ["bull", "volatile", "bear"]
    length: per-regime length
    """
    series = []
    for name in regimes:
        if name not in BASE_SCENARIOS:
            raise ValueError(f"Unknown regime: {name}")
        series.extend(BASE_SCENARIOS[name](length=length))
    return series


def regime_shift_short():
    return regime_schedule(["bull", "volatile", "bear"], length=15)


def regime_shift_long():
    return regime_schedule(["bull", "sideways", "volatile", "bear"], length=25)


SCENARIOS = {
    **BASE_SCENARIOS,
    "regime_shift_short": regime_shift_short,
    "regime_shift_long": regime_shift_long,
}
