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


SCENARIOS = {
    "bull": bull_market,
    "bear": bear_market,
    "sideways": sideways_market,
    "volatile": volatile_market,
}
