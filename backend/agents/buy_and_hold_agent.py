class BuyAndHoldAgent:
    """
    Buys once at the start, then holds.
    """

    def __init__(self):
        self.has_bought = False

    def act(self, state):
        if not self.has_bought:
            self.has_bought = True
            return 1  # BUY
        return 0  # HOLD
