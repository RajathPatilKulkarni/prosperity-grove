class RuleBasedAgent:
    """
    Simple momentum-based heuristic agent.
    Buys on upward movement, sells on downward movement.
    """

    def __init__(self):
        self.prev_price = None

    def act(self, state):
        current_price = state["price"]

        if self.prev_price is None:
            self.prev_price = current_price
            return 0  # HOLD on first step

        if current_price > self.prev_price:
            action = 1  # BUY
        elif current_price < self.prev_price:
            action = 2  # SELL
        else:
            action = 0  # HOLD

        self.prev_price = current_price
        return action
