class MarketEnvironment:
    """
    A simple financial market environment for portfolio decision-making.
    """

    def __init__(self, prices, initial_cash=10000, trade_size=1):
        self.prices = prices
        self.initial_cash = initial_cash
        self.trade_size = trade_size
        self.reset()

    def reset(self):
        self.timestep = 0
        self.cash = self.initial_cash
        self.holdings = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        """
        Actions:
        0 = HOLD
        1 = BUY
        2 = SELL
        """
        if self.done:
            raise RuntimeError("Episode has ended. Call reset().")

        current_price = self.prices[self.timestep]
        prev_value = self._portfolio_value(current_price)

        # Execute action
        if action == 1:  # BUY
            cost = current_price * self.trade_size
            if self.cash >= cost:
                self.cash -= cost
                self.holdings += self.trade_size

        elif action == 2:  # SELL
            if self.holdings >= self.trade_size:
                self.cash += current_price * self.trade_size
                self.holdings -= self.trade_size

        # Move to next timestep
        self.timestep += 1
        if self.timestep >= len(self.prices) - 1:
            self.done = True

        next_price = self.prices[self.timestep]
        current_value = self._portfolio_value(next_price)

        reward = current_value - prev_value
        state = self._get_state()

        return state, reward, self.done

    def _portfolio_value(self, price):
        return self.cash + self.holdings * price

    def _get_state(self):
        price = self.prices[self.timestep]
        return {
            "timestep": self.timestep,
            "price": price,
            "cash": self.cash,
            "holdings": self.holdings,
            "portfolio_value": self._portfolio_value(price),
        }