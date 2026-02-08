class MarketEnvironment:
    """
    A simple financial market environment for portfolio decision-making.
    """

    def __init__(
        self,
        prices,
        initial_cash=10000,
        trade_size=1,
        reward_mode="raw",
        drawdown_coeff=0.01,
        volatility_coeff=0.01,
        trade_penalty_coeff=0.0,
    ):
        self.holdings = None
        self.cash = None
        self.done = None
        self.timestep = None
        self.prices = prices
        self.initial_cash = initial_cash
        self.trade_size = trade_size
        self.state_dim = 4
        self.reward_mode = reward_mode
        self.drawdown_coeff = drawdown_coeff
        self.volatility_coeff = volatility_coeff
        self.trade_penalty_coeff = trade_penalty_coeff

        self.max_portfolio_value = None
        self.prev_price = None
        self.trade_count = None
        self.reset()

    def reset(self):
        self.timestep = 0
        self.cash = self.initial_cash
        self.holdings = 0
        self.done = False
        self.max_portfolio_value = self.initial_cash
        self.prev_price = self.prices[0]
        self.trade_count = 0
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
        traded = False
        if action == 1:  # BUY
            cost = current_price * self.trade_size
            if self.cash >= cost:
                self.cash -= cost
                self.holdings += self.trade_size
                traded = True

        elif action == 2:  # SELL
            if self.holdings >= self.trade_size:
                self.cash += current_price * self.trade_size
                self.holdings -= self.trade_size
                traded = True

        # Move to next timestep
        self.timestep += 1
        if self.timestep >= len(self.prices) - 1:
            self.done = True

        next_price = self.prices[self.timestep]
        current_value = self._portfolio_value(next_price)

        raw_reward = current_value - prev_value
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)

        if self.reward_mode == "risk_adjusted":
            drawdown = self.max_portfolio_value - current_value
            volatility = abs(next_price - current_price)
            trade_penalty = (
                self.trade_penalty_coeff if traded else 0.0
            )
            reward = (
                raw_reward
                - self.drawdown_coeff * drawdown
                - self.volatility_coeff * volatility
                - trade_penalty
            )
        else:
            reward = raw_reward

        if traded:
            self.trade_count += 1
        self.prev_price = next_price
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
    
