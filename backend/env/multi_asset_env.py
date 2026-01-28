class MultiAssetMarketEnvironment:
    """
    Multi-asset financial environment for portfolio allocation.
    """

    ACTION_HOLD = 0

    def __init__(self, price_matrix, initial_cash=10000, trade_size=1):
        """
        price_matrix: list of lists
            shape = (num_assets, timesteps)
        """
        self.price_matrix = price_matrix
        self.num_assets = len(price_matrix)
        self.timesteps = len(price_matrix[0])

        self.initial_cash = initial_cash
        self.trade_size = trade_size

        # action space: HOLD + (BUY, SELL) per asset
        self.action_space_size = 1 + 2 * self.num_assets

        # state: prices + holdings + cash + portfolio_value
        self.state_dim = 2 * self.num_assets + 2

        self.reset()

    def reset(self):
        self.timestep = 0
        self.cash = self.initial_cash
        self.holdings = [0 for _ in range(self.num_assets)]
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        prev_value = self._portfolio_value()

        # Decode action
        if action != self.ACTION_HOLD:
            asset_id = (action - 1) // 2
            is_buy = (action - 1) % 2 == 0

            price = self.price_matrix[asset_id][self.timestep]

            if is_buy:
                cost = price * self.trade_size
                if self.cash >= cost:
                    self.cash -= cost
                    self.holdings[asset_id] += self.trade_size
            else:
                if self.holdings[asset_id] >= self.trade_size:
                    self.cash += price * self.trade_size
                    self.holdings[asset_id] -= self.trade_size

        # Advance time
        self.timestep += 1
        if self.timestep >= self.timesteps - 1:
            self.done = True

        reward = self._portfolio_value() - prev_value
        return self._get_state(), reward, self.done

    def _portfolio_value(self):
        value = self.cash
        for i in range(self.num_assets):
            price = self.price_matrix[i][self.timestep]
            value += self.holdings[i] * price
        return value

    def _get_state(self):
        state = []
        for i in range(self.num_assets):
            state.append(self.price_matrix[i][self.timestep])
            state.append(self.holdings[i])

        state.append(self.cash)
        state.append(self._portfolio_value())
        return state