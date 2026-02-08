import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backend.env.market_env import MarketEnvironment


class RLMarketEnv(gym.Env):
    """
    Gym-compatible wrapper around MarketEnvironment.
    """

    metadata = {"render.modes": []}

    def __init__(self, prices, reward_mode="raw"):
        super().__init__()
        self.env = MarketEnvironment(prices, reward_mode=reward_mode)

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

    @staticmethod
    def _state_to_array(state):
        return np.array(
            [
                state["price"],
                state["cash"],
                state["holdings"],
                state["portfolio_value"],
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        state = self.env.reset()
        return self._state_to_array(state), {}

    def step(self, action):
        state, reward, done = self.env.step(action)

        terminated = done
        truncated = False  # no time-limit truncation yet
        info = {}

        return (
            self._state_to_array(state),
            reward,
            terminated,
            truncated,
            info,
        )
