from stable_baselines3 import PPO
from backend.env.rl_env import RLMarketEnv


def train_ppo(prices, timesteps=10_000):
    env = RLMarketEnv(prices)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=42,
    )

    model.learn(total_timesteps=timesteps)
    return model