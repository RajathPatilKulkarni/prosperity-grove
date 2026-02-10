from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from backend.env.rl_env import RLMarketEnv


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, log_every=5000, label="PPO"):
        super().__init__()
        self.total_timesteps = max(1, int(total_timesteps))
        self.log_every = max(1, int(log_every))
        self.next_log = self.log_every
        self.label = label

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_log:
            pct = (self.num_timesteps / self.total_timesteps) * 100.0
            print(
                f"[{self.label}] {self.num_timesteps}/{self.total_timesteps} "
                f"({pct:.1f}%)"
            )
            self.next_log += self.log_every
        return True


def train_ppo(
    prices,
    timesteps=10_000,
    reward_mode="raw",
    drawdown_coeff=0.01,
    volatility_coeff=0.01,
    trade_penalty_coeff=0.0,
    invalid_action_penalty=0.0,
    inactivity_penalty=0.0,
    trade_size=1,
    entropy_coef=0.0,
    progress=False,
    log_every=5000,
    progress_label="PPO",
):
    env = RLMarketEnv(
        prices,
        reward_mode=reward_mode,
        drawdown_coeff=drawdown_coeff,
        volatility_coeff=volatility_coeff,
        trade_penalty_coeff=trade_penalty_coeff,
        invalid_action_penalty=invalid_action_penalty,
        inactivity_penalty=inactivity_penalty,
        trade_size=trade_size,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=42,
        ent_coef=entropy_coef,
    )

    callback = None
    if progress:
        callback = ProgressCallback(
            timesteps, log_every=log_every, label=progress_label
        )

    model.learn(total_timesteps=timesteps, callback=callback)
    return model
