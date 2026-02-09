class ExperimentConfig:
    """
    Configuration object for running experiments.
    """

    def __init__(
        self,
        scenario: str,
        agent_type: str,
        episodes: int = 10,
        timesteps: int = 10_000,
        reward_mode: str = "raw",
        schedule: list | None = None,
        schedule_length: int = 20,
        drawdown_coeff: float = 0.01,
        volatility_coeff: float = 0.01,
        trade_penalty_coeff: float = 0.0,
        invalid_action_penalty: float = 0.0,
        inactivity_penalty: float = 0.0,
    ):
        self.scenario = scenario
        self.agent_type = agent_type
        self.episodes = episodes
        self.timesteps = timesteps
        self.reward_mode = reward_mode
        self.schedule = schedule
        self.schedule_length = schedule_length
        self.drawdown_coeff = drawdown_coeff
        self.volatility_coeff = volatility_coeff
        self.trade_penalty_coeff = trade_penalty_coeff
        self.invalid_action_penalty = invalid_action_penalty
        self.inactivity_penalty = inactivity_penalty

    def __repr__(self):
        return (
            f"ExperimentConfig("
            f"scenario={self.scenario}, "
            f"agent={self.agent_type}, "
            f"episodes={self.episodes}, "
            f"timesteps={self.timesteps}, "
            f"reward_mode={self.reward_mode}, "
            f"schedule={self.schedule}, "
            f"schedule_length={self.schedule_length}, "
            f"drawdown_coeff={self.drawdown_coeff}, "
            f"volatility_coeff={self.volatility_coeff}, "
            f"trade_penalty_coeff={self.trade_penalty_coeff}, "
            f"invalid_action_penalty={self.invalid_action_penalty}, "
            f"inactivity_penalty={self.inactivity_penalty})"
        )
