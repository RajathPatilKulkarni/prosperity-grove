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
    ):
        self.scenario = scenario
        self.agent_type = agent_type
        self.episodes = episodes
        self.timesteps = timesteps
        self.reward_mode = reward_mode
        self.schedule = schedule
        self.schedule_length = schedule_length

    def __repr__(self):
        return (
            f"ExperimentConfig("
            f"scenario={self.scenario}, "
            f"agent={self.agent_type}, "
            f"episodes={self.episodes}, "
            f"timesteps={self.timesteps}, "
            f"reward_mode={self.reward_mode}, "
            f"schedule={self.schedule}, "
            f"schedule_length={self.schedule_length})"
        )
