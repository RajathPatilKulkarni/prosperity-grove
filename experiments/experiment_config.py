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
    ):
        self.scenario = scenario
        self.agent_type = agent_type
        self.episodes = episodes
        self.timesteps = timesteps

    def __repr__(self):
        return (
            f"ExperimentConfig("
            f"scenario={self.scenario}, "
            f"agent={self.agent_type}, "
            f"episodes={self.episodes}, "
            f"timesteps={self.timesteps})"
        )