from pydantic import BaseModel


class ExperimentRequest(BaseModel):
    scenario: str
    agent_type: str
    episodes: int = 10
    timesteps: int = 10_000


class ExperimentResponse(BaseModel):
    result: dict
