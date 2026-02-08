from pydantic import BaseModel
from typing import List, Optional


class ExperimentRequest(BaseModel):
    scenario: str
    agent_type: str
    episodes: int = 10
    timesteps: int = 10_000
    reward_mode: str = "raw"
    schedule: Optional[List[str]] = None
    schedule_length: int = 20


class ExperimentResponse(BaseModel):
    result: dict
