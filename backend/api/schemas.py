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
    drawdown_coeff: float = 0.01
    volatility_coeff: float = 0.01
    trade_penalty_coeff: float = 0.0
    invalid_action_penalty: float = 0.0
    inactivity_penalty: float = 0.0
    trade_size: int = 1


class ExperimentResponse(BaseModel):
    result: dict
