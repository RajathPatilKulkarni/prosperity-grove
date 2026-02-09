from fastapi import FastAPI, HTTPException

from experiments.experiment_config import ExperimentConfig
from backend.simulations.run_simulation import run_configured_experiment
from backend.api.schemas import ExperimentRequest, ExperimentResponse

app = FastAPI(
    title="Prosperity Grove API",
    description="API for running financial simulations and RL experiments",
    version="1.0",
)


@app.get("/")
def root():
    return {"status": "Prosperity Grove API running"}


@app.post("/run-experiment", response_model=ExperimentResponse)
def run_experiment_api(request: ExperimentRequest):
    try:
        config = ExperimentConfig(
            scenario=request.scenario,
            agent_type=request.agent_type,
            episodes=request.episodes,
            timesteps=request.timesteps,
            reward_mode=request.reward_mode,
            schedule=request.schedule,
            schedule_length=request.schedule_length,
            drawdown_coeff=request.drawdown_coeff,
            volatility_coeff=request.volatility_coeff,
            trade_penalty_coeff=request.trade_penalty_coeff,
            invalid_action_penalty=request.invalid_action_penalty,
            inactivity_penalty=request.inactivity_penalty,
        )

        result = run_configured_experiment(config)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
