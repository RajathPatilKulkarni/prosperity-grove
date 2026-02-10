from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import queue
import threading

from experiments.experiment_config import ExperimentConfig
from backend.simulations.run_simulation import run_configured_experiment
from backend.api.schemas import ExperimentRequest, ExperimentResponse

app = FastAPI(
    title="Prosperity Grove API",
    description="API for running financial simulations and RL experiments",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://prosperity-grove.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            trade_size=request.trade_size,
        )

        result = run_configured_experiment(config)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.options("/run-experiment")
def run_experiment_options():
    return Response(status_code=204)


@app.post("/run-experiment/stream")
def run_experiment_stream(request: ExperimentRequest):
    event_queue = queue.Queue()

    def emit(event):
        event_queue.put(event)

    def progress_hook(step, total, pct):
        emit({"type": "progress", "pct": pct})

    def worker():
        try:
            emit({"type": "log", "message": "> run started"})
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
                trade_size=request.trade_size,
            )
            if request.agent_type != "ppo":
                emit({"type": "progress", "pct": 0.0})
            result = run_configured_experiment(
                config,
                progress=True,
                log_every=1000,
                progress_hook=progress_hook,
            )
            emit({"type": "progress", "pct": 100.0})
            emit({"type": "result", "result": result})
        except Exception as exc:
            emit({"type": "error", "message": str(exc)})
        finally:
            emit({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            event = event_queue.get()
            if event.get("type") == "done":
                break
            yield f"{json.dumps(event)}\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.options("/run-experiment/stream")
def run_experiment_stream_options():
    return Response(status_code=204)
