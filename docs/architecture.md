# Architecture Overview

Prosperity Grove is a modular system that separates simulation, agent logic, API orchestration, and the UI.

**Core modules**
- `backend/env/`: market environment and reward shaping
- `backend/agents/`: random, rule_based, buy_and_hold, PPO
- `backend/simulations/`: experiment runners, benchmarks, reporting
- `backend/app/`: FastAPI service and streaming endpoint
- `frontend/`: console UI and visual diagnostics

**Data flow**
1. UI sends a configuration to the FastAPI service.
2. The API builds an `ExperimentConfig` and dispatches a run.
3. Simulations return metrics, trajectories, and action traces.
4. The UI renders metrics, charts, and logs.

**Streaming progress**
For PPO runs, the API streams progress events via `POST /run-experiment/stream`. The UI listens to the NDJSON stream to update the execution log and progress bar in real time.
