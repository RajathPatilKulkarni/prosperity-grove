# Prosperity Grove

[![UI](https://img.shields.io/badge/UI-Netlify-00C7B7?logo=netlify&logoColor=white)](https://prosperity-grove.netlify.app)
[![API](https://img.shields.io/badge/API-Render-46E3B7?logo=render&logoColor=white)](https://prosperity-grove.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Prosperity Grove is an AI-powered financial decision-making simulator focused on long-term wealth behavior and risk dynamics. It combines a market simulator, baseline agents, PPO training, and a console-style UI for running and inspecting experiments.

**Core Objectives**
- Provide a controllable market simulator with multiple regimes.
- Compare baseline agents against PPO under raw and risk-adjusted rewards.
- Offer a reproducible benchmark pipeline for research reporting.
- Present results through a console UI with live progress and diagnostics.

**Key Capabilities**
- Market regime simulator with reward shaping and trade penalties.
- Agents: random, rule_based, buy_and_hold, PPO.
- Regime-shift benchmark + reporting utilities.
- Streaming experiment progress (NDJSON) in the UI.
- Live metrics, action mix, and trajectory visualization.

**Technology Stack**
- Backend: FastAPI, Python 3.11/3.12.
- RL: stable-baselines3 (PPO), gymnasium, torch.
- Data: numpy, pandas.
- Frontend: Vite, React, CSS animations.
- Deployment: Render (API) + Netlify (UI).

**UI Animations**
- Panel reveal: subtle fade + slide on load.
- Sparkline draw: animated path for the latest trajectory.
- Bar rise: action mix bars animate into place.
- Log fade: new log lines fade in for readability.

**Repository Layout**
- `backend/`: FastAPI service, agents, environments, simulations.
- `frontend/`: Vite + React research console.
- `experiments/`: scenarios, configs, and result tools.
- `docs/`: benchmark and architecture notes.

**Quickstart (Local)**
1. Backend
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
2. Frontend
```
cd frontend
npm install
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

**API Endpoints**
- `GET /` health check.
- `POST /run-experiment` returns a complete experiment result.
- `POST /run-experiment/stream` streams progress + result (NDJSON).

**Experiment Flow**
```mermaid
flowchart LR
  UI["Console UI"] -->|POST config| API["FastAPI"]
  API --> SIM["Experiment Runner"]
  SIM --> ENV["Market Environment"]
  SIM --> AGENT["Agent (Random/Rule/PPO)"]
  ENV --> SIM
  AGENT --> SIM
  SIM -->|metrics + trajectory| API
  API -->|NDJSON stream| UI
```

**Streaming Progress Flow**
```mermaid
sequenceDiagram
  participant UI as Console UI
  participant API as FastAPI
  participant PPO as PPO Trainer
  UI->>API: POST /run-experiment/stream (config)
  API->>PPO: start training
  loop progress
    PPO-->>API: step, pct
    API-->>UI: NDJSON {type:"progress", pct}
  end
  PPO-->>API: result
  API-->>UI: NDJSON {type:"result"}
```

**Benchmark Pipeline**
```mermaid
flowchart LR
  RUN["run_benchmark.py"] --> CSV["benchmark_results.csv"]
  CSV --> SUM["summarize_results.py"]
  SUM --> OUT["benchmark_summary.csv"]
  OUT --> PLOT["plot_results.py"]
  OUT --> TABLE["report_results.py"]
```

**Visual System Maps (Unique)**
```mermaid
flowchart TB
  classDef console fill:#0b120e,stroke:#39ff88,color:#b6ffd6,stroke-width:1.5px;
  classDef engine fill:#0b0f0d,stroke:#6fe3a1,color:#b6ffd6;
  classDef data fill:#060807,stroke:#7fcfa1,color:#b6ffd6;

  UI["Research Console"]:::console
  CTRL["Experiment Control"]:::console
  LOGS["Execution Log"]:::console
  METRICS["Metrics + Charts"]:::console

  API["FastAPI Stream"]:::engine
  SIM["Simulation Engine"]:::engine
  ENV["Market Regimes"]:::engine
  AGENT["Agents (Random / Rule / PPO)"]:::engine

  DATA["Results Store"]:::data
  REPORT["Report + Plots"]:::data

  UI --> CTRL --> API
  API --> SIM --> ENV
  SIM --> AGENT --> SIM
  SIM --> DATA --> REPORT --> UI
  API --> LOGS
  API --> METRICS
```

```mermaid
stateDiagram-v2
  [*] --> Configure
  Configure --> Validate
  Validate --> Run
  Run --> Stream
  Stream --> Render
  Render --> Complete
  Complete --> [*]

  state Run {
    [*] --> "Reward (Raw)"
    "Reward (Raw)" --> "Reward (Risk Adjusted)"
  }
```

**Benchmarking**
See `docs/benchmark.md` for the full workflow, including reporting and plot generation.
```
python3 experiments/run_paper_suite.py --episodes 5 --ppo-repeats 5 --error-bars
```

**Deployment (Render + Netlify)**
1. Render (backend)
- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
2. Netlify (frontend)
- Base directory: `frontend`
- Build command: `npm run build`
- Publish directory: `dist`
- Env var: `VITE_API_BASE=https://<your-render-app>.onrender.com`

**Notes**
- Use Python 3.11 or 3.12 for compatibility with `torch` and `stable-baselines3`.
