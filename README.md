# Prosperity Grove

Prosperity Grove is an AI-powered financial decision-making simulator focused on long-term wealth behavior and risk dynamics. It combines a simulation environment, baseline agents, PPO training, and a console-style UI for running and inspecting experiments.

**Highlights**
- Market regime simulator with risk-adjusted rewards
- Baselines: random, rule_based, buy_and_hold, PPO
- Regime-shift benchmark and reporting utilities
- Live console UI with streaming progress

**Repository layout**
- `backend/`: FastAPI service, agents, environments, simulations
- `frontend/`: Vite + React research console
- `experiments/`: scenarios, configs, and result tools
- `docs/`: benchmark and architecture notes

**Quickstart (local)**
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

**API**
- `POST /run-experiment` returns a full experiment result
- `POST /run-experiment/stream` streams progress + results (NDJSON)
- `GET /` health check

**Benchmarking**
See `docs/benchmark.md` for the full benchmark workflow, including reporting and plot generation. A one-shot script is available:
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
- Use Python 3.11 or 3.12 for best compatibility with `torch` and `stable-baselines3`.
