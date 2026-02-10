# Regime-Shift Robustness Benchmark

This benchmark evaluates agents under scheduled market regime shifts to
measure robustness and risk-adjusted performance.

## Protocol
1. Choose a regime schedule, e.g. ["bull", "volatile", "bear"].
2. Run each agent across N episodes and report mean/variance.
3. Compare both raw reward and risk-adjusted reward modes.

## Core Metrics
- Final portfolio value
- Max drawdown
- Volatility (std of returns)
- Sharpe ratio (normalized by episode length)
- Turnover (trade frequency)

## Example Schedules
- `regime_shift_short`: bull -> volatile -> bear
- `regime_shift_long`: bull -> sideways -> volatile -> bear

## Runner
Use the benchmark runner to export CSV for analysis:
```
python3 -m backend.simulations.run_benchmark --episodes 20 --timesteps 10000
```

Include the buy-and-hold baseline:
```
python3 -m backend.simulations.run_benchmark \
  --agents random,rule_based,buy_and_hold,ppo
```

You can increase reward shaping strength:
```
python3 -m backend.simulations.run_benchmark \
  --drawdown-coeff 0.02 --volatility-coeff 0.02
```

Penalize invalid actions (e.g., selling with no holdings):
```
python3 -m backend.simulations.run_benchmark \
  --invalid-action-penalty 0.1
```

Encourage trading (inactivity penalty):
```
python3 -m backend.simulations.run_benchmark \
  --inactivity-penalty 0.01
```

Encourage exploration (PPO entropy):
```
python3 -m backend.simulations.run_benchmark \
  --entropy-coef 0.01
```

Increase trade impact:
```
python3 -m backend.simulations.run_benchmark \
  --trade-size 50
```

## Recommended defaults (fast improvement)
These flags tend to produce non-degenerate behavior quickly:
```
python3 -m backend.simulations.run_benchmark \
  --drawdown-coeff 0.02 --volatility-coeff 0.02 \
  --invalid-action-penalty 0.5 \
  --inactivity-penalty 0.02 \
  --entropy-coef 0.05 \
  --trade-size 50
```

## Paper suite
Run a full sweep, plot, and table export in one command:
```
python3 experiments/run_paper_suite.py \
  --episodes 5 --ppo-repeats 5 --error-bars
```
Outputs are written to `experiments/results/`.
## Summaries + Plot
Summarize the raw CSV into mean/std tables:
```
python3 -m backend.simulations.summarize_results \
  --in experiments/results/benchmark_results.csv
```

Generate a bar chart (default):
```
python3 -m backend.simulations.plot_results \
  --in experiments/results/benchmark_summary.csv
```

Add error bars (std):
```
python3 -m backend.simulations.plot_results \
  --in experiments/results/benchmark_summary.csv \
  --error-bars
```

Generate a scatter plot if needed:
```
python3 -m backend.simulations.plot_results \
  --in experiments/results/benchmark_summary.csv \
  --kind scatter
```

If raw vs risk_adjusted overlaps, generate separate plots:
```
python3 -m backend.simulations.plot_results \
  --in experiments/results/benchmark_summary.csv \
  --split-modes
```

Generate a markdown table for the paper:
```
python3 -m backend.simulations.report_results \
  --in experiments/results/benchmark_summary.csv \
  --out experiments/results/benchmark_table.md
```

## PPO Raw vs Risk-Adjusted
Direct comparison for PPO:
```
python3 -m backend.simulations.run_ppo_compare \
  --scenario regime_shift_short \
  --timesteps 50000 \
  --repeats 3
```

Show PPO training progress:
```
python3 -m backend.simulations.run_ppo_compare \
  --scenario regime_shift_short \
  --timesteps 50000 \
  --repeats 3 \
  --progress --log-every 5000
```

Example PPO compare with exploration + penalties:
```
python3 -m backend.simulations.run_ppo_compare \
  --scenario regime_shift_short \
  --timesteps 20000 \
  --repeats 2 \
  --drawdown-coeff 0.05 \
  --volatility-coeff 0.05 \
  --invalid-action-penalty 0.1 \
  --inactivity-penalty 0.01 \
  --entropy-coef 0.01 \
  --progress --log-every 2000
```

## Real-world price CSV
Use a local CSV (e.g., exported from Yahoo Finance):
```
python3 -m backend.simulations.run_benchmark \
  --prices-csv data/SPY.csv \
  --price-col Close \
  --date-col Date \
  --train-ratio 0.7 \
  --agents random,rule_based,buy_and_hold,ppo \
  --episodes 10 --timesteps 20000 --ppo-repeats 1 \
  --out experiments/results/realdata_results.csv
```
