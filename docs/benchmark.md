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
